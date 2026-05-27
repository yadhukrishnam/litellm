"""
Custom LiteLLM provider that wraps the Claude Agent SDK (claude-agent-sdk).
Uses the locally authenticated Claude Code CLI — no Anthropic API billing required.

Flow: LiteLLM /chat/completions → this provider → claude_agent_sdk → Claude (Pro plan)

Performance features:
- Fix 2: setting_sources=["user"] skips CLAUDE.md/project-settings loading per request
- Fix 3: StreamEvent parsing yields individual text deltas for true token streaming
- Fix 4: ClaudeProcessPool keeps N warm subprocesses; requests skip cold-start entirely
"""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Iterator, Optional, Union

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk, ModelResponse

logger = logging.getLogger(__name__)

_POOL_SIZE = int(os.environ.get("CLAUDE_POOL_SIZE", "2"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_system_and_prompt(messages: list) -> tuple[Optional[str], str]:
    """Convert OpenAI messages list to (system_prompt, conversation_prompt)."""
    system: Optional[str] = None
    history: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            )

        if role == "system":
            system = content
        elif role == "user":
            history.append(f"Human: {content}")
        elif role == "assistant":
            history.append(f"Assistant: {content}")

    return system, "\n\n".join(history)


def _embed_system(system: Optional[str], prompt: str) -> str:
    """Inline system content into the prompt (used by pool path, no system arg at connect time)."""
    if not system:
        return prompt
    return f"<system>\n{system}\n</system>\n\n{prompt}"


def _model_name(model: str) -> Optional[str]:
    """Strip the provider prefix from a model string, e.g. 'claude_code/sonnet' → 'sonnet'."""
    return model.split("/", 1)[1] if "/" in model else model or None


def _apply_usage(model_response: ModelResponse, usage: dict) -> None:
    input_tokens = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    model_response.usage.prompt_tokens = input_tokens
    model_response.usage.completion_tokens = output_tokens
    model_response.usage.total_tokens = input_tokens + output_tokens


# ---------------------------------------------------------------------------
# Cold-path query (sync fallback and sync completion/streaming)
# ---------------------------------------------------------------------------

async def _run_query(
    prompt: str,
    system: Optional[str],
    model: Optional[str],
    stream: bool,
) -> AsyncIterator:
    """Async generator that yields SDK events; used by sync paths and as fallback."""
    from claude_agent_sdk import query
    from claude_agent_sdk.types import ClaudeAgentOptions, SystemPromptFile

    tmp_path: Optional[str] = None
    try:
        system_prompt_arg = None
        if system:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(system)
                tmp_path = tmp.name
            system_prompt_arg = SystemPromptFile(type="file", path=tmp_path)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt_arg,
            model=model,
            tools=[],
            permission_mode="dontAsk",
            include_partial_messages=stream,
            setting_sources=["user"],           # Fix 2: skip project/CLAUDE.md loading
            extra_args={"no-session-persistence": None},  # Fix 2: skip session file I/O
        )

        async def _stdin_prompt():
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": None,
            }

        async for event in query(prompt=_stdin_prompt(), options=options):
            yield event

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Fix 4: Pre-warmed process pool
# ---------------------------------------------------------------------------

class ClaudeProcessPool:
    """
    Maintains N warm ClaudeSDKClient subprocesses. Each request pulls a client,
    uses it for one query, then discards it (conversation history would accumulate
    otherwise). The pool immediately refills in the background.
    """

    def __init__(self, size: int = _POOL_SIZE) -> None:
        self._size = size
        self._queue: asyncio.Queue = asyncio.Queue()

    async def _make_client(self):
        from claude_agent_sdk import ClaudeSDKClient
        from claude_agent_sdk.types import ClaudeAgentOptions
        options = ClaudeAgentOptions(
            tools=[],
            permission_mode="dontAsk",
            setting_sources=["user"],
            extra_args={"no-session-persistence": None},
            include_partial_messages=True,  # always on; non-streaming paths ignore StreamEvents
        )
        client = ClaudeSDKClient(options)
        await client.connect()
        return client

    async def start(self) -> None:
        for _ in range(self._size):
            self._queue.put_nowait(await self._make_client())

    @asynccontextmanager
    async def acquire(self, model: Optional[str] = None):
        client = await self._queue.get()
        asyncio.get_event_loop().create_task(self._refill())
        try:
            if model:
                await client.set_model(model)
            yield client
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

    async def _refill(self) -> None:
        try:
            self._queue.put_nowait(await self._make_client())
        except Exception as exc:
            logger.warning("ClaudeProcessPool: refill failed: %s", exc)


_pool: Optional[ClaudeProcessPool] = None


async def _start_pool() -> None:
    global _pool
    try:
        pool = ClaudeProcessPool()
        await pool.start()
        _pool = pool
        logger.info("ClaudeProcessPool: %d warm clients ready", _POOL_SIZE)
    except Exception as exc:
        logger.warning("ClaudeProcessPool: startup failed, will use cold queries: %s", exc)


# Warm the pool when the proxy imports this module (event loop is already running).
try:
    _loop = asyncio.get_event_loop()
    if _loop.is_running():
        _loop.create_task(_start_pool())
    else:
        _loop.run_until_complete(_start_pool())
except RuntimeError:
    pass  # no event loop at import time — pool stays None


# ---------------------------------------------------------------------------
# Streaming chunk builder helpers (Fix 3)
# ---------------------------------------------------------------------------

def _stream_event_to_chunk(raw: dict) -> Optional[GenericStreamingChunk]:
    """Extract a text delta chunk from a raw Anthropic stream event dict, or None."""
    if (
        raw.get("type") == "content_block_delta"
        and raw.get("delta", {}).get("type") == "text_delta"
    ):
        text = raw["delta"].get("text", "")
        if text:
            return {"text": text, "is_finished": False, "finish_reason": "", "usage": None, "index": 0}
    return None


def _result_to_chunk(event) -> GenericStreamingChunk:
    usage = event.usage or {}
    input_tokens = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "text": "",
        "is_finished": True,
        "finish_reason": event.stop_reason or "stop",
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "index": 0,
    }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ClaudeCodeProvider(CustomLLM):
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float]] = None,
        client=None,
    ) -> ModelResponse:
        system, prompt = _extract_system_and_prompt(messages)

        async def _run():
            from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

            text_parts: list[str] = []
            result_msg = None

            async for event in _run_query(
                prompt=prompt, system=system, model=_model_name(model), stream=False
            ):
                if isinstance(event, AssistantMessage):
                    for block in event.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(event, ResultMessage):
                    result_msg = event

            return "".join(text_parts), result_msg

        try:
            text, result_msg = asyncio.run(_run())
        except Exception as e:
            raise CustomLLMError(status_code=500, message=str(e))

        if not text and result_msg and result_msg.is_error:
            raise CustomLLMError(
                status_code=500,
                message=result_msg.result or "Claude Agent SDK returned an error",
            )

        model_response.choices[0].message.content = text
        model_response.choices[0].finish_reason = (
            result_msg.stop_reason if result_msg else "stop"
        )
        if result_msg and result_msg.usage:
            _apply_usage(model_response, result_msg.usage)

        return model_response

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        system, prompt = _extract_system_and_prompt(messages)

        async def _collect() -> list[GenericStreamingChunk]:
            from claude_agent_sdk.types import ResultMessage, StreamEvent

            chunks: list[GenericStreamingChunk] = []
            async for event in _run_query(
                prompt=prompt, system=system, model=_model_name(model), stream=True
            ):
                if isinstance(event, StreamEvent):
                    chunk = _stream_event_to_chunk(event.event)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(event, ResultMessage):
                    chunks.append(_result_to_chunk(event))
            return chunks

        try:
            yield from asyncio.run(_collect())
        except Exception as e:
            raise CustomLLMError(status_code=500, message=str(e))

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> ModelResponse:
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

        system, prompt = _extract_system_and_prompt(messages)
        text_parts: list[str] = []
        result_msg = None

        if _pool is not None:
            # Fix 4: use warm pool client
            embedded_prompt = _embed_system(system, prompt)
            async with _pool.acquire(model=_model_name(model)) as pool_client:
                await pool_client.query(embedded_prompt)
                async for event in pool_client.receive_response():
                    if isinstance(event, AssistantMessage):
                        for block in event.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                    elif isinstance(event, ResultMessage):
                        result_msg = event
        else:
            # Fallback: cold query
            async for event in _run_query(
                prompt=prompt, system=system, model=_model_name(model), stream=False
            ):
                if isinstance(event, AssistantMessage):
                    for block in event.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(event, ResultMessage):
                    result_msg = event

        text = "".join(text_parts)

        if not text and result_msg and result_msg.is_error:
            raise CustomLLMError(
                status_code=500,
                message=result_msg.result or "Claude Agent SDK returned an error",
            )

        model_response.choices[0].message.content = text
        model_response.choices[0].finish_reason = (
            result_msg.stop_reason if result_msg else "stop"
        )
        if result_msg and result_msg.usage:
            _apply_usage(model_response, result_msg.usage)

        return model_response

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout=None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        from claude_agent_sdk.types import ResultMessage, StreamEvent

        system, prompt = _extract_system_and_prompt(messages)

        if _pool is not None:
            # Fix 4: use warm pool client (always has include_partial_messages=True)
            embedded_prompt = _embed_system(system, prompt)
            async with _pool.acquire(model=_model_name(model)) as pool_client:
                await pool_client.query(embedded_prompt)
                async for event in pool_client.receive_response():
                    if isinstance(event, StreamEvent):
                        chunk = _stream_event_to_chunk(event.event)
                        if chunk:
                            yield chunk
                    elif isinstance(event, ResultMessage):
                        yield _result_to_chunk(event)
        else:
            # Fallback: cold query with Fix 3 StreamEvent handling
            async for event in _run_query(
                prompt=prompt, system=system, model=_model_name(model), stream=True
            ):
                if isinstance(event, StreamEvent):
                    chunk = _stream_event_to_chunk(event.event)
                    if chunk:
                        yield chunk
                elif isinstance(event, ResultMessage):
                    yield _result_to_chunk(event)


# The instance LiteLLM references via custom_provider_map
claude_code_handler = ClaudeCodeProvider()
