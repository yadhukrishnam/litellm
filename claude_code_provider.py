"""
Custom LiteLLM provider that wraps the Claude Agent SDK (claude-agent-sdk).
Uses the locally authenticated Claude Code CLI — no Anthropic API billing required.

Flow: LiteLLM /chat/completions → this provider → claude_agent_sdk → Claude (Pro plan)

Performance features:
- Fix 2: setting_sources=["user"] skips CLAUDE.md/project-settings loading per request
- Fix 3: StreamEvent parsing yields individual text deltas for true token streaming
- Fix 4: ClaudeProcessPool keeps N warm subprocesses; requests skip cold-start entirely
- Fix 5: Tool call simulation — serializes OpenAI tool schemas into the system prompt
          and parses TOOL_CALL XML tags back out of Claude's text responses.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Iterator, Optional, Union

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk, ModelResponse

logger = logging.getLogger(__name__)

_TOOL_CALL_TAG = "TOOL_CALL"

_POOL_SIZE = int(os.environ.get("CLAUDE_POOL_SIZE", "2"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tools_to_system_injection(tools: list) -> str:
    """Serialize OpenAI tool schemas into a system prompt block Claude can follow."""
    if not tools:
        return ""
    lines = [
        "\n\n---\n## Available Tools\n\n",
        f"To call a tool, emit a JSON block wrapped in `<{_TOOL_CALL_TAG}>` tags "
        f"(exactly as shown, one tool call per response):\n\n",
        f"<{_TOOL_CALL_TAG}>\n"
        f'  {{"name": "tool_name", "parameters": {{...}}}}\n'
        f"</{_TOOL_CALL_TAG}>\n\n",
        "Wait for the tool result before calling another tool. "
        "Tool results will appear as `[Tool result]: ...` messages.\n\n",
    ]
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = json.dumps(fn.get("parameters", {}), indent=2)
        lines.append(f"### {name}\n{desc}\nParameters:\n```json\n{params}\n```\n\n")
    return "".join(lines)


def _extract_system_and_prompt(
    messages: list, tools: Optional[list] = None
) -> tuple[Optional[str], str]:
    """Convert OpenAI messages list to (system_prompt, conversation_prompt).

    Handles system, user, assistant, and tool (result) roles.
    Serializes tool schemas into the system prompt when provided.
    """
    system: Optional[str] = None
    history: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system = content
            continue

        # Flatten list-type content (multipart messages)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            )

        if role == "user":
            history.append(f"Human: {content}")
        elif role == "assistant":
            # Reconstruct assistant turn: text + any tool calls it made
            parts: list[str] = []
            if content:
                parts.append(content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    params = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    params = {}
                parts.append(
                    f"<{_TOOL_CALL_TAG}>\n"
                    + json.dumps({"name": fn.get("name", ""), "parameters": params})
                    + f"\n</{_TOOL_CALL_TAG}>"
                )
            history.append(f"Assistant: {'  '.join(parts)}")
        elif role == "tool":
            # Tool result — associate with the preceding assistant turn
            tool_call_id = msg.get("tool_call_id", "")
            history.append(f"Human: [Tool result{' id=' + tool_call_id if tool_call_id else ''}]: {content}")

    if tools:
        system = (system or "") + _tools_to_system_injection(tools)

    return system, "\n\n".join(history)


def _parse_tool_calls(text: str) -> tuple[str, list]:
    """Extract TOOL_CALL blocks from text; returns (remaining_text, openai_tool_calls)."""
    tool_calls: list[dict] = []
    pattern = rf"<{_TOOL_CALL_TAG}>(.*?)</{_TOOL_CALL_TAG}>"

    def _extract(match: re.Match) -> str:
        try:
            data = json.loads(match.group(1).strip())
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(
                        data.get("parameters", data.get("arguments", {}))
                    ),
                },
            })
        except (json.JSONDecodeError, AttributeError):
            pass
        return ""

    remaining = re.sub(pattern, _extract, text, flags=re.DOTALL).strip()
    return remaining, tool_calls


def _apply_tool_calls(model_response: ModelResponse, text: str) -> None:
    """Parse tool calls from text and populate model_response accordingly."""
    remaining, tool_calls = _parse_tool_calls(text)
    if tool_calls:
        model_response.choices[0].message.tool_calls = tool_calls
        model_response.choices[0].message.content = remaining or None
        model_response.choices[0].finish_reason = "tool_calls"
    else:
        model_response.choices[0].message.content = text


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
        tools = optional_params.get("tools") or []
        system, prompt = _extract_system_and_prompt(messages, tools=tools)

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

        _apply_tool_calls(model_response, text)
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
        tools = optional_params.get("tools") or []
        system, prompt = _extract_system_and_prompt(messages, tools=tools)

        async def _collect() -> list[GenericStreamingChunk]:
            from claude_agent_sdk.types import ResultMessage, StreamEvent

            raw_chunks: list[GenericStreamingChunk] = []
            result_chunk: Optional[GenericStreamingChunk] = None
            async for event in _run_query(
                prompt=prompt, system=system, model=_model_name(model), stream=True
            ):
                if isinstance(event, StreamEvent):
                    chunk = _stream_event_to_chunk(event.event)
                    if chunk:
                        raw_chunks.append(chunk)
                elif isinstance(event, ResultMessage):
                    result_chunk = _result_to_chunk(event)

            # Reassemble full text to extract any tool calls before streaming
            full_text = "".join(c["text"] for c in raw_chunks if c.get("text"))
            remaining, tool_calls = _parse_tool_calls(full_text)

            if tool_calls:
                # Emit tool call deltas instead of text chunks
                chunks: list[GenericStreamingChunk] = []
                for i, tc in enumerate(tool_calls):
                    chunks.append({
                        "text": "",
                        "is_finished": False,
                        "finish_reason": "",
                        "usage": None,
                        "index": 0,
                        "tool_use": {
                            "id": tc["id"],
                            "type": "function",
                            "function": tc["function"],
                        },
                    })
                if result_chunk:
                    result_chunk["finish_reason"] = "tool_calls"
                    chunks.append(result_chunk)
                return chunks

            # No tool calls — stream text chunks as-is
            out = list(raw_chunks)
            if result_chunk:
                out.append(result_chunk)
            return out

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

        tools = optional_params.get("tools") or []
        system, prompt = _extract_system_and_prompt(messages, tools=tools)
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

        _apply_tool_calls(model_response, text)
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

        tools = optional_params.get("tools") or []
        system, prompt = _extract_system_and_prompt(messages, tools=tools)

        raw_chunks: list[GenericStreamingChunk] = []
        result_chunk: Optional[GenericStreamingChunk] = None

        async def _collect_events(source):
            nonlocal result_chunk
            async for event in source:
                if isinstance(event, StreamEvent):
                    chunk = _stream_event_to_chunk(event.event)
                    if chunk:
                        raw_chunks.append(chunk)
                elif isinstance(event, ResultMessage):
                    result_chunk = _result_to_chunk(event)

        if _pool is not None:
            # Fix 4: use warm pool client (always has include_partial_messages=True)
            embedded_prompt = _embed_system(system, prompt)
            async with _pool.acquire(model=_model_name(model)) as pool_client:
                await pool_client.query(embedded_prompt)
                await _collect_events(pool_client.receive_response())
        else:
            await _collect_events(
                _run_query(prompt=prompt, system=system, model=_model_name(model), stream=True)
            )

        full_text = "".join(c["text"] for c in raw_chunks if c.get("text"))
        remaining, tool_calls = _parse_tool_calls(full_text)

        if tool_calls:
            for tc in tool_calls:
                yield {
                    "text": "",
                    "is_finished": False,
                    "finish_reason": "",
                    "usage": None,
                    "index": 0,
                    "tool_use": {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    },
                }
            if result_chunk:
                result_chunk["finish_reason"] = "tool_calls"
                yield result_chunk
        else:
            for chunk in raw_chunks:
                yield chunk
            if result_chunk:
                yield result_chunk


# The instance LiteLLM references via custom_provider_map
claude_code_handler = ClaudeCodeProvider()
