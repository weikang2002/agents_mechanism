"""LiteLLM wrapper, message builders and tool-call helpers."""

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import litellm

from tools import ToolDefinition

Message = dict[str, Any]

# Tool-call-only steps never need long outputs; the cap stops degenerate token loops.
# Only applied to non-reasoning models since reasoning tokens count against max_tokens.
TOOL_STEP_MAX_TOKENS = 1024

LLM_STEP_TIMEOUT_S = 300
LLM_REPORT_TIMEOUT_S = 900


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.args)},
        }


@dataclass
class StepResult:
    content: str | None
    tool_calls: list[ToolCall]


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class LLMClient:
    def __init__(self, model: str, is_reasoning_model: bool | None) -> None:
        self.model = model
        self.is_reasoning_model = (
            is_reasoning_model
            if is_reasoning_model is not None
            else self._detect_reasoning_model(model)
        )
        self.max_input_tokens = self._lookup_max_input_tokens(model)

    @staticmethod
    def _detect_reasoning_model(model: str) -> bool:
        try:
            return bool(litellm.supports_reasoning(model=model))
        except Exception:
            return False

    @staticmethod
    def _lookup_max_input_tokens(model: str) -> int | None:
        try:
            value = litellm.get_model_info(model).get("max_input_tokens")
        except Exception:
            return None
        return int(value) if value else None

    def tool_step(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: str,
    ) -> StepResult:
        kwargs: dict[str, Any] = {}
        if not self.is_reasoning_model:
            kwargs["max_tokens"] = TOOL_STEP_MAX_TOKENS
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            timeout=LLM_STEP_TIMEOUT_S,
            **kwargs,
        )
        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=tool_call.id or f"call_{uuid.uuid4().hex[:24]}",
                name=tool_call.function.name,
                args=_parse_tool_args(tool_call.function.arguments),
            )
            for tool_call in (message.tool_calls or [])
        ]
        return StepResult(content=message.content, tool_calls=tool_calls)

    def stream_text(
        self,
        messages: list[Message],
        max_tokens: int | None,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=True,
            timeout=LLM_REPORT_TIMEOUT_S,
            **kwargs,
        )
        parts: list[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                parts.append(delta)
                if on_delta:
                    on_delta(delta)
        return "".join(parts)


def system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant_tool_call_message(tool_calls: list[ToolCall]) -> Message:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call.to_openai_format() for tool_call in tool_calls],
    }


def tool_response_message(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def last_n_user_messages(history: list[Message], n: int) -> list[Message]:
    """Keep everything from the n-th most recent user message onward."""
    seen = 0
    for index in range(len(history) - 1, -1, -1):
        if history[index]["role"] == "user":
            seen += 1
            if seen == n:
                return history[index:]
    return history


def find_tool_call(tool_calls: list[ToolCall], name: str) -> ToolCall | None:
    return next((tc for tc in tool_calls if tc.name == name), None)
