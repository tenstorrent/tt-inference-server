# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Dependency-light token accounting for mini-swe-agent requests."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InputTokenBudgetExceeded(RuntimeError):
    """The complete, untruncated agent history exceeds its declared budget."""


class TokenBudgetConfigurationError(RuntimeError):
    """The configured tokenizer cannot authoritatively count the API input."""


def _normalize_tools_for_chat_template(tools: list[dict]) -> list[dict]:
    """Return Hugging Face tool schemas without changing the API payload.

    mini-swe-agent sends tools to LiteLLM in OpenAI's wrapper form::

        {"type": "function", "function": {"name": ..., "parameters": ...}}

    Hugging Face chat templates consume the function JSON schema itself.  In
    particular, Qwen3.6 iterates the schema's ``parameters.properties`` and
    raises inside Jinja when handed the outer OpenAI wrapper.  Token counting
    is a separate, local render; normalize a deep copy for that render only and
    leave the request passed to LiteLLM byte-for-byte unchanged.
    """
    rendered_tools: list[dict] = []
    for tool in deepcopy(tools):
        if not isinstance(tool, dict):
            raise TokenBudgetConfigurationError(
                "tool schema must be an object for tokenizer rendering"
            )
        if "function" in tool or "type" in tool:
            if tool.get("type") != "function" or not isinstance(
                tool.get("function"), dict
            ):
                raise TokenBudgetConfigurationError(
                    "OpenAI tool schema must contain type='function' and a "
                    "function object"
                )
            rendered_tools.append(tool["function"])
        else:
            rendered_tools.append(tool)
    return rendered_tools


def _normalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
    """Return a template-safe copy without changing the API request history.

    OpenAI-compatible tool-call responses may represent absent assistant text as
    ``content: null``. Some otherwise valid Hugging Face chat templates perform
    string operations whenever the key is present and therefore cannot render
    that representation. Empty text is token-equivalent here and lets the
    authoritative tokenizer count the complete request, including tool calls.
    """
    normalized = deepcopy(messages)
    for rendered_message in normalized:
        for text_field in ("content", "thinking"):
            if (
                rendered_message.get(text_field) is None
                and text_field in rendered_message
            ):
                rendered_message[text_field] = ""
        for tool_call in rendered_message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise TokenBudgetConfigurationError(
                        "tool-call arguments are not valid JSON"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise TokenBudgetConfigurationError(
                        "tool-call arguments JSON must decode to an object"
                    )
                function["arguments"] = arguments
    return normalized


def count_chat_input_tokens(
    tokenizer: Any, messages: list[dict], tools: list[dict]
) -> int:
    """Count the exact rendered chat input, including generation prompt and tools."""
    if not getattr(tokenizer, "chat_template", None):
        raise TokenBudgetConfigurationError("configured tokenizer has no chat_template")
    rendered_messages = _normalize_messages_for_chat_template(messages)
    normalized_tools = _normalize_tools_for_chat_template(tools)
    # HF templates are not uniform at this boundary. Qwen consumes bare JSON
    # function schemas, while Gemma 4 consumes OpenAI's {type, function}
    # wrapper. Prefer the historical bare form and retry the exact API wrapper
    # only when the tokenizer itself rejects it. The request sent to LiteLLM is
    # never changed.
    tool_variants = [normalized_tools]
    if tools and normalized_tools != tools:
        tool_variants.append(deepcopy(tools))
    failures: list[str] = []
    encoded = None
    for rendered_tools in tool_variants:
        try:
            encoded = tokenizer.apply_chat_template(
                rendered_messages,
                tools=rendered_tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            break
        except Exception as exc:
            failures.append(str(exc))
    if encoded is None:
        detail = "; wrapper fallback: ".join(failures)
        raise TokenBudgetConfigurationError(
            f"configured tokenizer could not render messages plus tool schema: {detail}"
        )
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded is None:
        raise TokenBudgetConfigurationError("tokenizer returned no input_ids")
    if hasattr(encoded, "shape"):
        shape = tuple(int(x) for x in encoded.shape)
        if not shape:
            raise TokenBudgetConfigurationError("tokenizer returned scalar input_ids")
        return shape[-1]
    if not isinstance(encoded, (list, tuple)):
        raise TokenBudgetConfigurationError(
            f"unsupported input_ids type {type(encoded).__name__}"
        )
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise TokenBudgetConfigurationError(
                f"expected one rendered conversation, got batch {len(encoded)}"
            )
        encoded = encoded[0]
    return len(encoded)


def record_token_count(
    path: Path,
    *,
    tokenizer_name: str,
    actual_input_tokens: int,
    max_input_tokens: int,
    message_count: int,
    admitted: bool,
    observation_retained_payload_chars: int | None = None,
) -> None:
    """Append one content-free, process-safe token admission receipt."""
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer_name": tokenizer_name,
        "actual_input_tokens": actual_input_tokens,
        "max_input_tokens": max_input_tokens,
        "message_count": message_count,
        "tool_schema_included": True,
        "history_truncated": False,
        "observation_retained_payload_chars": observation_retained_payload_chars,
        "admitted": admitted,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def enforce_token_budget(
    *,
    actual_input_tokens: int,
    max_input_tokens: int,
) -> None:
    if (
        not isinstance(max_input_tokens, int)
        or isinstance(max_input_tokens, bool)
        or max_input_tokens <= 0
    ):
        raise TokenBudgetConfigurationError(
            f"max_input_tokens must be a positive integer, got {max_input_tokens!r}"
        )
    if actual_input_tokens > max_input_tokens:
        raise InputTokenBudgetExceeded(
            "mini-swe-agent input token budget exceeded: "
            f"actual={actual_input_tokens}, limit={max_input_tokens}; "
            "full history rejected without truncation"
        )
