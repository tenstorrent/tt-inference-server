# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Dependency-light token accounting for mini-swe-agent requests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InputTokenBudgetExceeded(RuntimeError):
    """The complete, untruncated agent history exceeds its declared budget."""


class TokenBudgetConfigurationError(RuntimeError):
    """The configured tokenizer cannot authoritatively count the API input."""


def count_chat_input_tokens(
    tokenizer: Any, messages: list[dict], tools: list[dict]
) -> int:
    """Count the exact rendered chat input, including generation prompt and tools."""
    if not getattr(tokenizer, "chat_template", None):
        raise TokenBudgetConfigurationError("configured tokenizer has no chat_template")
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise TokenBudgetConfigurationError(
            f"configured tokenizer could not render messages plus tool schema: {exc}"
        ) from exc
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
