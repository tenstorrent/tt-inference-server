#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Filter eval logs/artifacts without changing lm_eval request semantics."""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path


TT_KEY_RE = re.compile(rb"sk-tt-[A-Za-z0-9._=-]+")
TT_KEY_TEXT_RE = re.compile(r"sk-tt-[A-Za-z0-9._=-]+")
TT_KEY_PREFIX = b"sk-tt-"
TOKEN_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._=-")
REDACTION = b"sk-tt-REDACTED"
TEXT_REDACTION = "sk-tt-REDACTED"
SUPPRESSED_STREAM_PATTERNS = (
    b"Cannot determine EOS string to pass to stop sequence.",
    b"generation_kwargs:",
    b"generation_kwargs: {'stream': True} specified through cli",
    b"[Task: r1_aime24] metric acc is defined, but aggregation is not.",
    b"[Task: r1_aime24] metric acc is defined, but higher_is_better is not.",
    b"Chat template formatting change affects loglikelihood and multiple-choice tasks.",
    b"local-chat-completions (",
    b"local-completions (",
)


def redact_text(text: str) -> str:
    return TT_KEY_TEXT_RE.sub(TEXT_REDACTION, text)


def redact_bytes(data: bytes) -> bytes:
    return TT_KEY_RE.sub(REDACTION, data)


def split_safe_prefix(data: bytes) -> tuple[bytes, bytes]:
    """Return bytes safe to redact now plus a possible partial key suffix."""

    hold_start = len(data)

    for prefix_len in range(1, len(TT_KEY_PREFIX)):
        if data.endswith(TT_KEY_PREFIX[:prefix_len]):
            hold_start = min(hold_start, len(data) - prefix_len)

    idx = data.rfind(TT_KEY_PREFIX)
    if idx != -1 and all(byte in TOKEN_BYTES for byte in data[idx + len(TT_KEY_PREFIX) :]):
        hold_start = min(hold_start, idx)

    return data[:hold_start], data[hold_start:]


def redact_stream() -> int:
    key_carry = b""
    held_line = b""

    def emit(data: bytes, *, final: bool = False) -> None:
        nonlocal key_carry
        if final:
            data = key_carry + data
            key_carry = b""
            if data:
                os.write(sys.stdout.fileno(), redact_bytes(data))
            return

        emit_now, key_carry = split_safe_prefix(key_carry + data)
        if emit_now:
            os.write(sys.stdout.fileno(), redact_bytes(emit_now))

    while True:
        chunk = os.read(sys.stdin.fileno(), 4096)
        if not chunk:
            break
        data = held_line + chunk
        held_line = b""

        while True:
            newline_index = data.find(b"\n")
            if newline_index == -1:
                break

            line = data[: newline_index + 1]
            data = data[newline_index + 1 :]
            if any(pattern in line for pattern in SUPPRESSED_STREAM_PATTERNS):
                continue
            emit(line)

        # Most progress output is not newline-terminated. Emit it immediately
        # unless it already contains a warning we want to drop at line granularity.
        if any(pattern in data for pattern in SUPPRESSED_STREAM_PATTERNS):
            held_line = data
        else:
            emit(data)

    if held_line and not any(pattern in held_line for pattern in SUPPRESSED_STREAM_PATTERNS):
        emit(held_line)
    emit(b"", final=True)
    return 0


def redact_files(paths: list[str]) -> int:
    for raw_path in paths:
        path = Path(raw_path)
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        redacted = redact_text(original)
        if redacted != original:
            path.write_text(redacted, encoding="utf-8")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        return redact_files(sys.argv[1:])
    return redact_stream()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
