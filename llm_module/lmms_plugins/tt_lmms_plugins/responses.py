# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Response handling for the TT chat client (no lmms-eval import, unit-testable)."""

DEFAULT_THREADS = 32


def response_text(response) -> str:
    """The assistant text of a chat completion, "" when the server sent none.

    With a reasoning parser a thought that never closes leaves ``content`` as
    None; upstream passes that None through and the task's selection filter
    then raises IndexError after the whole run, losing every answer.
    """
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError):
        return ""
    return text if isinstance(text, str) else ""


def thread_cap(n_tasks: int, num_concurrent=None) -> int:
    """Workers for one batch: the client's num_concurrent (default 32), never
    more than the batch has requests."""
    cap = int(num_concurrent) if num_concurrent else DEFAULT_THREADS
    return max(1, min(n_tasks, cap))


def parse_enable_thinking(value):
    """None (server default), or the boolean an ``enable_thinking`` model arg
    spells (``true``/``false``, ``1``/``0``, ``on``/``off``)."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "none", "default"):
        return None
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"enable_thinking={value!r} is not a boolean")
