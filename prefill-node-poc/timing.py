# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Latency measurement for the prefill pipeline. Use @timed() or @timed("custom_label")
on the main steps; step names are derived from module (file) and method name
and highlighted in the log line.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])

TIMING_LOGGER = logging.getLogger("prefill_node_poc.timing")


def _step_label(fn: Callable[..., object]) -> str:
    """Derive step label from module (file) and method name, e.g. scheduler.schedule."""
    module = getattr(fn, "__module__", "") or ""
    qualname = getattr(fn, "__qualname__", fn.__name__)
    method = qualname.split(".")[-1] if "." in qualname else qualname
    return f"{module}.{method}" if module else method


def timed(
    step_name: str | None = None,
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    """
    Decorator that measures elapsed time with time.perf_counter() and logs
    latency_ms with step name. Step name defaults to module.method (file + method).
    Use on sync methods/functions only.
    """

    def decorator(fn: F) -> F:
        label = step_name if step_name is not None else _step_label(fn)

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            log = logger or TIMING_LOGGER
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                log.info("  [ %s ]  latency_ms=%.3f", label, elapsed_ms)

        return wrapper  # type: ignore[return-value]

    return decorator
