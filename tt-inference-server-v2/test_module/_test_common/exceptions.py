# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Control-flow exceptions a test can raise to declare a non-error outcome.

These let a test opt out of the default "any exception is an ERROR" handling
in :meth:`BaseTest.run_tests`. Both require a human-readable ``reason`` so the
report can explain *why* the test did not produce a graded result. An optional
``data`` mapping carries structured diagnostics (e.g. chip counts) that get
merged into the emitted Block without overriding its envelope keys.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


class TestOutcomeSignal(Exception):
    """Base class for exceptions that map to a non-error :class:`TestStatus`."""

    def __init__(self, reason: str, data: Optional[Mapping[str, Any]] = None):
        if not reason or not reason.strip():
            raise ValueError(f"{type(self).__name__} requires a non-empty reason")
        super().__init__(reason)
        self.reason = reason
        self.data: Dict[str, Any] = dict(data or {})


class SkipTest(TestOutcomeSignal):
    """Raise to mark a run as ``SKIP`` (intentionally not run, non-blocking)."""


class NotApplicable(TestOutcomeSignal):
    """Raise to mark a run as ``NA`` (ran but not gradable, non-blocking)."""


__all__ = ["NotApplicable", "SkipTest", "TestOutcomeSignal"]
