# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Canonical test-outcome vocabulary shared by the test and report modules."""

from __future__ import annotations

from enum import Enum
from typing import Optional


class TestStatus(str, Enum):
    """Outcome of a single test run.

    Distinguishes the ways a test can "not pass" so the report and the process
    exit code tell the same story:

    - ``PASS``  ran and met its criteria.
    - ``FAIL``  ran and did not meet its criteria           -> blocking.
    - ``ERROR`` raised unexpectedly (crash / setup failure) -> blocking.
    - ``SKIP``  intentionally not run (needs a ``reason``)  -> non-blocking.
    - ``NA``    ran but produced no gradable result         -> non-blocking.
    """

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
    NA = "na"

    @property
    def is_blocking(self) -> bool:
        """Whether this status should fail the workflow / count as a blocker."""
        return self in _BLOCKING_STATUSES

    @classmethod
    def from_value(cls, value: object) -> Optional["TestStatus"]:
        """Coerce a raw ``block.data['status']`` value to a member, or None.

        Accepts an existing member or its string value (case-insensitively);
        anything unrecognised returns ``None`` so callers can fall back to the
        legacy boolean ``success`` field.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                return None
        return None

    @classmethod
    def from_legacy(cls, success: object, skipped: bool = False) -> "TestStatus":
        """Derive a status from the pre-enum ``success``/``skipped`` fields.

        Used for blocks produced before they carried an explicit ``status``.
        """
        if skipped:
            return cls.SKIP
        return cls.PASS if success is True else cls.FAIL


_BLOCKING_STATUSES = frozenset({TestStatus.FAIL, TestStatus.ERROR})


STATUS_GLYPHS: dict[TestStatus, str] = {
    TestStatus.PASS: "✅",
    TestStatus.FAIL: "❌",
    TestStatus.ERROR: "❌",
    TestStatus.SKIP: "⏭️",
    TestStatus.NA: "🟨",
}


def glyph_for(status: TestStatus) -> str:
    return STATUS_GLYPHS.get(status, "❌")


def glyph_for_label(label: str) -> str:
    status = TestStatus.from_value(label)
    return STATUS_GLYPHS.get(status, "") if status else ""


__all__ = ["TestStatus", "STATUS_GLYPHS", "glyph_for", "glyph_for_label"]
