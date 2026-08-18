# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the :class:`report_module.status.TestStatus` outcome enum."""

from __future__ import annotations

import pytest

from report_module.status import TestStatus


@pytest.mark.parametrize(
    "status, blocking",
    [
        (TestStatus.PASS, False),
        (TestStatus.FAIL, True),
        (TestStatus.ERROR, True),
        (TestStatus.SKIP, False),
        (TestStatus.NA, False),
    ],
)
def test_is_blocking(status, blocking):
    assert status.is_blocking is blocking


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("pass", TestStatus.PASS),
        ("FAIL", TestStatus.FAIL),
        ("  Skip  ", TestStatus.SKIP),
        (TestStatus.NA, TestStatus.NA),
    ],
)
def test_from_value_accepts_known_spellings(raw, expected):
    assert TestStatus.from_value(raw) is expected


@pytest.mark.parametrize("raw", ["", "bogus", None, 3, True])
def test_from_value_returns_none_for_unknown(raw):
    assert TestStatus.from_value(raw) is None


def test_from_legacy_maps_success_boolean():
    assert TestStatus.from_legacy(True) is TestStatus.PASS
    assert TestStatus.from_legacy(False) is TestStatus.FAIL
    assert TestStatus.from_legacy(None) is TestStatus.FAIL


def test_from_legacy_skipped_wins_over_success():
    assert TestStatus.from_legacy(False, skipped=True) is TestStatus.SKIP
    assert TestStatus.from_legacy(True, skipped=True) is TestStatus.SKIP
