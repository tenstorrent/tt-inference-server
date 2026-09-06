# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tripwires on the MiniMax-H3 deployment constants in tt_model_runners/dit_runners.py.

The module itself is not importable in the unit-test environment (tests/conftest.py stubs it and the
tt-metal tree is not on sys.path), so the values are read from the source text.  Each test names
the incident behind it.
"""
import os
import re

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt_model_runners", "dit_runners.py")


def _const(name: str) -> int:
    text = open(SRC).read()
    m = re.search(rf"^{name}\s*=\s*([0-9_]+)", text, re.M)
    assert m, f"{name} not found in dit_runners.py"
    return int(m.group(1).replace("_", ""))


# Was xfail(strict) against the 150 MB region; fixed upstream in cce5708f ("update for bucketed tracing"):
# 150_000_000 -> 1_005_000_000. Kept as a plain tripwire: the bucketed design keeps up to six denoise
# captures resident and an overflow is a TT_FATAL in end_trace_capture that fails the job and leaks that
# capture's budget.
def test_trace_region_matches_what_upstream_validates():
    assert _const("MINIMAX_H3_TRACE_REGION_BYTES") >= 1_005_000_000   # cce5708f: six t2va/fl2va + six ref2va rungs captured at construction


def test_trace_region_is_at_least_the_measured_minimum():
    """Five captures fit in 150 MB on quad1 (2026-09-05); anything smaller is unexplored."""
    assert _const("MINIMAX_H3_TRACE_REGION_BYTES") >= 150_000_000


def test_h3_runner_reads_duration_and_aspect_from_the_request():
    """The runner must keep resolving the served shape from duration_seconds / aspect_ratio (the SP path
    dropped them once; the pure mapping lives in _resolve_shape)."""
    text = open(SRC).read()
    assert "def _resolve_shape" in text
    assert 'getattr(request, "duration_seconds"' in text and 'getattr(request, "aspect_ratio"' in text
