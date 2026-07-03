# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the pure TTS metric helpers used by the eval and
benchmark runners. """

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module.test_status import TtsTestStatus
from test_module.eval_tests.tts_eval_tests import (
    DEFAULT_TTS_TEXT,
    _tts_num_calls,
    _tts_rtr,
    _tts_tail_latency,
    _tts_test_text,
    _tts_ttft,
)
from test_module.benchmark_tests.tts_benchmark_tests import _tts_avg


def _status(ttft_ms=None, rtr=None) -> TtsTestStatus:
    return TtsTestStatus(status=True, elapsed=1.0, ttft_ms=ttft_ms, rtr=rtr)


class TestTtsTtftAverage:
    def test_averages_only_valid_samples(self):
        statuses = [_status(ttft_ms=100.0), _status(ttft_ms=200.0), _status(ttft_ms=None)]
        assert _tts_ttft(statuses) == pytest.approx(150.0)

    def test_all_none_returns_none(self):
        assert _tts_ttft([_status(ttft_ms=None), _status(ttft_ms=None)]) is None

    def test_empty_returns_none(self):
        assert _tts_ttft([]) is None


class TestTtsRtrAverage:
    def test_averages_only_valid_samples(self):
        statuses = [_status(rtr=2.0), _status(rtr=4.0), _status(rtr=None)]
        assert _tts_rtr(statuses) == pytest.approx(3.0)

    def test_all_none_returns_none(self):
        assert _tts_rtr([_status(rtr=None)]) is None

    def test_empty_returns_none(self):
        assert _tts_rtr([]) is None


class TestTtsGenericAverage:
    """``_tts_avg`` is the benchmark module's generalized averaging helper."""

    def test_average_ttft_field(self):
        statuses = [_status(ttft_ms=100.0), _status(ttft_ms=200.0), _status(ttft_ms=None)]
        assert _tts_avg(statuses, "ttft_ms") == pytest.approx(150.0)

    def test_average_rtr_field(self):
        statuses = [_status(rtr=1.0), _status(rtr=2.0)]
        assert _tts_avg(statuses, "rtr") == pytest.approx(1.5)

    def test_all_none_returns_none(self):
        assert _tts_avg([_status(ttft_ms=None)], "ttft_ms") is None

    def test_empty_returns_none(self):
        assert _tts_avg([], "rtr") is None


class TestTtsTailLatency:
    def test_p90_p95_indices_on_ten_samples(self):
        # Unsorted input verifies the helper sorts before indexing.
        # n=10: p90 index = ceil(9.0)-1 = 8 -> 90; p95 index = ceil(9.5)-1 = 9 -> 100.
        values = [50.0, 10.0, 100.0, 30.0, 20.0, 90.0, 40.0, 70.0, 60.0, 80.0]
        statuses = [_status(ttft_ms=v) for v in values]
        assert _tts_tail_latency(statuses) == (90.0, 100.0)

    def test_none_samples_are_ignored(self):
        statuses = [_status(ttft_ms=None), _status(ttft_ms=30.0), _status(ttft_ms=10.0), _status(ttft_ms=20.0)]
        assert _tts_tail_latency(statuses) == (30.0, 30.0)

    def test_single_sample(self):
        assert _tts_tail_latency([_status(ttft_ms=42.0)]) == (42.0, 42.0)

    def test_empty_returns_zeroes(self):
        assert _tts_tail_latency([]) == (0.0, 0.0)

    def test_all_none_returns_zeroes(self):
        assert _tts_tail_latency([_status(ttft_ms=None)]) == (0.0, 0.0)


class TestTtsNumCalls:
    def test_eval_default_when_unconfigured(self):
        # all_params with .tasks (not a list) -> get_num_calls returns the
        # sentinel 2, so the TTS-specific eval default (5) applies.
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[object()]))
        assert _tts_num_calls(ctx, is_eval=True) == 5

    def test_benchmark_default_when_unconfigured(self):
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[object()]))
        assert _tts_num_calls(ctx, is_eval=False) == 10

    def test_configured_value_overrides_default(self):
        # A list of params carrying num_eval_runs is the "configured" case;
        # the explicit value wins for both eval and benchmark.
        ctx = SimpleNamespace(all_params=[SimpleNamespace(num_eval_runs=7)])
        assert _tts_num_calls(ctx, is_eval=True) == 7
        assert _tts_num_calls(ctx, is_eval=False) == 7


class TestTtsTestText:
    def test_prefers_task_text(self):
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[SimpleNamespace(text="custom text")]))
        assert _tts_test_text(ctx) == "custom text"

    def test_falls_back_to_task_name(self):
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[SimpleNamespace(task_name="my_task")]))
        assert _tts_test_text(ctx) == "my_task"

    def test_default_when_params_is_list(self):
        ctx = SimpleNamespace(all_params=[])
        assert _tts_test_text(ctx) == DEFAULT_TTS_TEXT

    def test_default_when_no_tasks(self):
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[]))
        assert _tts_test_text(ctx) == DEFAULT_TTS_TEXT
