# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the pure TTS metric helpers used by the eval and
benchmark runners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module.test_status import TtsTestStatus
from test_module.benchmark_tests.tts_benchmark_tests import (
    DEFAULT_TTS_TEXT,
    _tts_avg,
    _tts_num_calls,
    _tts_test_text,
    _tts_throughput_rps,
    _tts_ttft_percentiles,
)


def _status(ttft_ms=None, rtr=None) -> TtsTestStatus:
    return TtsTestStatus(status=True, elapsed=1.0, ttft_ms=ttft_ms, rtr=rtr)


class TestTtsGenericAverage:
    """``_tts_avg`` is the benchmark module's generalized averaging helper."""

    def test_average_ttft_field(self):
        statuses = [
            _status(ttft_ms=100.0),
            _status(ttft_ms=200.0),
            _status(ttft_ms=None),
        ]
        assert _tts_avg(statuses, "ttft_ms") == pytest.approx(150.0)

    def test_average_rtr_field(self):
        statuses = [_status(rtr=1.0), _status(rtr=2.0)]
        assert _tts_avg(statuses, "rtr") == pytest.approx(1.5)

    def test_all_none_returns_none(self):
        assert _tts_avg([_status(ttft_ms=None)], "ttft_ms") is None

    def test_empty_returns_none(self):
        assert _tts_avg([], "rtr") is None


class TestTtsTtftPercentiles:
    def test_p50_p90_p95_indices_on_ten_samples(self):
        # Unsorted input verifies the helper sorts before indexing. n=10:
        # p50 -> ceil(5.0)-1 = 4 -> 50; p90 -> ceil(9.0)-1 = 8 -> 90;
        # p95 -> ceil(9.5)-1 = 9 -> 100.
        values = [50.0, 10.0, 100.0, 30.0, 20.0, 90.0, 40.0, 70.0, 60.0, 80.0]
        statuses = [_status(ttft_ms=v) for v in values]
        assert _tts_ttft_percentiles(statuses) == (50.0, 90.0, 100.0)

    def test_none_samples_are_ignored(self):
        # valid sorted [10, 20, 30], n=3: p50 -> idx 1 -> 20; p90/p95 -> idx 2 -> 30.
        statuses = [
            _status(ttft_ms=None),
            _status(ttft_ms=30.0),
            _status(ttft_ms=10.0),
            _status(ttft_ms=20.0),
        ]
        assert _tts_ttft_percentiles(statuses) == (20.0, 30.0, 30.0)

    def test_single_sample(self):
        assert _tts_ttft_percentiles([_status(ttft_ms=42.0)]) == (42.0, 42.0, 42.0)

    def test_empty_returns_zeroes(self):
        assert _tts_ttft_percentiles([]) == (0.0, 0.0, 0.0)

    def test_all_none_returns_zeroes(self):
        assert _tts_ttft_percentiles([_status(ttft_ms=None)]) == (0.0, 0.0, 0.0)


class TestTtsThroughputRps:
    def test_serial_requests_over_wall_clock(self):
        statuses = [_status(ttft_ms=10.0) for _ in range(4)]
        assert _tts_throughput_rps(statuses, wall_seconds=2.0) == pytest.approx(2.0)

    def test_only_successful_requests_counted(self):
        statuses = [
            TtsTestStatus(status=True, elapsed=1.0),
            TtsTestStatus(status=False, elapsed=1.0),
            TtsTestStatus(status=True, elapsed=1.0),
        ]
        assert _tts_throughput_rps(statuses, wall_seconds=1.0) == pytest.approx(2.0)

    def test_non_positive_wall_seconds_returns_none(self):
        assert _tts_throughput_rps([_status()], wall_seconds=0.0) is None

    def test_no_successful_requests_returns_none(self):
        statuses = [TtsTestStatus(status=False, elapsed=1.0)]
        assert _tts_throughput_rps(statuses, wall_seconds=5.0) is None


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
        ctx = SimpleNamespace(
            all_params=SimpleNamespace(tasks=[SimpleNamespace(text="custom text")])
        )
        assert _tts_test_text(ctx) == "custom text"

    def test_falls_back_to_task_name(self):
        ctx = SimpleNamespace(
            all_params=SimpleNamespace(tasks=[SimpleNamespace(task_name="my_task")])
        )
        assert _tts_test_text(ctx) == "my_task"

    def test_default_when_params_is_list(self):
        ctx = SimpleNamespace(all_params=[])
        assert _tts_test_text(ctx) == DEFAULT_TTS_TEXT

    def test_default_when_no_tasks(self):
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[]))
        assert _tts_test_text(ctx) == DEFAULT_TTS_TEXT
