# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the workflow-facing TTS load benchmark adapter."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from test_module._test_common import ReportCheckTypes
from test_module.benchmark_tests import tts_benchmark_tests as benchmark
from test_module.benchmark_tests.tts_benchmark_tests import (
    DEFAULT_TTS_TEXT,
    _tts_avg,
    _tts_num_calls,
    _tts_test_text,
    _tts_throughput_rps,
    _tts_ttft_percentiles,
)
from test_module.test_status import TtsTestStatus


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
        values = [50.0, 10.0, 100.0, 30.0, 20.0, 90.0, 40.0, 70.0, 60.0, 80.0]
        statuses = [_status(ttft_ms=value) for value in values]
        assert _tts_ttft_percentiles(statuses) == (50.0, 90.0, 100.0)

    def test_none_samples_are_ignored(self):
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
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[object()]))
        assert _tts_num_calls(ctx, is_eval=True) == 5

    def test_benchmark_default_when_unconfigured(self):
        ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=[object()]))
        assert _tts_num_calls(ctx, is_eval=False) == 10

    def test_configured_value_overrides_default(self):
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


def _harness_row(concurrency=1, *, offset=0.0, errors=0):
    return {
        "conc": concurrency,
        "C": concurrency - 0.1,
        "n_ok": 20,
        "n_err": errors,
        "rps": 2.5,
        "mchar_h": 31.2,
        "chunks": 12.0,
        "ttfb_p50": 100.0 + offset,
        "ttfb_p95": 200.0 + offset,
        "ttfs_p50": 210.0 + offset,
        "ttfs_p95": 220.0 + offset,
        "ttft_p50": 500.0 + offset,
        "ttft_p95": 520.0 + offset,
    }


def test_report_rows_use_fc_sc_tc_names_and_normalize_nan():
    row = _harness_row()
    row["ttfs_p95"] = math.nan

    [result] = benchmark._report_rows([row])

    assert result["fc_p50_ms"] == 100.0
    assert result["sc_p95_ms"] is None
    assert result["tc_p95_ms"] == 520.0
    assert "ttfb_p50" not in result


def test_worst_latency_values_require_every_concurrency_to_pass():
    rows = benchmark._report_rows([_harness_row(1), _harness_row(16, offset=75.0)])

    worst = benchmark._worst_latency_values(rows)

    assert worst == {
        "fc_p50_ms": 175.0,
        "fc_p95_ms": 275.0,
        "sc_p50_ms": 285.0,
        "sc_p95_ms": 295.0,
        "tc_p50_ms": 575.0,
        "tc_p95_ms": 595.0,
    }


def test_target_uses_context_host_port_and_ci_text_shape():
    ctx = SimpleNamespace(
        base_url="http://quad-driver.example:8123",
        server_port=8123,
    )

    target = benchmark._target(ctx)

    assert target.host == "quad-driver.example"
    assert target.port == 8123
    assert len(target.text) == round(
        benchmark.TTS_CI_TEXT_TOKENS * benchmark.TTS_CI_CHARS_PER_TOKEN
    )


def test_tts_latency_targets_are_inclusive():
    rows = benchmark._report_rows(
        [
            {
                **_harness_row(concurrency),
                "ttfb_p50": 150.0,
                "ttfb_p95": 350.0,
                "ttfs_p50": 270.0,
                "ttfs_p95": 270.0,
                "ttft_p50": 570.0,
                "ttft_p95": 570.0,
            }
            for concurrency in benchmark.TTS_CI_CONCURRENCY
        ]
    )
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(
            hf_model_repo="propritery/tts-2",
            cli_args={"device": "quad_galaxy"},
        )
    )

    checks, verdict = benchmark._target_checks(ctx, rows)

    assert verdict == ReportCheckTypes.PASS
    assert all(
        value == ReportCheckTypes.PASS
        for name, value in checks["target"].items()
        if name.endswith("_check")
    )


@pytest.mark.parametrize(
    "rows",
    [
        [_harness_row(concurrency) for concurrency in (1, 2, 4, 8)],
        [
            _harness_row(concurrency, errors=1 if concurrency == 16 else 0)
            for concurrency in benchmark.TTS_CI_CONCURRENCY
        ],
        [
            {
                **_harness_row(concurrency),
                "ttfs_p95": math.nan if concurrency == 16 else 220.0,
            }
            for concurrency in benchmark.TTS_CI_CONCURRENCY
        ],
    ],
    ids=("missing-level", "request-error", "missing-latency"),
)
def test_incomplete_load_results_fail_the_integrity_check(monkeypatch, rows):
    monkeypatch.setattr(
        benchmark,
        "run_tiered_check",
        lambda ctx, specs: (
            {
                tier: {"fc_p50_ms_check": ReportCheckTypes.PASS}
                for tier in ("functional", "complete", "target")
            },
            ReportCheckTypes.PASS,
        ),
    )

    checks, verdict = benchmark._target_checks(
        SimpleNamespace(), benchmark._report_rows(rows)
    )

    assert verdict == ReportCheckTypes.FAIL
    assert checks["target"]["benchmark_integrity_check"] == ReportCheckTypes.FAIL


def test_run_tts_benchmark_adapts_harness_document(monkeypatch, tmp_path):
    ctx = SimpleNamespace(
        output_path=str(tmp_path),
        model_spec=SimpleNamespace(impl=SimpleNamespace(impl_name="tt-tts")),
    )
    document = {"raw": "document"}
    rows = [_harness_row(1), _harness_row(16, errors=2)]
    observed = {}

    monkeypatch.setattr(
        benchmark,
        "require_health",
        lambda actual: observed.setdefault("health", actual),
    )
    monkeypatch.setattr(benchmark, "_target", lambda actual: "target")
    monkeypatch.setattr(benchmark, "block_id", lambda actual: "tts-quad")

    def fake_sweep(args, target):
        observed["args"] = args
        observed["target"] = target
        return document

    monkeypatch.setattr(benchmark.tts_load_harness, "sweep", fake_sweep)
    monkeypatch.setattr(
        benchmark.tts_load_harness,
        "aggregate",
        lambda actual: (rows, 3543),
    )
    monkeypatch.setattr(
        benchmark,
        "_target_checks",
        lambda actual_ctx, report_rows: (
            {"target": {"fc_p50_ms_check": ReportCheckTypes.PASS}},
            ReportCheckTypes.PASS,
        ),
    )

    block = benchmark.run_tts_benchmark(ctx)

    assert observed["health"] is ctx
    assert observed["target"] == "target"
    assert observed["args"].concurrency == "1,2,4,8,16"
    assert observed["args"].out == str(tmp_path / "tts_load_results.json")
    assert block.kind == "benchmarks"
    assert block.id == "tts-quad"
    assert block.data["Benchmarks"]["num_requests"] == 42
    assert block.data["Benchmarks"]["num_successful"] == 40
    assert block.data["Benchmarks"]["error_request_count"] == 2
    assert len(block.data["Latency by Concurrency"]) == 2


def test_tts_benchmark_dispatch_preserves_legacy_implementations(monkeypatch):
    load_result = object()
    legacy_result = object()
    monkeypatch.setattr(benchmark, "run_tts_load_benchmark", lambda ctx: load_result)
    monkeypatch.setattr(
        benchmark, "_run_legacy_tts_benchmark", lambda ctx: legacy_result
    )

    tt_tts_ctx = SimpleNamespace(
        model_spec=SimpleNamespace(impl=SimpleNamespace(impl_name="tt-tts"))
    )
    speech_t5_ctx = SimpleNamespace(
        model_spec=SimpleNamespace(impl=SimpleNamespace(impl_name="speecht5-tts"))
    )

    assert benchmark.run_tts_benchmark(tt_tts_ctx) is load_result
    assert benchmark.run_tts_benchmark(speech_t5_ctx) is legacy_result


def test_legacy_tts_benchmark_keeps_previous_report_shape(monkeypatch):
    statuses = [
        TtsTestStatus(status=True, elapsed=1.0, ttft_ms=100.0, rtr=2.0),
        TtsTestStatus(status=True, elapsed=1.0, ttft_ms=200.0, rtr=4.0),
    ]
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="speecht5_tts"),
        device=SimpleNamespace(name="N150"),
    )
    monkeypatch.setattr(benchmark, "require_health", lambda actual: None)
    monkeypatch.setattr(benchmark, "_tts_num_calls", lambda actual, is_eval=False: 2)
    monkeypatch.setattr(
        benchmark,
        "_run_tts_benchmark",
        lambda actual, calls: statuses,
    )
    monkeypatch.setattr(
        benchmark,
        "_tts_target_checks",
        lambda actual, ttft, rtr: ({"target": {}}, ReportCheckTypes.PASS),
    )
    monkeypatch.setattr(benchmark, "block_id", lambda actual: "legacy-tts")
    times = iter((10.0, 12.0))
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(times))

    block = benchmark._run_legacy_tts_benchmark(ctx)

    assert set(block.data) == {"Benchmarks"}
    assert block.title == "Text-to-Speech Benchmark"
    assert block.data["Benchmarks"] == {
        "num_requests": 2,
        "ttft": 0.15,
        "rtr": 3.0,
        "ttft_p50": 0.1,
        "ttft_p90": 0.2,
        "ttft_p95": 0.2,
        "throughput_rps": 1.0,
        "target_check": ReportCheckTypes.PASS,
        "target_checks": {"target": {}},
    }


def test_preflight_raises_instead_of_exiting_workflow(monkeypatch):
    monkeypatch.setattr(
        benchmark.tts_load_harness,
        "request",
        lambda target: {"ok": False, "error": "connection refused"},
    )

    with pytest.raises(RuntimeError, match="preflight FAILED 3x"):
        benchmark.tts_load_harness._preflight(object())


def test_standalone_harness_keeps_clean_preflight_exit(monkeypatch):
    monkeypatch.setattr(
        benchmark.tts_load_harness,
        "sweep",
        lambda args, target: (_ for _ in ()).throw(
            benchmark.tts_load_harness.PreflightError("preflight failed")
        ),
    )

    with pytest.raises(SystemExit, match="preflight failed"):
        benchmark.tts_load_harness.main(["sweep", "--no-table"])
