# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the workflow-facing TTS load benchmark adapter."""

from __future__ import annotations

import json
import math
import threading
import time
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
        "ttf4_p50": 540.0 + offset,
        "ttf4_p95": 560.0 + offset,
        "ttfb_max": 300.0 + offset,
        "ttfs_max": 230.0 + offset,
        "ttft_max": 530.0 + offset,
        "ttf4_max": 570.0 + offset,
    }


def test_report_rows_use_fc_sc_tc_names_and_normalize_nan():
    row = _harness_row()
    row["ttfs_p95"] = math.nan

    [result] = benchmark._report_rows([row])

    assert result["fc_p50_ms"] == 100.0
    assert result["sc_p95_ms"] is None
    assert result["tc_p95_ms"] == 520.0
    assert result["c4_p50_ms"] == 540.0
    assert result["c4_p95_ms"] == 560.0
    assert (result["fc_max_ms"], result["sc_max_ms"]) == (300.0, 230.0)
    assert (result["tc_max_ms"], result["c4_max_ms"]) == (530.0, 570.0)
    assert "ttfb_p50" not in result


def test_aggregate_reports_max_and_counts_empty_audio_as_errors():
    harness = benchmark.tts_load_harness

    def rec(t_send, ok=True, error=None, ttfb=0.1, gaps=(0.6, 0.9, 1.2)):
        return {
            "conc": 4,
            "t_send": t_send,
            "t_end": t_send + 5.0,
            "ok": ok,
            "error": error,
            "ttfb_s": ttfb if ok else None,
            "sc_s": 0.2 if ok else None,
            "tc_gaps": list(gaps) if ok else [],
            "nchunks": 5 if ok else 0,
            "chars": 10 if t_send == 1.0 else 30,
        }

    doc = {
        "meta": {"chars_per_request": 100},
        "levels": [{"conc": 4, "window_start": 0.0, "window_end": 60.0}],
        "records": [
            rec(1.0),
            rec(2.0, ttfb=0.4, gaps=(0.6, 0.9, 1.5)),
            rec(3.0, ok=False, error="empty audio (header only)"),
            rec(4.0, ok=False, error="http 429: full"),
        ],
    }

    [row], _ = harness.aggregate(doc)

    assert (row["n_ok"], row["n_err"]) == (2, 2)
    assert row["errors"] == {"empty audio (header only)": 1, "http 429: full": 1}
    assert row["ttfb_max"] == pytest.approx(400.0)
    assert row["ttfs_max"] == pytest.approx(200.0)
    assert row["ttft_max"] == pytest.approx(600.0)
    assert row["ttf4_max"] == pytest.approx(1500.0)
    # Characters/hour credits each delivered response's own text (10 + 30 chars).
    assert row["mchar_h"] == pytest.approx(40 / 60.0 * 3600.0 / 1e6)


def test_prewarm_sends_drawn_prompts(monkeypatch):
    harness = benchmark.tts_load_harness
    sent = []
    base = harness.Target(host="quad-driver.example", text="base")

    def fake_request(target, uid, rec):
        sent.append(target.text)
        time.sleep(0.01)

    monkeypatch.setattr(harness, "request", fake_request)
    done = harness.prewarm(base, 2, 0.05, draw=lambda: base.with_prompt("drawn", [7]))

    assert done > 0 and sent and set(sent) == {"drawn"}


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
        "c4_p50_ms": 615.0,
        "c4_p95_ms": 635.0,
    }


def test_target_uses_context_host_port_and_replay_preflight_text():
    ctx = SimpleNamespace(
        base_url="http://quad-driver.example:8123",
        server_port=8123,
    )

    target = benchmark._target(ctx)

    assert target.host == "quad-driver.example"
    assert target.port == 8123
    assert target.text == benchmark.DEFAULT_TTS_TEXT
    assert target.speech_ids == []


def _workflow_ctx(workflow_args, **fields):
    return SimpleNamespace(
        runtime_config=SimpleNamespace(workflow_args=workflow_args), **fields
    )


@pytest.mark.parametrize(
    "workflow_args, expected",
    [
        (None, ((1, 2, 4, 8, 16), None)),
        ("", ((1, 2, 4, 8, 16), None)),
        ("concurrency=480 isl=256", ((480,), 256)),
        ("concurrency=64,160,320,160", ((64, 160, 320), None)),
        ("isl=64", ((1, 2, 4, 8, 16), 64)),
        ("concurrency=32 other-key=1", ((32,), None)),
    ],
    ids=("unset", "empty", "single-level", "ladder", "isl-only", "unknown-key"),
)
def test_workflow_args_override_ladder_and_isl(workflow_args, expected):
    assert benchmark._workflow_params(_workflow_ctx(workflow_args)) == expected


def test_eval_workflow_args_are_left_to_the_eval_without_warning(caplog):
    ctx = _workflow_ctx("concurrency=600,700 eval_requests=20 eval_concurrency=10")

    with caplog.at_level("WARNING"):
        assert benchmark._workflow_params(ctx) == ((600, 700), None)

    assert "ignores workflow args" not in caplog.text


@pytest.mark.parametrize(
    "workflow_args",
    ["concurrency=0", "concurrency=abc", "concurrency=", "isl=-1", "isl"],
)
def test_invalid_workflow_args_are_rejected(workflow_args):
    with pytest.raises(ValueError):
        benchmark._workflow_params(_workflow_ctx(workflow_args))


def test_target_text_follows_isl_workflow_arg():
    ctx = _workflow_ctx(
        "isl=256", base_url="http://quad-driver.example:8000", server_port=8000
    )

    target = benchmark._target(ctx)

    assert len(target.text) == round(256 * benchmark.TTS_CI_CHARS_PER_TOKEN)


def _ladder_sizes():
    with open(benchmark.TTS_CI_TRAFFIC_DATASET) as f:
        ladder = json.load(f)["ladder"]
    return {t["text"] for t in ladder["texts"]}, {
        v["voice_speech_ids"] for v in ladder["voices"]
    }


def test_traffic_draw_sends_prefixes_of_the_real_voice():
    texts, sizes = _ladder_sizes()
    real = list(range(100, 100 + max(sizes)))  # stands in for encoded real speech
    base = benchmark.tts_load_harness.Target(host="quad-driver.example", port=8123)

    draw = benchmark._traffic_draw(base, real)
    draws = [draw() for _ in range(200)]

    assert all(t.host == "quad-driver.example" and t.port == 8123 for t in draws)
    assert all(t.text in texts and len(t.speech_ids) in sizes for t in draws)
    assert len({t.text for t in draws}) > 1
    assert len({len(t.speech_ids) for t in draws}) > 1
    # A length-N voice prompt is the first N IDs of the same real speech.
    assert all(t.speech_ids == real[: len(t.speech_ids)] for t in draws)
    body = json.loads(draws[0].request.split(b"\r\n\r\n", 1)[1])
    assert body == {"text": draws[0].text, "speech_ids": draws[0].speech_ids}
    # Seeded: the same draws every run.
    again = benchmark._traffic_draw(base, real)
    assert [again().text for _ in range(5)] == [t.text for t in draws[:5]]


def test_level_timing_defaults_and_overrides():
    defaults = (benchmark.TTS_CI_WARMUP_SECONDS, benchmark.TTS_CI_DURATION_SECONDS)
    assert benchmark._level_timing(_workflow_ctx("concurrency=8")) == defaults
    assert benchmark._level_timing(_workflow_ctx("warmup=60 duration=180")) == (
        60.0,
        180.0,
    )
    with pytest.raises(ValueError):
        benchmark._level_timing(_workflow_ctx("duration=0"))


def test_traffic_draw_without_voice_ids_sends_seeded_random_ids(caplog):
    _, sizes = _ladder_sizes()
    base = benchmark.tts_load_harness.Target(host="quad-driver.example")

    with caplog.at_level("WARNING"):
        draw = benchmark._traffic_draw(base)
    draws = [draw() for _ in range(50)]

    assert all(len(t.speech_ids) in sizes for t in draws)
    ids = [i for t in draws for i in t.speech_ids]
    assert 0 <= min(ids) and max(ids) < benchmark.TTS_CODEBOOK_SIZE
    # Seeded: the same voice prompts every run.
    again = benchmark._traffic_draw(base)
    assert [again().speech_ids for _ in range(5)] == [t.speech_ids for t in draws[:5]]
    assert "random speech IDs" in caplog.text


def test_traffic_draw_rejects_voice_ids_shorter_than_the_captured_prompts():
    _, sizes = _ladder_sizes()
    base = benchmark.tts_load_harness.Target(host="quad-driver.example")

    with pytest.raises(ValueError, match=f"needs up to {max(sizes)}"):
        benchmark._traffic_draw(base, list(range(max(sizes) - 1)))


@pytest.mark.parametrize(
    "content", [[5, 6, 7], {"source": "speech.wav", "speech_ids": [5, 6, 7]}]
)
def test_voice_ids_file_is_read_from_workflow_args(tmp_path, content):
    path = tmp_path / "voice.json"
    path.write_text(json.dumps(content))

    ctx = _workflow_ctx(f"concurrency=8 voice_ids={path}")

    assert benchmark._voice_ids(ctx) == [5, 6, 7]
    assert benchmark._voice_ids(_workflow_ctx("concurrency=8")) is None


@pytest.mark.parametrize("content", [[], [-1], [65536], ["7"]])
def test_voice_ids_file_must_hold_codec_ids(tmp_path, content):
    path = tmp_path / "voice.json"
    path.write_text(json.dumps(content))

    with pytest.raises(ValueError, match="must hold speech IDs"):
        benchmark._voice_ids(_workflow_ctx(f"voice_ids={path}"))


def test_run_closed_sends_drawn_prompts_after_ramped_starts(monkeypatch):
    harness = benchmark.tts_load_harness
    sent, offsets = [], []
    base = harness.Target(host="quad-driver.example", text="base")
    prompts = iter(range(10_000))

    def fake_request(target, uid, rec):
        sent.append(target)
        rec.update(ok=True, chars=len(target.text), t_end=time.time())
        time.sleep(0.01)

    real_wait = threading.Event.wait

    def recording_wait(self, timeout=None):
        if timeout is None:  # Thread.start() waiting for the thread to run
            return real_wait(self)
        offsets.append(timeout)
        return real_wait(self, 0)

    monkeypatch.setattr(harness, "request", fake_request)
    monkeypatch.setattr(threading.Event, "wait", recording_wait)
    recs, _, _ = harness.run_closed(
        base,
        4,
        0.05,
        0.0,
        draw=lambda: base.with_prompt(f"p{next(prompts)}", [1, 2]),
        ramp=20.0,
    )

    assert recs and sent
    assert all(t.text.startswith("p") and t.speech_ids == [1, 2] for t in sent)
    assert len(offsets) == 4 and all(0 <= o < 20.0 for o in offsets)
    assert len(set(offsets)) == 4


def test_integrity_check_grades_the_requested_ladder(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "run_tiered_check",
        lambda ctx, specs: ({"target": {}}, ReportCheckTypes.PASS),
    )
    ctx = _workflow_ctx("concurrency=480")

    _, custom = benchmark._target_checks(
        ctx, benchmark._report_rows([_harness_row(480)])
    )
    _, default_ladder = benchmark._target_checks(
        ctx,
        benchmark._report_rows([_harness_row(c) for c in benchmark.TTS_CI_CONCURRENCY]),
    )

    assert custom == ReportCheckTypes.PASS
    assert default_ladder == ReportCheckTypes.FAIL


def test_tts_latency_targets_are_inclusive():
    rows = benchmark._report_rows(
        [
            {
                **_harness_row(concurrency),
                "ttfb_p50": 100.0,
                "ttfb_p95": 125.0,
                "ttfs_p50": 300.0,
                "ttfs_p95": 300.0,
                "ttft_p50": 600.0,
                "ttft_p95": 600.0,
                "ttf4_p50": 960.0,
                "ttf4_p95": 960.0,
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
        lambda actual, requirement=None: observed.setdefault("health", actual),
    )
    monkeypatch.setattr(benchmark, "_target", lambda actual: "target")
    monkeypatch.setattr(benchmark, "block_id", lambda actual: "tts-quad")

    def fake_sweep(args, target, draw=None):
        observed["args"] = args
        observed["target"] = target
        observed["draw"] = draw
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
    # No isl workflow arg: replay the captured traffic with ramped worker starts.
    assert callable(observed["draw"])
    assert observed["args"].ramp == benchmark.TTS_CI_WARMUP_SECONDS
    assert block.targets["text_tokens"] is None
    assert block.targets["workload"] == "tts_staged_2026-10-27.json, random voice IDs"
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
    monkeypatch.setattr(
        benchmark, "require_health", lambda actual, requirement=None: None
    )
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


def test_sweep_runs_a_nonascending_ladder_in_the_requested_order(monkeypatch, tmp_path):
    harness = benchmark.tts_load_harness
    ran = []

    def fake_run_closed(target, conc, duration, warmup, **kwargs):
        ran.append(conc)
        return [], 0.0, 1.0

    monkeypatch.setattr(harness, "_preflight", lambda target: {"gen_s": 1.0})
    monkeypatch.setattr(harness, "run_closed", fake_run_closed)
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: None)
    concurrency, _ = benchmark._workflow_params(
        _workflow_ctx("concurrency=320,64,160,64")
    )

    document = harness.sweep(
        benchmark._sweep_args(tmp_path, concurrency=concurrency),
        harness.Target(host="quad-driver.example", text="hello"),
    )

    assert ran == [320, 64, 160]
    assert [level["conc"] for level in document["levels"]] == [320, 64, 160]


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
