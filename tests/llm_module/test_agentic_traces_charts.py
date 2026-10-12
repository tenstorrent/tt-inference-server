# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the per-point telemetry charts of an agentic-trace run."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from llm_module.agentic_traces.charts import (
    PROFILE_EXPORT,
    TELEMETRY_CHART_PREFIX,
    TIMESLICES,
    load_requests,
    write_point_charts,
)

T0 = 1_000_000_000_000


def _record(start_s, end_s, *, phase="profiling", conv="c1", depth=0, ok=True, **m):
    meta = {
        "conversation_id": conv,
        "root_correlation_id": conv.split("::")[0],
        "x_correlation_id": conv,
        "turn_index": 1,
        "agent_depth": depth,
        "benchmark_phase": phase,
        "was_cancelled": False,
        "request_start_ns": T0 + int(start_s * 1e9),
        "request_end_ns": T0 + int(end_s * 1e9),
    }
    metrics = (
        {
            "time_to_first_token": {"value": 1500.0, "unit": "ms"},
            "request_latency": {"value": (end_s - start_s) * 1000, "unit": "ms"},
            "input_sequence_length": {"value": 40_000 + start_s, "unit": "tokens"},
            "output_sequence_length": {"value": 300, "unit": "tokens"},
            "output_token_throughput_per_user": {"value": 55.0, "unit": "t/s"},
        }
        if ok
        else {"error_isl": {"value": 9, "unit": "tokens"}}
    )
    record = {"metadata": meta, "metrics": metrics}
    if not ok:
        record["error"] = {"type": "InvalidInferenceResultError"}
    return json.dumps(record)


def _write_artifacts(tmp_path, *, slices=True):
    lines = [
        _record(-30, -20, phase="warmup"),
        _record(0, 10),
        _record(5, 25, conv="c2"),
        _record(12, 30, conv="c1::fa:001", depth=1),
        _record(40, 41, ok=False),
    ]
    (tmp_path / PROFILE_EXPORT).write_text("\n".join(lines) + "\n")
    if slices:
        rows = ["Timeslice,Start_NS,End_NS,Metric,Unit,Stat,Value"]
        for i in range(5):
            start = T0 + i * 10**9
            for metric in (
                "Effective Prefill Throughput",
                "Effective Decode Throughput",
                "Effective Concurrency",
                "Tokens In Flight",
            ):
                rows.append(f"{i},{start},{start + 10**9},{metric},x,avg,{i + 1}")
        (tmp_path / TIMESLICES).write_text("\n".join(rows) + "\n")


def test_load_requests_zeroes_time_at_profiling_start(tmp_path):
    _write_artifacts(tmp_path)
    requests, origin = load_requests(tmp_path / PROFILE_EXPORT)
    assert origin == T0
    assert requests[0].phase == "warmup" and requests[0].start == -30.0
    assert [r.ok for r in requests] == [True, True, True, True, False]
    assert requests[1].ttft == 1.5


def test_write_point_charts_emits_valid_svg_per_chart(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    out = tmp_path / "charts"

    written = write_point_charts(artifacts, out)

    assert set(written) == {
        "isl_osl",
        "latency",
        "interactivity",
        "throughput",
        "load",
    }
    for filename in written.values():
        ET.fromstring((out / filename).read_text())  # well-formed XML


def test_charts_without_timeslices_skip_only_the_slice_charts(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts, slices=False)

    written = write_point_charts(artifacts, tmp_path / "charts")

    assert "throughput" not in written
    assert {"isl_osl", "latency", "interactivity", "load"} <= set(written)


def test_no_export_means_no_charts(tmp_path):
    assert write_point_charts(tmp_path, tmp_path / "charts") == {}


def test_prefix_matches_report_renderer():
    from report_module.agentic_traces_renderer import (
        TELEMETRY_CHART_PREFIX as RENDERER_PREFIX,
    )

    assert RENDERER_PREFIX == TELEMETRY_CHART_PREFIX


def test_percentile_helpers():
    from llm_module.agentic_traces.charts import (
        _cumulative_percentile,
        _rolling_percentile,
    )

    points = [(float(i), float(i)) for i in range(1, 101)]
    rolling = _rolling_percentile(points, 0.5, window=10)
    assert rolling[-1][1] == 95.5  # median of 91..100
    cumulative = _cumulative_percentile(points, 1.0)
    assert [v for _, v in cumulative] == [float(i) for i in range(1, 101)]


def test_there_is_no_request_timeline_chart(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    assert "timeline" not in write_point_charts(artifacts, tmp_path / "charts")


def _server_metrics(path, hits, queries):
    """A server_metrics_export.json shaped like AIPerf writes it (indent=2)."""

    def block(name, per_window):
        return {
            "type": "counter",
            "series": [
                {
                    "endpoint_url": "http://e:1/metrics",
                    "timeslices": [
                        {"start_ns": T0 + i * 10**9, "end_ns": T0 + (i + 1) * 10**9, "total": v}
                        for i, v in enumerate(per_window)
                    ],
                }
            ],
        }

    doc = {
        "summary": {},
        "metrics": {
            "aigateway_x": {"type": "gauge", "series": []},
            "vllm:prefix_cache_hits": block("h", hits),
            "vllm:prefix_cache_queries": block("q", queries),
            "zzz_after": {"type": "gauge", "series": []},
        },
        "warmup_metrics": {
            "vllm:prefix_cache_hits": block("h", [999]),
            "vllm:prefix_cache_queries": block("q", [1]),
        },
    }
    path.write_text(json.dumps(doc, indent=2))


def test_prefix_cache_hit_rate_comes_from_the_engine_counters(tmp_path):
    from llm_module.agentic_traces.charts import (
        HIT_RATE_CUMULATIVE,
        HIT_RATE_ROLLING,
        load_prefix_cache_hit_rate,
    )

    path = tmp_path / "server_metrics_export.json"
    _server_metrics(path, hits=[0, 50, 100, 0], queries=[100, 100, 100, 0])

    series = load_prefix_cache_hit_rate(path, T0)

    # Cumulative: 0/100, 50/200, 150/300, then the empty window adds nothing.
    assert [round(v, 1) for _, v in series[HIT_RATE_CUMULATIVE]] == [0.0, 25.0, 50.0, 50.0]
    # The warmup phase's counters (999 hits / 1 query) are not mixed in.
    assert max(v for _, v in series[HIT_RATE_ROLLING]) <= 100.0


def test_load_chart_shows_hit_rate_when_the_engine_metrics_exist(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    _server_metrics(artifacts / "server_metrics_export.json", [10, 20, 30], [40, 40, 40])

    written = write_point_charts(artifacts, tmp_path / "charts")

    svg = (tmp_path / "charts" / written["load"]).read_text()
    assert "Prefix cache hit rate" in svg and "Tokens in flight" not in svg


def test_load_chart_falls_back_to_tokens_in_flight_without_engine_metrics(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)

    written = write_point_charts(artifacts, tmp_path / "charts")

    svg = (tmp_path / "charts" / written["load"]).read_text()
    assert "Tokens in flight" in svg and "Prefix cache hit rate" not in svg


def test_every_chart_has_a_mermaid_twin_for_the_job_summary(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    out = tmp_path / "charts"

    written = write_point_charts(artifacts, out)

    for name in written:
        fragment = (out / f"{name}.mmd").read_text()
        assert fragment.startswith("```mermaid") and "xychart-beta" in fragment
        # xychart only accepts plain numbers in its data arrays.
        for line in fragment.splitlines():
            if line.strip().startswith(("line [", "bar [")):
                values = line.split("[", 1)[1].rstrip("]").split(", ")
                assert all(float(v) == float(v) for v in values)


def test_rolling_window_shrinks_for_short_runs():
    from llm_module.agentic_traces.charts import _rolling_window

    assert _rolling_window(20_000) == 50
    assert _rolling_window(57) == 5
    assert _rolling_window(300) == 30


def test_mermaid_steps_average_rather_than_sample():
    from llm_module.agentic_traces.charts import _MERMAID_POINTS, _mm_grid

    # A series that is 0 except one spike per step must not read as all zeros.
    points = [(i / 10, 10.0 if i % 10 == 5 else 0.0) for i in range(_MERMAID_POINTS * 10)]
    grid = _mm_grid(points, float(_MERMAID_POINTS))
    # Each step holds about ten points with one spike: ~1.0, never 0.
    assert all(0.8 < v < 1.2 for v in grid)
