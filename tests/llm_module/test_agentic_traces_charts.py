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
                "Effective Prefill Concurrency",
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
                        {
                            "start_ns": T0 + i * 10**9,
                            "end_ns": T0 + (i + 1) * 10**9,
                            "total": v,
                        }
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
        load_engine_metrics,
    )

    path = tmp_path / "server_metrics_export.json"
    _server_metrics(path, hits=[0, 50, 100, 0], queries=[100, 100, 100, 0])

    series = load_engine_metrics(path, T0)

    # Cumulative: 0/100, 50/200, 150/300, then the empty window adds nothing.
    assert [round(v, 1) for _, v in series[HIT_RATE_CUMULATIVE]] == [
        0.0,
        25.0,
        50.0,
        50.0,
    ]
    # The warmup phase's counters (999 hits / 1 query) are not mixed in.
    assert max(v for _, v in series[HIT_RATE_ROLLING]) <= 100.0


def test_load_chart_shows_hit_rate_when_the_engine_metrics_exist(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    _server_metrics(
        artifacts / "server_metrics_export.json", [10, 20, 30], [40, 40, 40]
    )

    written = write_point_charts(artifacts, tmp_path / "charts")

    svg = (tmp_path / "charts" / written["load"]).read_text()
    assert "Prefix cache hit rate (engine counters)" in svg


def _server_metrics_by_source(path, by_source):
    """``vllm:prompt_tokens_by_source`` per window, one series per source, next
    to hit/query counters stuck at zero, as a lane-DP TT server exports them."""

    def series(per_window, labels):
        return {
            "endpoint_url": "http://e:1/metrics",
            "labels": labels,
            "timeslices": [
                {
                    "start_ns": T0 + i * 10**9,
                    "end_ns": T0 + (i + 1) * 10**9,
                    "total": v,
                }
                for i, v in enumerate(per_window)
            ],
        }

    windows = len(next(iter(by_source.values())))
    zeros = {"type": "counter", "series": [series([0.0] * windows, {})]}
    doc = {
        "summary": {},
        "metrics": {
            "vllm:prefix_cache_hits": zeros,
            "vllm:prefix_cache_queries": zeros,
            "vllm:prompt_tokens_by_source": {
                "type": "counter",
                "series": [
                    series(v, {"source": source}) for source, v in by_source.items()
                ],
            },
        },
    }
    path.write_text(json.dumps(doc, indent=2))


def test_hit_rate_comes_from_the_token_source_partition(tmp_path):
    """External KV counts as a hit, and zeroed hit/query counters do not
    blank the panel."""
    from llm_module.agentic_traces.charts import (
        HIT_RATE_CUMULATIVE,
        HIT_RATE_SOURCE,
        SOURCE_ENGINE,
        load_engine_metrics,
    )

    path = tmp_path / "server_metrics_export.json"
    _server_metrics_by_source(
        path,
        {
            "local_cache_hit": [60, 0, 30],
            "external_kv_transfer": [25, 0, 0],
            "local_compute": [15, 0, 70],
        },
    )

    series = load_engine_metrics(path, T0)

    # 85/100, the empty window adds nothing, then 115/200.
    assert [round(v, 1) for _, v in series[HIT_RATE_CUMULATIVE]] == [
        85.0,
        85.0,
        57.5,
    ]
    assert series[HIT_RATE_SOURCE] == [(0.0, SOURCE_ENGINE)]


def test_load_chart_shows_hit_rate_from_the_token_source_partition(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    _server_metrics_by_source(
        artifacts / "server_metrics_export.json",
        {"local_cache_hit": [80, 80, 80], "local_compute": [20, 20, 20]},
    )

    written = write_point_charts(artifacts, tmp_path / "charts")

    svg = (tmp_path / "charts" / written["load"]).read_text()
    assert "Prefix cache hit rate (engine counters)" in svg


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
    points = [
        (i / 10, 10.0 if i % 10 == 5 else 0.0) for i in range(_MERMAID_POINTS * 10)
    ]
    grid = _mm_grid(points, float(_MERMAID_POINTS))
    # Each step holds about ten points with one spike: ~1.0, never 0.
    assert all(0.8 < v < 1.2 for v in grid)


def _gauge(name, per_engine):
    return {
        "type": "gauge",
        "series": [
            {
                "endpoint_url": f"http://e{i}:1/metrics",
                "labels": {"engine": "0"},
                "timeslices": [
                    {
                        "start_ns": T0 + t * 10**9 + i * 12_000_000,
                        "end_ns": T0 + (t + 1) * 10**9 + i * 12_000_000,
                        "avg": v,
                    }
                    for t, v in enumerate(values)
                ],
            }
            for i, values in enumerate(per_engine)
        ],
    }


def test_engine_kv_usage_and_queue_come_from_the_engine_gauges(tmp_path):
    from llm_module.agentic_traces.charts import (
        ENGINE_WAITING,
        KV_USAGE_PREFIX,
        load_engine_metrics,
    )

    doc = {
        "metrics": {
            "vllm:kv_cache_usage_perc": _gauge("kv", [[0.1, 0.5], [0.2, 0.25]]),
            "vllm:num_requests_waiting": _gauge("q", [[1, 3], [2, 4]]),
        },
        "warmup_metrics": {},
    }
    path = tmp_path / "server_metrics_export.json"
    path.write_text(json.dumps(doc, indent=2))

    series = load_engine_metrics(path, T0)

    # Fractions become percent, one line per engine.
    assert [v for _, v in series[KV_USAGE_PREFIX + "engine 1"]] == [10.0, 50.0]
    assert [v for _, v in series[KV_USAGE_PREFIX + "engine 2"]] == [20.0, 25.0]
    # Two engines scraped a few ms apart still land in the same second: 1+2, 3+4.
    assert [v for _, v in series[ENGINE_WAITING]] == [3.0, 7.0]


def test_many_engines_collapse_to_mean_and_busiest(tmp_path):
    from llm_module.agentic_traces.charts import KV_USAGE_PREFIX, load_engine_metrics

    doc = {
        "metrics": {
            "vllm:kv_cache_usage_perc": _gauge("kv", [[0.1 * i] for i in range(1, 7)])
        },
        "warmup_metrics": {},
    }
    path = tmp_path / "server_metrics_export.json"
    path.write_text(json.dumps(doc, indent=2))

    series = load_engine_metrics(path, T0)

    assert sorted(k for k in series if k.startswith(KV_USAGE_PREFIX)) == [
        KV_USAGE_PREFIX + "busiest engine",
        KV_USAGE_PREFIX + "mean of engines",
    ]
    assert round(series[KV_USAGE_PREFIX + "busiest engine"][0][1]) == 60


def _write_usage_slices(directory, cached, prompt):
    rows = ["Timeslice,Start_NS,End_NS,Metric,Unit,Stat,Value"]
    for i, (c, p) in enumerate(zip(cached, prompt)):
        start = T0 + i * 10**9
        for metric, value in (
            ("Usage Prompt Cache Read Tokens", c),
            ("Usage Prompt Tokens", p),
        ):
            rows.append(f"{i},{start},{start + 10**9},{metric},tokens,sum,{value}")
        rows.append(
            f"{i},{start},{start + 10**9},Tokens In Flight,tokens,avg,{1000 * (i + 1)}"
        )
        rows.append(
            f"{i},{start},{start + 10**9},Effective Prefill Concurrency,requests,avg,{i + 1}"
        )
    (directory / TIMESLICES).write_text("\n".join(rows) + "\n")


def test_hit_rate_falls_back_to_the_servers_response_usage(tmp_path):
    from llm_module.agentic_traces.charts import (
        HIT_RATE_CUMULATIVE,
        HIT_RATE_SOURCE,
        load_timeslices,
        usage_hit_rate,
    )

    _write_usage_slices(tmp_path, cached=[0, 50, 100], prompt=[100, 100, 100])
    slices = load_timeslices(
        tmp_path / TIMESLICES,
        T0,
        (),
        ("Usage Prompt Cache Read Tokens", "Usage Prompt Tokens"),
    )

    lines = usage_hit_rate(slices)

    assert [round(v, 1) for _, v in lines[HIT_RATE_CUMULATIVE]] == [0.0, 25.0, 50.0]
    assert HIT_RATE_SOURCE not in lines  # the entry point records the source


def test_load_chart_uses_usage_hit_rate_and_client_side_stand_ins(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / PROFILE_EXPORT).write_text(
        "\n".join([_record(0, 10), _record(5, 25, conv="c2")]) + "\n"
    )
    _write_usage_slices(artifacts, cached=[10, 20, 30], prompt=[40, 40, 40])

    written = write_point_charts(artifacts, tmp_path / "charts")

    svg = (tmp_path / "charts" / written["load"]).read_text()
    assert "Prefix cache hit rate (from response usage)" in svg
    # No engine metrics: client-side KV demand, and the queue is the client's view.
    assert "Tokens in flight (client-side KV demand)" in svg
    assert "Request queue (client view)" in svg
    assert "waiting in the engine queue" not in svg


def test_queue_panel_adds_the_engine_queue_when_scraped(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_artifacts(artifacts)
    doc = {
        "metrics": {"vllm:num_requests_waiting": _gauge("q", [[1, 3], [2, 4]])},
        "warmup_metrics": {},
    }
    (artifacts / "server_metrics_export.json").write_text(json.dumps(doc, indent=2))

    written = write_point_charts(artifacts, tmp_path / "charts")

    svg = (tmp_path / "charts" / written["load"]).read_text()
    assert "Request queue (client view and engine)" in svg
    assert "waiting in the engine queue" in svg
