# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the prefix-cache AIPerf driver's worker-metrics wiring.

Covers the benchmark-side fix for scraping ``tt_prefix_cache_*`` counters
off the cpp_server worker (not the prefix-unaware Dynamo frontend):

* ``_normalize_metrics_url`` URL handling.
* ``_build_aiperf_cmd`` emitting ``--server-metrics`` while keeping load
  on the frontend ``--url``.
* ``_parse_server_metrics_for_prefix_cache`` recognizing the ``tt_`` names
  and summing hit/query deltas across ``endpoint_url``-tagged series.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_module.drivers.aiperf_prefix_cache import (
    _build_aiperf_cmd,
    _normalize_metrics_url,
    _parse_server_metrics_for_prefix_cache,
)
from llm_module.prefix_cache import PrefixCacheRun


def _synthetic_run() -> PrefixCacheRun:
    return PrefixCacheRun(
        scenario="baseline",
        label="baseline_isl128_c1_constant",
        isl_mean=128,
        isl_stddev=0,
        osl_mean=128,
        osl_stddev=0,
        concurrency=1,
        request_count=8,
        arrival_pattern="constant",
    )


class TestNormalizeMetricsUrl:
    def test_bare_host_port_gets_scheme_and_metrics_path(self):
        assert (
            _normalize_metrics_url("worker-a:9000")
            == "http://worker-a:9000/metrics"
        )

    def test_existing_path_is_preserved(self):
        assert (
            _normalize_metrics_url("worker-a:9000/metrics")
            == "http://worker-a:9000/metrics"
        )

    def test_full_url_passed_through(self):
        assert (
            _normalize_metrics_url("https://host.example.com:8443/metrics")
            == "https://host.example.com:8443/metrics"
        )

    def test_root_path_is_replaced_with_metrics(self):
        assert _normalize_metrics_url("http://h:9000/") == "http://h:9000/metrics"

    def test_whitespace_is_trimmed(self):
        assert _normalize_metrics_url("  h:9000 ") == "http://h:9000/metrics"


class TestBuildAiperfCmd:
    def test_server_metrics_emitted_for_each_worker(self):
        cmd = _build_aiperf_cmd(
            run=_synthetic_run(),
            venv_python=Path("/tmp/venv/bin/python"),
            model_name="m",
            tokenizer="m",
            url="http://dynamo-frontend:8000",
            artifact_dir="/tmp/art",
            auth_token="",
            metrics_urls=("worker-a:9000", "worker-b:9000/metrics"),
        )
        # Load target (frontend) is untouched.
        assert cmd[cmd.index("--url") + 1] == "http://dynamo-frontend:8000"
        # --server-metrics carries both normalized worker endpoints, in order.
        idx = cmd.index("--server-metrics")
        assert cmd[idx + 1] == "http://worker-a:9000/metrics"
        assert cmd[idx + 2] == "http://worker-b:9000/metrics"
        # JSONL export stays pinned so the parser can read it.
        assert cmd[cmd.index("--server-metrics-formats") + 1] == "jsonl"

    def test_server_metrics_omitted_when_no_workers(self):
        cmd = _build_aiperf_cmd(
            run=_synthetic_run(),
            venv_python=Path("/tmp/venv/bin/python"),
            model_name="m",
            tokenizer="m",
            url="http://dynamo-frontend:8000",
            artifact_dir="/tmp/art",
            auth_token="",
        )
        assert "--server-metrics" not in cmd

    def test_blank_worker_urls_are_skipped(self):
        cmd = _build_aiperf_cmd(
            run=_synthetic_run(),
            venv_python=Path("/tmp/venv/bin/python"),
            model_name="m",
            tokenizer="m",
            url="http://dynamo-frontend:8000",
            artifact_dir="/tmp/art",
            auth_token="",
            metrics_urls=("", "worker-a:9000"),
        )
        idx = cmd.index("--server-metrics")
        assert cmd[idx + 1] == "http://worker-a:9000/metrics"
        # Only the one real endpoint follows the flag (next token is a flag).
        assert cmd[idx + 2].startswith("--")


def _write_jsonl(path: Path, lines: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


class TestParseServerMetrics:
    def test_tt_counters_single_worker(self, tmp_path):
        jsonl = tmp_path / "server_metrics_export.jsonl"
        url = "http://worker-a:9000/metrics"
        _write_jsonl(
            jsonl,
            [
                {
                    "endpoint_url": url,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 10.0}],
                        "tt_prefix_cache_queries_total": [{"value": 20.0}],
                    },
                },
                {
                    "endpoint_url": url,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 40.0}],
                        "tt_prefix_cache_queries_total": [{"value": 60.0}],
                    },
                },
            ],
        )
        out = _parse_server_metrics_for_prefix_cache(tmp_path)
        # delta = last - first: hits 30, queries 40 -> 0.75
        assert out["prefix_cache_hits_delta"] == 30.0
        assert out["prefix_cache_queries_delta"] == 40.0
        assert out["prefix_cache_hit_rate"] == 0.75
        assert out["prefix_cache_hits_final"] == 40.0
        assert out["prefix_cache_queries_final"] == 60.0

    def test_sums_deltas_across_endpoints(self, tmp_path):
        jsonl = tmp_path / "server_metrics_export.jsonl"
        a = "http://worker-a:9000/metrics"
        b = "http://worker-b:9000/metrics"
        # Lines interleave the two workers, as AIPerf writes one line per
        # endpoint per scrape. A flat last-first would be nonsensical;
        # per-endpoint deltas must be summed.
        _write_jsonl(
            jsonl,
            [
                {
                    "endpoint_url": a,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 0.0}],
                        "tt_prefix_cache_queries_total": [{"value": 0.0}],
                    },
                },
                {
                    "endpoint_url": b,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 100.0}],
                        "tt_prefix_cache_queries_total": [{"value": 100.0}],
                    },
                },
                {
                    "endpoint_url": a,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 30.0}],
                        "tt_prefix_cache_queries_total": [{"value": 50.0}],
                    },
                },
                {
                    "endpoint_url": b,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 170.0}],
                        "tt_prefix_cache_queries_total": [{"value": 150.0}],
                    },
                },
            ],
        )
        out = _parse_server_metrics_for_prefix_cache(tmp_path)
        # worker A delta: hits 30, queries 50; worker B delta: hits 70,
        # queries 50. Summed: hits 100, queries 100 -> 1.0.
        assert out["prefix_cache_hits_delta"] == 100.0
        assert out["prefix_cache_queries_delta"] == 100.0
        assert out["prefix_cache_hit_rate"] == 1.0

    def test_missing_counters_returns_nulls(self, tmp_path):
        jsonl = tmp_path / "server_metrics_export.jsonl"
        _write_jsonl(
            jsonl,
            [
                {
                    "endpoint_url": "http://frontend:8000/metrics",
                    "metrics": {"vllm:num_requests_running": [{"value": 1.0}]},
                }
            ],
        )
        out = _parse_server_metrics_for_prefix_cache(tmp_path)
        assert out["prefix_cache_hit_rate"] is None
        assert out["prefix_cache_hits_delta"] is None
