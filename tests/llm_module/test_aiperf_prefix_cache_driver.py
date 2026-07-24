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
    _split_role_and_url,
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
        assert _normalize_metrics_url("worker-a:9000") == "http://worker-a:9000/metrics"

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

    def test_goodput_slo_passed_as_single_token(self):
        run = _synthetic_run()
        slo = "time_to_first_token:4000 output_token_throughput_per_user:45"
        run.goodput = slo
        cmd = _build_aiperf_cmd(
            run=run,
            venv_python=Path("/tmp/venv/bin/python"),
            model_name="m",
            tokenizer="m",
            url="http://dynamo-frontend:8000",
            artifact_dir="/tmp/art",
            auth_token="",
        )
        idx = cmd.index("--goodput")
        # AIPerf's --goodput is a single-token flag (its validator splits the
        # string internally); the whole SLO must be one argv element so the
        # second pair is not consumed positionally.
        assert cmd[idx + 1] == slo
        # The next token after the SLO is a flag, not a stray KEY:VALUE pair.
        if idx + 2 < len(cmd):
            assert cmd[idx + 2].startswith("--")

    def test_goodput_omitted_when_unset(self):
        cmd = _build_aiperf_cmd(
            run=_synthetic_run(),
            venv_python=Path("/tmp/venv/bin/python"),
            model_name="m",
            tokenizer="m",
            url="http://dynamo-frontend:8000",
            artifact_dir="/tmp/art",
            auth_token="",
        )
        assert "--goodput" not in cmd


class TestSplitRoleAndUrl:
    def test_prefill_role_is_peeled_off(self):
        assert _split_role_and_url("prefill=worker0:9091") == (
            "prefill",
            "worker0:9091",
        )

    def test_decode_role_case_insensitive(self):
        assert _split_role_and_url("DECODE=http://w:9091/metrics") == (
            "decode",
            "http://w:9091/metrics",
        )

    def test_bare_host_port_has_no_role(self):
        assert _split_role_and_url("worker0:9091") == ("", "worker0:9091")

    def test_full_url_has_no_role(self):
        assert _split_role_and_url("http://worker0:9091/metrics") == (
            "",
            "http://worker0:9091/metrics",
        )

    def test_unknown_prefix_is_not_a_role(self):
        # A non-role token before ``=`` (e.g. a query string) stays part of
        # the URL, not misread as a role.
        assert _split_role_and_url("http://w:9091/metrics?foo=bar") == (
            "",
            "http://w:9091/metrics?foo=bar",
        )

    def test_role_url_normalizes_to_match_scrape_endpoint(self):
        # The role map key must line up with the endpoint_url AIPerf stamps,
        # which is the normalized URL. Prove the round-trip matches.
        role, url = _split_role_and_url("prefill=worker0:9091")
        assert role == "prefill"
        assert _normalize_metrics_url(url) == "http://worker0:9091/metrics"


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

class TestParseServerMetricsByRole:
    def test_role_split_prefill_vs_decode(self, tmp_path):
        """Prefill and decode get their own denominators, never blended."""
        jsonl = tmp_path / "server_metrics_export.jsonl"
        prefill = _normalize_metrics_url("worker-prefill:9091")
        decode = _normalize_metrics_url("worker-decode:9091")
        _write_jsonl(
            jsonl,
            [
                # Baselines (first snapshot per endpoint).
                {
                    "endpoint_url": prefill,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 0.0}],
                        "tt_prefix_cache_queries_total": [{"value": 0.0}],
                    },
                },
                {
                    "endpoint_url": decode,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 0.0}],
                        "tt_prefix_cache_queries_total": [{"value": 0.0}],
                    },
                },
                # Finals: prefill matched 200/1000 prompt tokens -> 0.2;
                # decode matched 900/1000 prompt tokens -> 0.9.
                {
                    "endpoint_url": prefill,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 200.0}],
                        "tt_prefix_cache_queries_total": [{"value": 1000.0}],
                    },
                },
                {
                    "endpoint_url": decode,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 900.0}],
                        "tt_prefix_cache_queries_total": [{"value": 1000.0}],
                    },
                },
            ],
        )
        endpoint_roles = {prefill: "prefill", decode: "decode"}
        out = _parse_server_metrics_for_prefix_cache(
            tmp_path, endpoint_roles=endpoint_roles
        )

        assert out["prefix_cache_hits_delta_prefill"] == 200.0
        assert out["prefix_cache_queries_delta_prefill"] == 1000.0
        assert out["prefix_cache_hit_rate_prefill"] == 0.2

        assert out["prefix_cache_hits_delta_decode"] == 900.0
        assert out["prefix_cache_queries_delta_decode"] == 1000.0
        assert out["prefix_cache_hit_rate_decode"] == 0.9

        # The role-blind blend (1100 / 2000 = 0.55) is exactly the bug this
        # fix removes: neither role rate may collapse to it.
        blended = (200.0 + 900.0) / (1000.0 + 1000.0)
        assert out["prefix_cache_hit_rate_prefill"] != blended
        assert out["prefix_cache_hit_rate_decode"] != blended

    def test_no_roles_preserves_legacy_combined_rate(self, tmp_path):
        """No roles supplied (every current caller, aggregated/``regular``
        deployments): identical to today — all endpoints sum into the single
        ``prefix_cache_hit_rate``. The per-role keys exist but are null."""
        jsonl = tmp_path / "server_metrics_export.jsonl"
        a = "http://worker-a:9000/metrics"
        b = "http://worker-b:9000/metrics"
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
                        "tt_prefix_cache_hits_total": [{"value": 0.0}],
                        "tt_prefix_cache_queries_total": [{"value": 0.0}],
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
                        "tt_prefix_cache_hits_total": [{"value": 70.0}],
                        "tt_prefix_cache_queries_total": [{"value": 50.0}],
                    },
                },
            ],
        )
        out = _parse_server_metrics_for_prefix_cache(tmp_path)
        # Unchanged legacy behavior: summed 100 / 100 -> 1.0.
        assert out["prefix_cache_hit_rate"] == 1.0
        assert out["prefix_cache_hits_delta"] == 100.0
        assert out["prefix_cache_queries_delta"] == 100.0
        # No role information -> no per-role rate.
        assert out["prefix_cache_hit_rate_prefill"] is None
        assert out["prefix_cache_hit_rate_decode"] is None

    def test_untagged_endpoint_falls_back_to_combined_only(self, tmp_path):
        """Partial classification (operator forgot to tag one URL): the untagged
        endpoint must not corrupt a role bucket. It still counts toward the
        combined legacy total, but only tagged endpoints populate role rates."""
        jsonl = tmp_path / "server_metrics_export.jsonl"
        decode = _normalize_metrics_url("worker-decode:9091")
        untagged = "http://mystery:9000/metrics"
        _write_jsonl(
            jsonl,
            [
                {
                    "endpoint_url": decode,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 0.0}],
                        "tt_prefix_cache_queries_total": [{"value": 0.0}],
                    },
                },
                {
                    "endpoint_url": untagged,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 0.0}],
                        "tt_prefix_cache_queries_total": [{"value": 0.0}],
                    },
                },
                {
                    "endpoint_url": decode,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 40.0}],
                        "tt_prefix_cache_queries_total": [{"value": 50.0}],
                    },
                },
                {
                    "endpoint_url": untagged,
                    "metrics": {
                        "tt_prefix_cache_hits_total": [{"value": 10.0}],
                        "tt_prefix_cache_queries_total": [{"value": 50.0}],
                    },
                },
            ],
        )
        endpoint_roles = {decode: "decode"}  # ``untagged`` deliberately absent
        out = _parse_server_metrics_for_prefix_cache(
            tmp_path, endpoint_roles=endpoint_roles
        )
        # decode reported on its own tokens: 40 / 50 = 0.8.
        assert out["prefix_cache_hit_rate_decode"] == 0.8
        # No endpoint tagged prefill -> null.
        assert out["prefix_cache_hit_rate_prefill"] is None
        # Combined legacy total still sums BOTH endpoints: 50 / 100 = 0.5.
        assert out["prefix_cache_hit_rate"] == 0.5
