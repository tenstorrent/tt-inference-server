# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the spec-decode driver's worker-metrics scrape wiring.

Covers the benchmark-side fix for scraping ``vllm:spec_decode_*`` acceptance
counters off the vLLM worker(s) (not the spec-decode-unaware Dynamo
frontend that receives the load):

* ``_normalize_metrics_url`` URL handling.
* ``_resolve_metrics_targets`` preferring the worker endpoint(s) over the
  load target, and falling back to the load target when unset.
* ``scrape_spec_decode_metrics_multi`` summing before/after counter deltas
  across multiple workers before deriving acceptance-rate / mean-accepted-
  length / per-position figures.
"""

from __future__ import annotations

import pytest

from llm_module.config import ServerConnection
from llm_module.drivers.aiperf_spec_decode import _resolve_metrics_targets
from llm_module.spec_decode import metrics as sd_metrics
from llm_module.spec_decode.metrics import (
    ACCEPTED_COUNTER,
    DRAFT_COUNTER,
    NUM_DRAFTS_COUNTER,
    PER_POS_PREFIX,
    _normalize_metrics_url,
    scrape_spec_decode_metrics_multi,
    snapshot_worker_counters,
)


class TestNormalizeMetricsUrl:
    def test_bare_host_port_gets_scheme_and_metrics_path(self):
        assert _normalize_metrics_url("worker-a:9000") == "http://worker-a:9000/metrics"

    def test_existing_path_is_preserved(self):
        assert (
            _normalize_metrics_url("worker-a:9000/metrics")
            == "http://worker-a:9000/metrics"
        )

    def test_full_https_url_passed_through(self):
        assert (
            _normalize_metrics_url("https://host.example.com:8443/metrics")
            == "https://host.example.com:8443/metrics"
        )

    def test_root_path_is_replaced_with_metrics(self):
        assert _normalize_metrics_url("http://h:9000/") == "http://h:9000/metrics"

    def test_whitespace_is_trimmed(self):
        assert _normalize_metrics_url("  h:9000 ") == "http://h:9000/metrics"


class TestResolveMetricsTargets:
    def test_worker_urls_take_precedence_over_load_target(self):
        server = ServerConnection(
            base_url="http://frontend",
            service_port=8000,
            model="m",
            spec_decode_metrics_urls=("worker-a:9000", "worker-b:9000/metrics"),
        )
        # Load target (frontend :8000) is NOT scraped; the workers are.
        assert _resolve_metrics_targets(server) == [
            "worker-a:9000",
            "worker-b:9000/metrics",
        ]
        assert server.url_with_port not in _resolve_metrics_targets(server)

    def test_falls_back_to_load_target_when_unset(self):
        server = ServerConnection(
            base_url="http://localhost", service_port=8000, model="m"
        )
        assert _resolve_metrics_targets(server) == ["http://localhost:8000"]


def _counters(accepted: float, draft: float, num_drafts: float, **per_pos) -> dict:
    out = {
        ACCEPTED_COUNTER: accepted,
        DRAFT_COUNTER: draft,
        NUM_DRAFTS_COUNTER: num_drafts,
    }
    for pos, count in per_pos.items():
        # e.g. pos="p0" -> position="0"
        out[f'{PER_POS_PREFIX}{{position="{pos[1:]}"}}'] = count
    return out


class TestScrapeMulti:
    def test_single_worker_delta(self, monkeypatch):
        url = "http://worker-a:9000/metrics"
        before = {url: _counters(100, 200, 50)}
        monkeypatch.setattr(
            sd_metrics,
            "fetch_worker_counters",
            lambda u, **_: _counters(180, 300, 90),
        )
        result = scrape_spec_decode_metrics_multi([url], before)
        assert result["accepted_tokens"] == 80
        assert result["draft_tokens"] == 100
        assert result["acceptance_rate"] == pytest.approx(0.8)
        # 1 + accepted/num_drafts = 1 + 80/40
        assert result["mean_accepted_length"] == pytest.approx(3.0)

    def test_sums_deltas_across_workers(self, monkeypatch):
        a = "http://worker-a:9000/metrics"
        b = "http://worker-b:9000/metrics"
        before = {a: _counters(100, 100, 20), b: _counters(0, 0, 0)}
        after = {
            a: _counters(180, 200, 60),  # delta: 80 / 100 / 40
            b: _counters(40, 100, 20),  # delta: 40 / 100 / 20
        }
        monkeypatch.setattr(
            sd_metrics, "fetch_worker_counters", lambda u, **_: after[u]
        )
        result = scrape_spec_decode_metrics_multi([a, b], before)
        # Summed: accepted=120, draft=200, num_drafts=60
        assert result["accepted_tokens"] == 120
        assert result["draft_tokens"] == 200
        assert result["acceptance_rate"] == pytest.approx(0.6)
        assert result["mean_accepted_length"] == pytest.approx(1 + 120 / 60)

    def test_per_position_summed_across_workers(self, monkeypatch):
        a = "http://worker-a:9000/metrics"
        b = "http://worker-b:9000/metrics"
        before = {a: _counters(0, 0, 0), b: _counters(0, 0, 0)}
        after = {
            a: _counters(0, 0, 0, p0=10, p1=4),
            b: _counters(0, 0, 0, p0=6, p1=1),
        }
        monkeypatch.setattr(
            sd_metrics, "fetch_worker_counters", lambda u, **_: after[u]
        )
        result = scrape_spec_decode_metrics_multi([a, b], before)
        assert result["accepted_per_pos"] == [(0, 16.0), (1, 5.0)]

    def test_missing_counters_yield_zero_and_none(self, monkeypatch):
        url = "http://frontend:8000/metrics"
        monkeypatch.setattr(sd_metrics, "fetch_worker_counters", lambda u, **_: {})
        result = scrape_spec_decode_metrics_multi([url], {url: {}})
        assert result["acceptance_rate"] == 0.0
        assert result["mean_accepted_length"] is None
        assert result["num_drafts"] is None


class TestSnapshot:
    def test_keys_are_normalized(self, monkeypatch):
        monkeypatch.setattr(
            sd_metrics, "fetch_worker_counters", lambda u, **_: {ACCEPTED_COUNTER: 1.0}
        )
        snap = snapshot_worker_counters(["worker-a:9000", "worker-b:9000/metrics"])
        assert set(snap) == {
            "http://worker-a:9000/metrics",
            "http://worker-b:9000/metrics",
        }

    def test_blank_urls_skipped(self, monkeypatch):
        monkeypatch.setattr(sd_metrics, "fetch_worker_counters", lambda u, **_: {})
        snap = snapshot_worker_counters(["", "worker-a:9000"])
        assert set(snap) == {"http://worker-a:9000/metrics"}
