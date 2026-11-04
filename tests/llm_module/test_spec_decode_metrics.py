# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the spec-decode Prometheus scrape helpers.

Covers the multi-worker scrape used in Dynamo deployments
(``--spec-decode-metrics-url``): values for the same series are summed
across worker endpoints so the before/after delta of the merged snapshot
equals the sum of the per-worker deltas, and the ``tt_spec_decode_*``
cpp_server spellings are recognized alongside ``vllm:spec_decode_*``.
"""

from __future__ import annotations

import pytest

from llm_module.spec_decode import metrics
from llm_module.spec_decode.metrics import (
    fetch_prometheus_counters_multi,
    parse_prometheus_text,
    scrape_spec_decode_metrics_multi,
)

VLLM_TEXT = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Accepted tokens.
vllm:spec_decode_num_accepted_tokens_total{model="m"} 120.0
vllm:spec_decode_num_draft_tokens_total{model="m"} 200.0
vllm:spec_decode_num_drafts_total{model="m"} 40.0
vllm:spec_decode_num_accepted_tokens_per_pos{model="m",position="0"} 60.0
vllm:spec_decode_num_accepted_tokens_per_pos{model="m",position="1"} 40.0
vllm:num_requests_running{model="m"} 3.0
"""

TT_TEXT = """\
tt_spec_decode_num_accepted_tokens_total 30.0
tt_spec_decode_num_draft_tokens_total 50.0
tt_spec_decode_num_drafts_total 10.0
tt_spec_decode_num_accepted_tokens_per_pos{position="0"} 20.0
tt_spec_decode_num_accepted_tokens_per_pos{position="1"} 10.0
tt_prefix_cache_hits_total 7.0
"""


class TestParsePrometheusText:
    def test_vllm_counters_parsed_by_default(self):
        out = parse_prometheus_text(VLLM_TEXT)
        assert out['vllm:spec_decode_num_accepted_tokens_total{model="m"}'] == 120.0
        assert out['vllm:spec_decode_num_drafts_total{model="m"}'] == 40.0
        # Non spec-decode series are dropped.
        assert not any("num_requests_running" in k for k in out)

    def test_tt_counters_recognized_alongside_vllm(self):
        out = parse_prometheus_text(TT_TEXT)
        assert out["tt_spec_decode_num_accepted_tokens_total"] == 30.0
        assert out["tt_spec_decode_num_draft_tokens_total"] == 50.0
        # Other tt_* families are not spec-decode counters.
        assert not any("prefix_cache" in k for k in out)

    def test_single_prefix_still_supported(self):
        out = parse_prometheus_text(
            VLLM_TEXT + TT_TEXT, prefix=metrics.TT_SPEC_DECODE_PREFIX
        )
        assert out == {
            "tt_spec_decode_num_accepted_tokens_total": 30.0,
            "tt_spec_decode_num_draft_tokens_total": 50.0,
            "tt_spec_decode_num_drafts_total": 10.0,
            'tt_spec_decode_num_accepted_tokens_per_pos{position="0"}': 20.0,
            'tt_spec_decode_num_accepted_tokens_per_pos{position="1"}': 10.0,
        }


class TestFetchPrometheusCountersMulti:
    def test_sums_series_across_endpoints(self, monkeypatch):
        scrapes = {
            "http://worker-a:9000/metrics": parse_prometheus_text(VLLM_TEXT),
            "http://worker-b:9000/metrics": parse_prometheus_text(TT_TEXT),
        }
        monkeypatch.setattr(
            metrics, "fetch_metrics_endpoint", lambda url, *, timeout: scrapes[url]
        )

        out = fetch_prometheus_counters_multi(list(scrapes))

        assert out['vllm:spec_decode_num_accepted_tokens_total{model="m"}'] == 120.0
        assert out["tt_spec_decode_num_accepted_tokens_total"] == 30.0

    def test_same_series_on_two_workers_is_summed(self, monkeypatch):
        text_a = "vllm:spec_decode_num_draft_tokens_total 100.0\n"
        text_b = "vllm:spec_decode_num_draft_tokens_total 250.0\n"
        scrapes = {
            "http://worker-a:9000/metrics": parse_prometheus_text(text_a),
            "http://worker-b:9000/metrics": parse_prometheus_text(text_b),
        }
        monkeypatch.setattr(
            metrics, "fetch_metrics_endpoint", lambda url, *, timeout: scrapes[url]
        )

        out = fetch_prometheus_counters_multi(list(scrapes))

        assert out["vllm:spec_decode_num_draft_tokens_total"] == 350.0

    def test_failing_endpoint_is_skipped(self, monkeypatch):
        def _fake_fetch(url, *, timeout):
            if "worker-b" in url:
                raise ConnectionError("refused")
            return parse_prometheus_text(VLLM_TEXT)

        monkeypatch.setattr(metrics, "fetch_metrics_endpoint", _fake_fetch)

        out = fetch_prometheus_counters_multi(
            ["http://worker-a:9000/metrics", "http://worker-b:9000/metrics"]
        )

        assert out['vllm:spec_decode_num_draft_tokens_total{model="m"}'] == 200.0

    def test_raises_when_no_endpoint_responds(self, monkeypatch):
        def _fake_fetch(url, *, timeout):
            raise ConnectionError("refused")

        monkeypatch.setattr(metrics, "fetch_metrics_endpoint", _fake_fetch)

        with pytest.raises(RuntimeError, match="no spec-decode metrics endpoint"):
            fetch_prometheus_counters_multi(
                ["http://worker-a:9000/metrics", "http://worker-b:9000/metrics"]
            )


class TestScrapeSpecDecodeMetricsMulti:
    def test_acceptance_from_summed_deltas(self, monkeypatch):
        # before: worker A 10/20 accepted/draft + 5 drafts, worker B 30/60 + 15
        # after:  worker A 40/80 + 20 drafts, worker B 90/180 + 45 drafts
        # summed deltas: accepted 90, draft 180, num_drafts 45
        # -> acceptance_rate 0.5, mean_accepted_length 1 + 90/45 = 3.0
        before = {
            "vllm:spec_decode_num_accepted_tokens_total": 40.0,
            "vllm:spec_decode_num_draft_tokens_total": 80.0,
            "vllm:spec_decode_num_drafts_total": 20.0,
        }
        after = {
            "vllm:spec_decode_num_accepted_tokens_total": 130.0,
            "vllm:spec_decode_num_draft_tokens_total": 260.0,
            "vllm:spec_decode_num_drafts_total": 65.0,
        }
        monkeypatch.setattr(
            metrics,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: after,
        )

        out = scrape_spec_decode_metrics_multi(
            ["http://worker-a:9000/metrics", "http://worker-b:9000/metrics"], before
        )

        assert out["accepted_tokens"] == 90.0
        assert out["draft_tokens"] == 180.0
        assert out["num_drafts"] == 45.0
        assert out["acceptance_rate"] == pytest.approx(0.5)
        assert out["mean_accepted_length"] == pytest.approx(3.0)

    def test_per_position_summed_across_workers(self, monkeypatch):
        after = {
            'vllm:spec_decode_num_accepted_tokens_per_pos{position="0"}': 100.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos{position="1"}': 50.0,
            'tt_spec_decode_num_accepted_tokens_per_pos{position="0"}': 20.0,
        }
        monkeypatch.setattr(
            metrics,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: after,
        )

        out = scrape_spec_decode_metrics_multi(["http://w:9000/metrics"], {})

        # vLLM and tt series for the same position are summed.
        assert out["accepted_per_pos"] == [(0, 120.0), (1, 50.0)]

    def test_tt_counters_drive_acceptance(self, monkeypatch):
        after = {
            "tt_spec_decode_num_accepted_tokens_total": 30.0,
            "tt_spec_decode_num_draft_tokens_total": 50.0,
            "tt_spec_decode_num_drafts_total": 10.0,
        }
        monkeypatch.setattr(
            metrics,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: after,
        )

        out = scrape_spec_decode_metrics_multi(["http://w:9000/metrics"], {})

        assert out["acceptance_rate"] == pytest.approx(0.6)
        assert out["mean_accepted_length"] == pytest.approx(4.0)

    def test_tt_engine_family_drives_acceptance_and_length(self, monkeypatch):
        def exposition(accepted, draft, drafts):
            text = ""
            for method, scale in (("dflash", 1), ("mtp", 0)):
                labels = f'{{dynamo_component="backend",method="{method}"}}'
                text += (
                    f"tt_engine_spec_decode_accepted_tokens_total{labels} {accepted * scale}\n"
                    f"tt_engine_spec_decode_draft_tokens_total{labels} {draft * scale}\n"
                    f"tt_engine_spec_decode_drafts_total{labels} {drafts * scale}\n"
                )
            return parse_prometheus_text(text)

        before = exposition(100, 300, 50)
        monkeypatch.setattr(
            metrics,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: exposition(130, 400, 60),
        )

        out = scrape_spec_decode_metrics_multi(["http://w:20020/metrics"], before)

        assert out["accepted_tokens"] == 30.0
        assert out["draft_tokens"] == 100.0
        assert out["acceptance_rate"] == pytest.approx(0.3)
        assert out["num_drafts"] == 10.0
        assert out["mean_accepted_length"] == pytest.approx(4.0)

    def test_cpp_server_worker_totals_drive_acceptance(self, monkeypatch):
        def exposition(accepts, rejects):
            return parse_prometheus_text(
                "# TYPE tt_worker_spec_accepts_total gauge\n"
                f'tt_worker_spec_accepts_total{{worker_id="0"}} {accepts}\n'
                f'tt_worker_spec_rejects_total{{worker_id="0"}} {rejects}\n'
                'tt_worker_total_acceptance_rate{worker_id="0"} 0.4\n'
            )

        before = exposition(100, 100)
        monkeypatch.setattr(
            metrics,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: exposition(130, 170),
        )

        out = scrape_spec_decode_metrics_multi(["http://w:8000/metrics"], before)

        assert out["accepted_tokens"] == 30.0
        assert out["draft_tokens"] == 100.0
        assert out["acceptance_rate"] == pytest.approx(0.3)
        assert out["mean_accepted_length"] is None

    def test_no_draft_tokens_yields_zero_rate_and_null_length(self, monkeypatch):
        monkeypatch.setattr(
            metrics,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: {},
        )

        out = scrape_spec_decode_metrics_multi(["http://w:9000/metrics"], {})

        assert out["acceptance_rate"] == 0.0
        assert out["mean_accepted_length"] is None
        assert out["num_drafts"] is None
