# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the spec-decode AIPerf driver's worker-metrics wiring.

Covers the benchmark-side fix for scraping ``vllm:spec_decode_*`` counters
off the worker(s) rather than the spec-decode-unaware Dynamo frontend:

* ``_worker_metrics_urls`` normalizing ``--spec-decode-metrics-url``
  entries (reusing the prefix-cache ``_normalize_metrics_url`` helper).
* ``run`` snapshotting/scraping the worker endpoint(s) while AIPerf load
  stays on the frontend ``--url``.
* Multi-worker deployments summing before/after deltas across endpoints.
* The fallback to the load target when the flag is unset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_module.config import DriverContext, ServerConnection
from llm_module.drivers import aiperf_spec_decode as driver_mod
from llm_module.drivers.aiperf_spec_decode import (
    AIPerfSpecDecodeDriver,
    _worker_metrics_urls,
)
from llm_module.spec_decode import SpecDecodeRun

FRONTEND = "http://dynamo-frontend:8000"


def _server(**overrides) -> ServerConnection:
    base = dict(
        base_url=FRONTEND,
        service_port=8000,
        model="m",
    )
    base.update(overrides)
    return ServerConnection(**base)


def _run() -> SpecDecodeRun:
    return SpecDecodeRun(
        public_dataset="speed_bench_coding",
        max_concurrency=1,
        num_prompts=4,
    )


def _driver(tmp_path: Path) -> AIPerfSpecDecodeDriver:
    return AIPerfSpecDecodeDriver(
        venv_python=Path("/tmp/venv/bin/python"),
        artifact_root=tmp_path / "artifacts",
        model_repo="m",
        model_id="m",
        output_dir=tmp_path / "out",
    )


def _patch_aiperf(monkeypatch, captured: dict):
    """Stub out the AIPerf subprocess + summary parse with a passing run."""

    def _fake_run_command(cmd, *, env, timeout_s):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(driver_mod, "run_command", _fake_run_command)
    monkeypatch.setattr(
        driver_mod,
        "_parse_aiperf_output",
        lambda artifact_dir: {
            "completed": 4,
            "mean_ttft_ms": 12.0,
            "mean_e2el_ms": 100.0,
        },
    )


class TestWorkerMetricsUrls:
    def test_bare_host_port_gets_scheme_and_metrics_path(self):
        server = _server(spec_decode_metrics_urls=("worker-a:9000",))
        assert _worker_metrics_urls(server) == ("http://worker-a:9000/metrics",)

    def test_full_url_and_existing_path_preserved(self):
        server = _server(
            spec_decode_metrics_urls=(
                "https://host.example.com:8443/metrics",
                "worker-b:9000/metrics",
            )
        )
        assert _worker_metrics_urls(server) == (
            "https://host.example.com:8443/metrics",
            "http://worker-b:9000/metrics",
        )

    def test_blank_entries_are_skipped(self):
        server = _server(spec_decode_metrics_urls=("", "  ", "worker-a:9000"))
        assert _worker_metrics_urls(server) == ("http://worker-a:9000/metrics",)

    def test_empty_when_flag_unset(self):
        assert _worker_metrics_urls(_server()) == ()


class TestRunScrapesWorkerEndpoints:
    def test_worker_scrape_with_load_on_frontend(self, monkeypatch, tmp_path):
        """--spec-decode-metrics-url redirects only the scrape, not the load."""
        captured: dict = {}
        _patch_aiperf(monkeypatch, captured)

        scrape_calls = []

        def _fake_fetch_multi(urls, *, timeout=10.0):
            scrape_calls.append(("before", tuple(urls)))
            return {"vllm:spec_decode_num_draft_tokens_total": 10.0}

        def _fake_scrape_multi(urls, before):
            scrape_calls.append(("after", tuple(urls)))
            assert before == {"vllm:spec_decode_num_draft_tokens_total": 10.0}
            return {
                "acceptance_rate": 0.5,
                "accepted_tokens": 90.0,
                "draft_tokens": 180.0,
                "num_drafts": 45.0,
                "mean_accepted_length": 3.0,
                "accepted_per_pos": [(0, 60.0)],
            }

        monkeypatch.setattr(
            driver_mod, "fetch_prometheus_counters_multi", _fake_fetch_multi
        )
        monkeypatch.setattr(
            driver_mod, "scrape_spec_decode_metrics_multi", _fake_scrape_multi
        )
        # The single-endpoint path must not be touched when workers are set.
        monkeypatch.setattr(
            driver_mod,
            "fetch_prometheus_counters",
            lambda url: pytest.fail("scraped the load target"),
        )
        monkeypatch.setattr(
            driver_mod,
            "scrape_spec_decode_metrics",
            lambda url, before: pytest.fail("scraped the load target"),
        )

        server = _server(
            spec_decode_metrics_urls=("worker-a:9000", "worker-b:9000/metrics")
        )
        result = _driver(tmp_path).run(
            _run(), server, DriverContext(output_dir=tmp_path / "out")
        )

        assert result.return_code == 0
        # Load target stays on the frontend.
        cmd = captured["cmd"]
        assert cmd[cmd.index("--url") + 1] == FRONTEND
        # Both snapshots hit the two normalized worker endpoints.
        expected = ("http://worker-a:9000/metrics", "http://worker-b:9000/metrics")
        assert scrape_calls == [("before", expected), ("after", expected)]
        # The acceptance block lands in the payload.
        assert result.payload["spec_decode_metrics"]["acceptance_rate"] == 0.5
        assert result.payload["spec_decode_metrics"]["mean_accepted_length"] == 3.0

    def test_fallback_scrapes_load_target_when_flag_unset(self, monkeypatch, tmp_path):
        captured: dict = {}
        _patch_aiperf(monkeypatch, captured)

        scrape_calls = []

        def _fake_fetch(url):
            scrape_calls.append(("before", url))
            return {}

        def _fake_scrape(url, before):
            scrape_calls.append(("after", url))
            return {
                "acceptance_rate": 0.0,
                "accepted_tokens": 0.0,
                "draft_tokens": 0.0,
                "num_drafts": None,
                "mean_accepted_length": None,
                "accepted_per_pos": [],
            }

        monkeypatch.setattr(driver_mod, "fetch_prometheus_counters", _fake_fetch)
        monkeypatch.setattr(driver_mod, "scrape_spec_decode_metrics", _fake_scrape)
        monkeypatch.setattr(
            driver_mod,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: pytest.fail("used the multi-endpoint path"),
        )

        result = _driver(tmp_path).run(
            _run(), _server(), DriverContext(output_dir=tmp_path / "out")
        )

        assert result.return_code == 0
        assert scrape_calls == [("before", FRONTEND), ("after", FRONTEND)]

    def test_failed_before_snapshot_still_runs(self, monkeypatch, tmp_path):
        """A failed before-snapshot degrades to before={} instead of aborting."""
        captured: dict = {}
        _patch_aiperf(monkeypatch, captured)

        def _fail_multi(urls, *, timeout=10.0):
            raise RuntimeError("no spec-decode metrics endpoint responded")

        monkeypatch.setattr(driver_mod, "fetch_prometheus_counters_multi", _fail_multi)
        monkeypatch.setattr(
            driver_mod,
            "scrape_spec_decode_metrics_multi",
            lambda urls, before: {
                "acceptance_rate": 0.25,
                "accepted_tokens": 5.0,
                "draft_tokens": 20.0,
                "num_drafts": 5.0,
                "mean_accepted_length": 2.0,
                "accepted_per_pos": [],
            },
        )

        server = _server(spec_decode_metrics_urls=("worker-a:9000",))
        result = _driver(tmp_path).run(
            _run(), server, DriverContext(output_dir=tmp_path / "out")
        )

        assert result.return_code == 0
        assert result.payload["spec_decode_metrics"]["acceptance_rate"] == 0.25

    def test_failed_after_scrape_omits_acceptance_block(self, monkeypatch, tmp_path):
        captured: dict = {}
        _patch_aiperf(monkeypatch, captured)

        monkeypatch.setattr(
            driver_mod,
            "fetch_prometheus_counters_multi",
            lambda urls, *, timeout=10.0: {},
        )

        def _fail_scrape(urls, before):
            raise ConnectionError("refused")

        monkeypatch.setattr(
            driver_mod, "scrape_spec_decode_metrics_multi", _fail_scrape
        )

        server = _server(spec_decode_metrics_urls=("worker-a:9000",))
        result = _driver(tmp_path).run(
            _run(), server, DriverContext(output_dir=tmp_path / "out")
        )

        assert result.return_code == 0
        assert "spec_decode_metrics" not in result.payload
