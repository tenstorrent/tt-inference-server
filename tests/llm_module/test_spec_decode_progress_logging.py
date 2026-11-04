# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Progress logging for the spec-decode sweep: per-run heartbeat, aiperf.log
tail on failure, and the health wait's /v1/models fallback."""

import json
import logging
import time

import pytest
import requests

from llm_module.drivers import aiperf_spec_decode as driver_mod
from test_module.llm_tests import spec_decode_tests


def test_heartbeat_logs_while_block_runs(caplog):
    caplog.set_level(logging.INFO)
    with driver_mod._heartbeat("run-a", time.monotonic(), interval_s=0.05):
        time.sleep(0.3)
    assert any("run-a still running" in r.getMessage() for r in caplog.records)


def test_heartbeat_logs_progress_status(caplog):
    caplog.set_level(logging.INFO)
    with driver_mod._heartbeat(
        "run-a", time.monotonic(), interval_s=0.05, progress=lambda e: "3/8 done"
    ):
        time.sleep(0.3)
    assert any("run-a 3/8 done" in r.getMessage() for r in caplog.records)


def _raw_line(phase="profiling", error=None, status=200):
    record = {"metadata": {"benchmark_phase": phase}, "status": status}
    if error:
        record["error"] = error
    return json.dumps(record) + "\n"


def test_progress_counts_raw_records_incrementally(tmp_path):
    raw = tmp_path / "raw_records" / "raw_records_processor_0.jsonl"
    raw.parent.mkdir()
    raw.write_text(_raw_line() + _raw_line(phase="warmup") + _raw_line(status=500))
    progress = driver_mod._RunProgress(artifact_dir=tmp_path, total=8)
    assert progress(10.0) == "[#####---------------] 2/8 (25%), 1 errors, ETA ~30s"
    # A partial trailing line waits for the next beat.
    with raw.open("a") as fh:
        fh.write(_raw_line(error={"message": "x"}) + _raw_line().rstrip("\n"))
    status = progress(20.0)
    assert "3/8 (38%), 2 errors" in status and "ETA ~33s" in status
    with raw.open("a") as fh:
        fh.write("\n")
    assert "4/8 (50%)" in progress(30.0)


def test_progress_without_total_counts_only(tmp_path):
    progress = driver_mod._RunProgress(artifact_dir=tmp_path, total=None)
    assert progress(5.0) == "0 requests done"


def test_aiperf_log_tail_is_logged(tmp_path, caplog):
    log = tmp_path / "logs" / "aiperf.log"
    log.parent.mkdir()
    log.write_text("\n".join(f"line {i}" for i in range(50)))
    driver_mod._log_aiperf_tail(tmp_path, lines=3)
    message = caplog.records[-1].getMessage()
    assert "line 49" in message and "line 46" not in message


class _Resp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


_MODELS = {"object": "list", "data": [{"id": "m"}]}


def _fake_get(monkeypatch, health_status, models=_MODELS):
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append(url)
        if url.endswith("/health"):
            return _Resp(health_status)
        return _Resp(200, models)

    monkeypatch.setattr(requests, "get", fake_get)
    return seen


@pytest.mark.parametrize("status", [404, 405, 501])
def test_health_wait_accepts_v1_models_when_health_is_missing(monkeypatch, status):
    seen = _fake_get(monkeypatch, status)
    assert spec_decode_tests._wait_for_url_healthy(
        "https://gw.example:443", timeout=5, interval=0
    )
    assert seen == ["https://gw.example:443/health", "https://gw.example:443/v1/models"]


def test_health_wait_ignores_v1_models_when_health_is_unhealthy(monkeypatch):
    seen = _fake_get(monkeypatch, 503)
    assert not spec_decode_tests._wait_for_url_healthy(
        "http://127.0.0.1:8080", timeout=0.2, interval=0.05
    )
    assert all(url.endswith("/health") for url in seen)


@pytest.mark.parametrize("health_status", [200, 404])
@pytest.mark.parametrize("models", [{"object": "list", "data": []}, None])
def test_health_wait_needs_a_listed_model(monkeypatch, caplog, health_status, models):
    # Dynamo's frontend: /health 200 with no workers, empty model list.
    _fake_get(monkeypatch, health_status, models)
    assert not spec_decode_tests._wait_for_url_healthy(
        "http://127.0.0.1:8080", timeout=0.2, interval=0.05
    )
    assert "no models listed" in caplog.text


def test_health_wait_logs_last_results_on_failure(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        requests, "get", lambda url, headers=None, timeout=None: _Resp(503)
    )
    assert not spec_decode_tests._wait_for_url_healthy(
        "http://127.0.0.1:8080", timeout=0.3, interval=0.05, log_every=0.1
    )
    messages = [r.getMessage() for r in caplog.records]
    assert any("still waiting" in m and "HTTP 503" in m for m in messages)
    assert any("not healthy after" in m and "HTTP 503" in m for m in messages)
