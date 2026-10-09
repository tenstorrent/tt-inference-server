# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Progress logging for the spec-decode sweep: per-run heartbeat, aiperf.log
tail on failure, and the health wait's /v1/models fallback."""

import logging
import time

import requests

from llm_module.drivers import aiperf_spec_decode as driver_mod
from test_module.llm_tests import spec_decode_tests


def test_heartbeat_logs_while_block_runs(caplog):
    caplog.set_level(logging.INFO)
    with driver_mod._heartbeat("run-a", time.monotonic(), interval_s=0.05):
        time.sleep(0.3)
    assert any("run-a still running" in r.getMessage() for r in caplog.records)


def test_aiperf_log_tail_is_logged(tmp_path, caplog):
    log = tmp_path / "logs" / "aiperf.log"
    log.parent.mkdir()
    log.write_text("\n".join(f"line {i}" for i in range(50)))
    driver_mod._log_aiperf_tail(tmp_path, lines=3)
    message = caplog.records[-1].getMessage()
    assert "line 49" in message and "line 46" not in message


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_health_wait_accepts_v1_models_when_health_is_missing(monkeypatch):
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append(url)
        return _Resp(404 if url.endswith("/health") else 200)

    monkeypatch.setattr(requests, "get", fake_get)
    assert spec_decode_tests._wait_for_url_healthy(
        "https://gw.example:443", timeout=5, interval=0
    )
    assert seen == ["https://gw.example:443/health", "https://gw.example:443/v1/models"]


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
