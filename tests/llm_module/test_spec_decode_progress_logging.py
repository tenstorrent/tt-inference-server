# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Progress logging for the spec-decode sweep: run_command heartbeat and timeout,
AIPerf record counting, and the health wait's /v1/models fallback."""

import json
import logging
import sys

import pytest

from llm_module.drivers import aiperf_spec_decode as driver_mod
from llm_module.drivers._subprocess import run_command
from test_module.llm_tests import spec_decode_tests


def test_heartbeat_logs_while_child_runs(caplog):
    caplog.set_level(logging.INFO)
    rc = run_command(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        heartbeat_s=0.1,
        heartbeat_label="unit",
        heartbeat_status=lambda: "3/8 requests recorded",
    )
    assert rc == 0
    beats = [r.getMessage() for r in caplog.records if "still running" in r.getMessage()]
    assert beats, "expected at least one heartbeat line"
    assert beats[0].startswith("[unit] still running")
    assert "3/8 requests recorded" in beats[0]


def test_heartbeat_keeps_timeout_semantics():
    rc = run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_s=0.3,
        heartbeat_s=0.1,
    )
    assert rc == 124


def test_heartbeat_status_errors_do_not_break_the_run():
    def broken():
        raise RuntimeError("boom")

    rc = run_command(
        [sys.executable, "-c", "import time; time.sleep(0.3)"],
        heartbeat_s=0.1,
        heartbeat_status=broken,
    )
    assert rc == 0


def test_no_heartbeat_returns_child_exit_code():
    assert run_command([sys.executable, "-c", "import sys; sys.exit(3)"]) == 3


def test_progress_text_counts_records_and_errors(tmp_path):
    export = tmp_path / "sub" / "profile_export.jsonl"
    export.parent.mkdir()
    export.write_text(
        "\n".join(
            [
                json.dumps({"metadata": {}, "error": None}),
                json.dumps({"metadata": {}, "error": {"code": 500}}),
                json.dumps({"metadata": {}}),
                "",
            ]
        )
    )
    assert driver_mod._progress_text(tmp_path, 80) == "3/80 requests recorded, 1 with errors"
    assert driver_mod._progress_text(tmp_path / "missing", None) == "0 requests recorded, 0 with errors"


def test_heartbeat_interval_env(monkeypatch):
    monkeypatch.delenv("TT_SPEC_DECODE_HEARTBEAT_S", raising=False)
    assert driver_mod._heartbeat_interval_s() == driver_mod.DEFAULT_HEARTBEAT_S
    monkeypatch.setenv("TT_SPEC_DECODE_HEARTBEAT_S", "0")
    assert driver_mod._heartbeat_interval_s() == 0.0
    monkeypatch.setenv("TT_SPEC_DECODE_HEARTBEAT_S", "abc")
    assert driver_mod._heartbeat_interval_s() == driver_mod.DEFAULT_HEARTBEAT_S


@pytest.mark.parametrize("ui", ["", "simple"])
def test_ui_type_is_opt_in(monkeypatch, ui):
    from llm_module.spec_decode import SpecDecodeRun

    if ui:
        monkeypatch.setenv("TT_SPEC_DECODE_AIPERF_UI", ui)
    else:
        monkeypatch.delenv("TT_SPEC_DECODE_AIPERF_UI", raising=False)
    cmd = driver_mod._build_aiperf_cmd(
        run=SpecDecodeRun(public_dataset="speed_bench_coding", max_concurrency=1, num_prompts=80),
        venv_python=sys.executable,
        model_name="m",
        tokenizer="m",
        url="http://127.0.0.1:8080",
        artifact_dir="/tmp/x",
        auth_token="",
        tokenizer_trust_remote_code=False,
    )
    if ui:
        assert cmd[cmd.index("--ui-type") + 1] == "simple"
    else:
        assert "--ui-type" not in cmd


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_health_wait_accepts_v1_models_when_health_is_missing(monkeypatch, caplog):
    import requests

    caplog.set_level(logging.INFO)
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append(url)
        return _Resp(404 if url.endswith("/health") else 200)

    monkeypatch.setattr(requests, "get", fake_get)
    assert spec_decode_tests._wait_for_url_healthy("https://gw.example:443", timeout=5, interval=0)
    assert seen == ["https://gw.example:443/health", "https://gw.example:443/v1/models"]
    assert any("endpoint healthy" in r.getMessage() for r in caplog.records)


def test_health_wait_logs_last_results_on_failure(monkeypatch, caplog):
    import requests

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: _Resp(503))
    assert not spec_decode_tests._wait_for_url_healthy(
        "http://127.0.0.1:8080", timeout=0.3, interval=0.05, log_every=0.1
    )
    messages = [r.getMessage() for r in caplog.records]
    assert any("still waiting for endpoint" in m and "HTTP 503" in m for m in messages)
    assert any("endpoint not healthy after" in m and "HTTP 503" in m for m in messages)
