# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The MiniMax-H3 stress run: the vendored kit driven against the local deployment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module.benchmark_tests import minimax_h3_stress as stress
from test_module.benchmark_tests import video_benchmark_tests as bench

BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "model_name, server",
    [
        ("MiniMax-H3-FL2VA-LightX2V", "fl2va"),
        ("MiniMax-H3-Ref2VA-LightX2V", "ref2va"),
    ],
)
def test_command_drives_only_the_local_deployment(tmp_path, model_name, server):
    argv, env = stress.build_stress_command(model_name, BASE_URL, tmp_path, minutes=30)

    assert Path(argv[1]) == stress.KIT_DIR / "h3_stress.py"
    assert argv[argv.index("--servers") + 1] == server
    assert argv[argv.index("--minutes") + 1] == "30"
    assert argv[argv.index("--out") + 1] == str(tmp_path)
    # Both URL variables point at the local server, so the kit's remote defaults never apply.
    assert env["H3_FL2VA_URL"] == BASE_URL
    assert env["H3_REF2VA_URL"] == BASE_URL
    assert env["H3_API_KEY"] == "your-secret-key"
    assert "tenstorrent.com" not in " ".join(argv + list(env.values()))


def test_vendored_kit_has_every_asset_its_cases_name():
    cases = json.loads((stress.KIT_DIR / "cases.json").read_text())["cases"]
    assert {c["server"] for c in cases} == {"fl2va", "ref2va"}
    for case in cases:
        for key in ("images", "videos", "audios"):
            for name in case.get(key) or []:
                assert (stress.KIT_DIR / "assets" / name).is_file(), (case["id"], name)
    assert (stress.KIT_DIR / "assets" / "prompts" / "bank.json").is_file()


def _rows(*rows):
    return [{"kind": "request", "server": "fl2va", **r} for r in rows]


def _run_meta(model_ready=True, http_code=200, restart=(0, 0), worker_alive=True):
    return {
        "servers_start": {"fl2va": {"liveness": {"restart_count": restart[0]}}},
        "servers_end": {
            "fl2va": {
                "liveness": {
                    "http_code": http_code,
                    "model_ready": model_ready,
                    "worker_alive": worker_alive,
                    "restart_count": restart[1],
                }
            }
        },
    }


GOOD = {"expect": "ok", "pass": True, "http_code": 202, "case_id": "FL-5s"}


def test_all_expected_ok_passing_on_a_live_server_passes():
    verdict = stress.judge(_rows(GOOD, GOOD), _run_meta(), "fl2va")
    assert verdict["passed"] is True
    assert verdict["reasons"] == []
    assert verdict["ok_sent"] == 2 and verdict["ok_passed"] == 2


def test_a_failed_expected_ok_request_fails():
    bad = {**GOOD, "pass": False, "error": "job failed"}
    verdict = stress.judge(_rows(GOOD, bad), _run_meta(), "fl2va")
    assert verdict["passed"] is False
    assert any("1/2" in r for r in verdict["reasons"])


def test_no_expected_ok_request_at_all_fails():
    verdict = stress.judge([], _run_meta(), "fl2va")
    assert verdict["passed"] is False


def test_any_5xx_fails():
    err = {"expect": "probe", "pass": None, "http_code": 503, "case_id": "P-x"}
    verdict = stress.judge(_rows(GOOD, err), _run_meta(), "fl2va")
    assert verdict["passed"] is False
    assert verdict["http_5xx"] == 1


@pytest.mark.parametrize(
    "meta",
    [
        _run_meta(model_ready=False),
        _run_meta(http_code=405),
        _run_meta(worker_alive=False),
        _run_meta(restart=(0, 1)),
    ],
)
def test_a_server_that_did_not_survive_fails(meta):
    verdict = stress.judge(_rows(GOOD), meta, "fl2va")
    assert verdict["passed"] is False


def test_an_accepted_reject_case_is_reported_but_does_not_fail():
    accepted = {
        "expect": "reject",
        "pass": False,
        "http_code": 202,
        "case_id": "N-tiny",
    }
    verdict = stress.judge(_rows(GOOD, accepted), _run_meta(), "fl2va")
    assert verdict["passed"] is True
    assert verdict["rejects_accepted"] == ["N-tiny"]


def test_video_benchmark_routes_h3_to_the_stress_run(monkeypatch):
    called = {}
    monkeypatch.setattr(bench, "require_health", lambda ctx: None)
    monkeypatch.setattr(
        bench, "run_minimax_h3_stress", lambda ctx: called.setdefault("ctx", ctx)
    )
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(
            model_name="MiniMax-H3-Ref2VA-LightX2V",
            hf_model_repo="MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V",
        ),
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )
    assert bench.run_video_benchmark(ctx) is ctx
    assert called["ctx"] is ctx
