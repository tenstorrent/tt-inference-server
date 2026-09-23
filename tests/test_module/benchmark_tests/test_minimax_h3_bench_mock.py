# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""End-to-end runs of MiniMaxH3BenchmarkTest against the mock H3 server (no
hardware; real mp4s from ffmpeg). Covers the happy path with every clip judged,
the capability xfail, and the fault paths the host state machine must catch."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from test_module._test_common import TestConfig
from test_module._test_common.minimax_h3_bench import models as M
from test_module.load_param_tests import minimax_h3_benchmark_test as T

MOCK = Path(__file__).resolve().parents[1] / "fixtures" / "mock_tt_h3_server.py"


@pytest.fixture(autouse=True)
def _ffmpeg_gate():
    """Skip without ffmpeg/ffprobe on a laptop; FAIL on a CI runner, where a skip would
    silently drop the whole end-to-end suite (test-gate.yml installs ffmpeg)."""
    if M.ffmpeg_binary() is not None and shutil.which("ffprobe") is not None:
        return
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(
            "ffmpeg/ffprobe missing on the CI runner: the mock encodes real clips and the "
            "judge probes them; see the ffmpeg step in .github/workflows/test-gate.yml",
            pytrace=False,
        )
    pytest.skip(
        "the mock encodes real clips with ffmpeg and the judge probes them with ffprobe"
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    # 1 s polls against 4 s generations: gen_s comes from several in_progress sightings and
    # cannot round to 0.0 because the second poll happened to land after completion.
    monkeypatch.setattr(M, "POLL_S", 1.0)


@pytest.fixture
def mock():
    processes = []

    def start(*extra, key="mock-key"):
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, str(MOCK), "--port", str(port), "--key", key, "--queue-seconds", "0.2",
             "--gen-seconds", "4", "--flat", *extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )  # fmt: skip
        processes.append(proc)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url + "/health", timeout=1):
                    return url
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("mock did not start")

    yield start
    for proc in processes:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def pack(tmp_path):
    """The repo prompts plus tiny stand-ins for the media the fl2va/ref2va cases name."""
    assets = tmp_path / "assets"
    shutil.copytree(M.REPO_ASSETS, assets)
    for name in ("img_min_256px_scientist.jpg", "img_std_old_man_portrait.jpg"):
        (assets / name).write_bytes(b"\xff\xd8\xff" + b"0" * 500)
    (assets / "img_max_27mb_astronaut.jpg").write_bytes(
        b"\xff\xd8\xff" + b"1" * 20_000_000
    )  # over the 10 M-char cap
    (assets / "vid_min_2s_city_skyline.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypisom" + b"2" * 2000
    )
    (assets / "aud_min_2s_score.wav").write_bytes(b"RIFF" + b"3" * 2000)
    return assets


def _run(url, out, **kw):
    kw.setdefault("verify_manifest", False)
    kw.setdefault("idle_wait_s", 5)
    kw.setdefault("api_key", "mock-key")
    return T.run_benchmark(base_url=url, out_dir=str(out), **kw)


def test_t2va_plan_passes_and_every_clip_is_judged(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="t2va",
                  plan=[{"cases": ["T2VA-L"], "runs": 2}, {"cases": ["T2VA-M", "T2VA-H"], "runs": 1}],
                  target_times_s={"T2VA-L": 90})  # fmt: skip
    assert result["pregate"] == [] and result["probe"] == []
    assert result["smoke"]["outcome"] == "ok" and result["smoke"]["problems"] == []
    assert [c["status"] for c in result["cases"]] == ["pass", "pass", "pass"]
    assert (
        result["cases"][0]["runs_ok"] == 2 and result["cases"][0]["timing_ok"] is True
    )
    assert result["success"], result["summary"]
    assert result["mesh"] == [4, 8]
    rows = [
        json.loads(line)
        for line in Path(result["artifacts"]["results_jsonl"]).read_text().splitlines()
    ]
    # smoke + (1 warmup + 2) + (1 + 1) + (1 + 1) = 8 rows, all ok, all 50-step 16:9 clips with audio
    assert len(rows) == 8 and all(r["outcome"] == "ok" for r in rows)
    assert all(r["steps_effective"] == 50 and r["has_audio"] for r in rows)
    assert {r["case"] for r in rows} == {"SMOKE", "T2VA-L", "T2VA-M", "T2VA-H"}
    assert os.path.exists(result["artifacts"]["results_csv"])
    status_log = tmp_path / "out" / f"smoke_status_{result['combo']}_t2va.log"
    assert status_log.read_text().splitlines()[0] == "ok"
    assert result["leftover_jobs"] == []


def test_fl2va_keyframe_over_the_cap_is_an_xfail(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "fl2va")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="fl2va",
                  plan=[{"cases": ["FL2VA-L", "FL2VA-H"], "runs": 1}])  # fmt: skip
    by_case = {c["case"]: c for c in result["cases"]}
    assert by_case["FL2VA-L"]["status"] == "pass"
    assert by_case["FL2VA-H"]["status"] == "xfail"
    assert all(
        f["failure_class"] == "client_capability"
        for f in by_case["FL2VA-H"]["failures"]
    )
    assert result["success"]  # a documented capability limit is not a failure


def test_refused_task_and_missing_asset_stop_before_any_generation(
    mock, pack, tmp_path, monkeypatch
):
    url = mock("--serve", "t2va")
    result = _run(
        url,
        tmp_path / "out",
        assets_dir=str(pack),
        task="fl2va",
        plan=[{"cases": ["FL2VA-L"], "runs": 1}],
    )
    assert any("refuses fl2va" in p for p in result["probe"])
    assert result["smoke"] is None and result["cases"] == []
    os.remove(pack / "prompt_min.txt")
    # files are resolved per name with the in-repo pack as the fallback; point that at an
    # empty directory so the deleted prompt is really missing
    monkeypatch.setattr(M, "REPO_ASSETS", str(tmp_path / "no-repo-pack"))
    result = _run(
        url,
        tmp_path / "out2",
        assets_dir=str(pack),
        task="t2va",
        plan=[{"cases": ["T2VA-L"], "runs": 1}],
    )
    assert (
        any("missing asset" in p for p in result["probe"]) and result["smoke"] is None
    )


def test_wrong_key_blocks_and_nothing_is_submitted(mock, pack, tmp_path):
    url = mock("--serve", "t2va")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="t2va", api_key="nope",
                  plan=[{"cases": ["T2VA-L"], "runs": 1}])  # fmt: skip
    assert any("rejects the configured key" in p for p in result["pregate"])
    assert result["smoke"] is None and not result["success"]


def test_broken_soundtrack_fails_the_smoke(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va", "--railed")
    result = _run(
        url,
        tmp_path / "out",
        assets_dir=str(pack),
        task="t2va",
        plan=[{"cases": ["T2VA-L"], "runs": 1}],
    )
    assert any("rails" in p for p in result["smoke"]["problems"]), result["smoke"]
    assert not result["success"]
    # the plan still ran and its railed clip failed the case too
    assert result["cases"] and result["cases"][0]["status"] == "fail", result["cases"]


def test_device_faults_strike_out_the_host(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va", "--fail", "t2va")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="t2va",
                  plan=[{"cases": ["T2VA-L", "T2VA-M"], "runs": 3}])  # fmt: skip
    assert result["smoke"]["outcome"] == "failed"
    assert result["stopped"] and "device-trouble" in result["stopped"]
    # the strikes accumulate inside the first case (warmup, r1); the second never starts
    assert [c["status"] for c in result["cases"]] == ["fail", "skipped"]
    assert not result["success"]


def test_hung_generation_wedges_the_host_and_cancels_the_job(
    mock, pack, tmp_path, monkeypatch
):
    monkeypatch.setenv("H3_TIMEOUT", "8")
    url = mock("--serve", "t2va", "--hang", "t2va:10")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="t2va",
                  plan=[{"cases": ["T2VA-M", "T2VA-L"], "runs": 1}])  # fmt: skip
    by_case = {c["case"]: c for c in result["cases"]}
    assert (
        by_case["T2VA-M"]["status"] == "fail" and "wedged" in by_case["T2VA-M"]["stop"]
    )
    assert by_case["T2VA-L"]["status"] == "skipped"
    assert result["leftover_jobs"] == []  # the hung job was cancelled at teardown
    with urllib.request.urlopen(
        urllib.request.Request(
            url + "/v1/videos/jobs", headers={"Authorization": "Bearer mock-key"}
        )
    ) as r:
        jobs = json.load(r)
    assert all(j["status"] in ("cancelled", "cancelling", "completed") for j in jobs), (
        jobs
    )


def test_synchronous_build_is_reported_as_a_contract_mismatch(
    mock, pack, tmp_path, monkeypatch
):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va", "--sync")
    result = _run(
        url,
        tmp_path / "out",
        assets_dir=str(pack),
        task="t2va",
        plan=[{"cases": ["T2VA-L"], "runs": 1}],
    )
    assert "synchronous deployment" in result["smoke"]["stop"]
    assert result["cases"][0]["status"] == "skipped" and not result["success"]


def test_resume_skips_measured_rows_already_present(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va")
    out = tmp_path / "out"
    first = _run(
        url,
        out,
        assets_dir=str(pack),
        task="t2va",
        plan=[{"cases": ["T2VA-L"], "runs": 1}],
        combo="C",
    )
    second = _run(
        url,
        out,
        assets_dir=str(pack),
        task="t2va",
        plan=[{"cases": ["T2VA-L"], "runs": 1}],
        combo="C",
    )
    assert first["success"] and second["success"]
    assert second["cases"][0]["resumed"] is True
    rows = [
        json.loads(line)
        for line in Path(second["artifacts"]["results_jsonl"]).read_text().splitlines()
    ]
    assert [r["tag"] for r in rows] == [
        "smoke",
        "warmup",
        "r1",
        "smoke",
    ]  # only the smoke re-ran


def _get(url, path, key="mock-key"):
    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def test_job_ids_and_legacy_links_never_echo_request_text(mock):
    import uuid

    url = mock()
    # a job id that is not a UUID is simply "no such job"
    code, _, _ = _get(url, "/v1/videos/generations/not-a-job")
    assert code == 404
    # the legacy route's successor link carries the route constant or the re-serialised UUID
    _, headers, _ = _get(url, "/video/jobs")
    assert headers.get("Link") == '</v1/videos/jobs>; rel="successor-version"'
    jid = str(uuid.uuid4())
    code, headers, _ = _get(url, f"/video/generations/{jid.upper()}")
    assert code == 404
    assert (
        headers.get("Link")
        == f'</v1/videos/generations/{jid}>; rel="successor-version"'
    )
    # request text that is neither a route nor a UUID gets no Link header at all
    code, headers, _ = _get(url, "/video/generations/%0d%0aX-Injected:%201")
    assert code == 404 and "Link" not in headers
    # a real job: the download's Content-Disposition names the server-issued id
    body = json.dumps(
        {"prompt": "a fox", "aspect_ratio": "16:9", "duration_seconds": 5}
    ).encode()
    req = urllib.request.Request(
        url + "/v1/videos/generations", data=body, method="POST",
        headers={"Authorization": "Bearer mock-key", "Content-Type": "application/json"},
    )  # fmt: skip
    with urllib.request.urlopen(req, timeout=10) as resp:
        job = json.loads(resp.read())
    deadline = time.time() + 60
    while time.time() < deadline:
        code, headers, _ = _get(url, f"/v1/videos/generations/{job['id']}/download")
        if code == 200:
            break
        time.sleep(0.5)
    assert code == 200
    assert (
        headers.get("Content-Disposition") == f'attachment; filename="{job["id"]}.mp4"'
    )


def test_enforce_timing_fails_a_slow_case(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="t2va",
                  plan=[{"cases": ["T2VA-L"], "runs": 1}],
                  target_times_s={"T2VA-L": 0.1}, enforce_timing=True)  # fmt: skip
    case = result["cases"][0]
    assert case["timing_ok"] is False and case["status"] == "fail"
    assert any("> 1.25x target" in p for p in case["problems"]), case
    assert not result["success"]


def test_deadline_skips_cases_it_could_not_start(mock, pack, tmp_path, monkeypatch):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    url = mock("--serve", "t2va")
    result = _run(url, tmp_path / "out", assets_dir=str(pack), task="t2va",
                  plan=[{"cases": ["T2VA-L", "T2VA-M"], "runs": 1}], deadline_s=0)  # fmt: skip
    assert result["smoke"]["outcome"] == "ok"  # the smoke ran; the plan did not start
    assert [c["status"] for c in result["cases"]] == ["skipped", "skipped"]
    assert all(c["stop"].startswith("deadline") for c in result["cases"])
    assert result["deadline_hit"] and not result["success"]


def test_spec_test_wrapper_never_resumes_and_plumbs_key_and_deadline(
    mock, pack, tmp_path, monkeypatch
):
    monkeypatch.setenv("H3_TIMEOUT", "60")
    for name in ("API_KEY", "MINIMAX_API_KEY", "TT_MINIMAX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TT_MINIMAX_API_KEY", "mock-key")  # via the shared resolver
    url = mock("--serve", "t2va")
    plan = [{"cases": ["T2VA-L"], "runs": 1}]
    targets = {"task": "t2va", "plan_ci": plan, "plan_full": plan, "out_dir": str(tmp_path / "out"),
               "assets_dir": str(pack), "verify_manifest": False, "idle_wait_s": 5, "combo": "C"}  # fmt: skip
    config = TestConfig(
        {
            "timeout": 900,
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )

    def run():
        test = T.MiniMaxH3BenchmarkTest(config, targets)
        test.base_url = url
        assert test._deadline_s() == 300.0  # 600 s before the template timeout
        return asyncio.run(test._run_specific_test_async())

    first, second = run(), run()
    assert first["success"] and second["success"], (first["summary"], second["summary"])
    # the second run generated again instead of reporting yesterday's rows
    assert second["cases"][0]["resumed"] is False and second["cases"][0]["runs_ok"] == 1
    rows = [
        json.loads(line)
        for line in Path(second["artifacts"]["results_jsonl"]).read_text().splitlines()
    ]
    assert [r["tag"] for r in rows] == [
        "smoke",
        "warmup",
        "r1",
        "smoke",
        "warmup2",
        "r2",
    ]
