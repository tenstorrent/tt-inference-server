# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Hardware-free checks of the vendored MiniMax-H3 benchmark engine: the shape
oracles and clip judge, the tt-h3 request shapes and capability caps, failure
classification and retry policy, budgets, resume bookkeeping and the host state
machine (strikes / wedge / synchronous-build stop)."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import pytest

from test_module._test_common.minimax_h3_bench import adapters as A
from test_module._test_common.minimax_h3_bench import host as H
from test_module._test_common.minimax_h3_bench import judge as J
from test_module._test_common.minimax_h3_bench import models as M
from test_module._test_common.minimax_h3_bench import runner as R

FFMPEG = M.ffmpeg_binary()
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or shutil.which("ffprobe") is None,
    reason="ffmpeg + ffprobe required",
)


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """A throwaway asset pack + results dir with a pinned manifest."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "prompt_min.txt").write_text(
        "An old man walks along an Amsterdam canal at dusk.\n"
    )
    (assets / "prompt_std.txt").write_text("std prompt\n")
    (assets / "prompt_max_7000chars.txt").write_text("x" * 7000)
    (assets / "img_min_256px_scientist.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 200)
    (assets / "img_std_old_man_portrait.jpg").write_bytes(b"\xff\xd8\xff" + b"1" * 300)
    (assets / "img_max_27mb_astronaut.jpg").write_bytes(
        b"\xff\xd8\xff" + b"2" * 30
    )  # tiny stand-in
    (assets / "vid_min_2s_city_skyline.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypisom" + b"3" * 100
    )
    (assets / "aud_min_2s_score.wav").write_bytes(b"RIFF" + b"4" * 100)
    manifest = (
        "\n".join(f"{M.sha256(str(p))}  {p.name}" for p in sorted(assets.iterdir()))
        + "\n"
    )
    (assets / "sha256s-bundle.txt").write_text(manifest)
    M.configure(assets_dir=str(assets), out_dir=str(tmp_path / "out"))
    M._prompt_cache.clear()
    monkeypatch.setenv("H3_MAX_RETRIES", "0")
    yield assets
    M.configure(
        assets_dir=M.REPO_ASSETS,
        out_dir=os.path.join(tempfile.gettempdir(), "minimax_h3_bench"),
    )
    M._prompt_cache.clear()


# ---------------------------------------------------------------- oracles + judge
@pytest.mark.parametrize(
    "seconds,frames", [(4, 107), (5, 124), (8, 192), (10, 243), (15, 362)]
)
def test_frame_rule_snaps_up_to_17n_plus_5(seconds, frames):
    assert J.expected_frames(seconds) == frames
    assert (frames - 5) % 17 == 0
    assert abs(J.expected_seconds(seconds) - frames / 24) < 1e-9


def _make_clip(
    path,
    seconds=5.0,
    size="1344x768",
    fps=24,
    audio="sine=frequency=440:sample_rate=48000",
    frames=None,
):
    n = frames or J.expected_frames(seconds)
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={fps}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", audio]
    cmd += [
        "-frames:v",
        str(n),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
    ]
    if audio:
        cmd += ["-t", f"{n / fps:.4f}", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return str(path)


@needs_ffmpeg
def test_judge_accepts_a_contract_clip(tmp_path):
    clip = _make_clip(tmp_path / "ok.mp4")
    problems, notes = J.judge(clip, 5)
    assert problems == [], (problems, notes)


@needs_ffmpeg
@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({"audio": None}, "no audio stream"),
        ({"size": "1280x720"}, "canvas 1280x720"),
        ({"frames": 100}, "duration"),
        ({"audio": "anullsrc=sample_rate=48000"}, "silent"),
        (
            {
                "audio": "aevalsrc=exprs='if(lt(mod(t\\,0.01)\\,0.005)\\,1\\,-1)':sample_rate=48000"
            },
            "rails",
        ),
    ],
)
def test_judge_rejects_broken_clips(tmp_path, kwargs, expect):
    clip = _make_clip(tmp_path / "bad.mp4", **kwargs)
    problems, _notes = J.judge(clip, 5)
    assert any(expect in p for p in problems), problems


@needs_ffmpeg
@pytest.mark.parametrize("clipped_fraction,railed", [(0.01, False), (0.2, True)])
def test_rail_share_is_a_fraction_of_all_samples(tmp_path, clipped_fraction, railed):
    # a 440 Hz tone at -10 dB with a full-scale burst for `clipped_fraction` of every second
    audio = f"aevalsrc=exprs='if(lt(mod(t\\,1)\\,{clipped_fraction})\\,1\\,0.3*sin(2*PI*440*t))':s=48000"
    clip = _make_clip(tmp_path / "burst.mp4", audio=audio)
    _mean, _peak, share = J.audio_stats(clip)
    assert (
        share is not None and abs(share - clipped_fraction) < clipped_fraction * 0.5
    ), share
    problems, _notes = J.judge(clip, 5)
    assert any("rails" in p for p in problems) is railed, problems


@needs_ffmpeg
def test_judge_notes_the_one_frame_gap(tmp_path):
    clip = _make_clip(tmp_path / "short.mp4", frames=J.expected_frames(5) - 1)
    problems, notes = J.judge(clip, 5)
    assert problems == [], problems
    assert any("short by one frame" in n for n in notes), notes


def test_judge_refuses_a_non_mp4(tmp_path):
    bogus = tmp_path / "x.mp4"
    bogus.write_bytes(b"not a video" * 1000)
    assert J.judge(str(bogus), 5)[0] == ["not an mp4 container"]


def test_run_problems_from_the_row_when_the_clip_is_gone():
    rec = {"out_file": "/nonexistent/clip.mp4", "mvhd_duration_s": 5.167, "has_audio": True,
           "api_exposes_progress": True, "gen_s": 70.0}  # fmt: skip
    problems, notes = J.run_problems(rec, {"duration_s": 5})
    assert problems == [] and notes
    rec.update(mvhd_duration_s=4.0, has_audio=False, gen_s=0)
    problems, _ = J.run_problems(rec, {"duration_s": 5})
    assert len(problems) == 3


# ---------------------------------------------------------------- adapter
def test_t2va_payload_never_names_a_step_count(pack):
    adapter = A.TenstorrentH3({"t2va": "http://127.0.0.1:1"}, api_key="k")
    case = {"id": "T2VA-L", "task": "t2va", "duration_s": 5, "steps": 50, "seed": 42, "aspect_ratio": "16:9",
            "prompt": "prompt_min.txt"}  # fmt: skip
    path, payload, grace = adapter.build(case)
    assert path == "/v1/videos/generations" and grace == 60
    assert payload == {"prompt": "An old man walks along an Amsterdam canal at dusk.", "duration_seconds": 5,
                       "seed": 42, "aspect_ratio": "16:9"}  # fmt: skip
    assert adapter.auth()["Authorization"] == "Bearer k"


def test_fl2va_and_ref2va_payloads_inline_base64(pack):
    adapter = A.TenstorrentH3({"all": "http://127.0.0.1:1"}, api_key="k")
    adapter._caps_cache["http://127.0.0.1:1"] = {
        "image": 10_000_000,
        "media": 80_000_000,
    }
    fl = {"id": "FL2VA-H1", "task": "fl2va", "duration_s": 15, "steps": 50, "seed": 42, "prompt": "prompt_std.txt",
          "images": ["img_min_256px_scientist.jpg", "img_std_old_man_portrait.jpg"]}  # fmt: skip
    path, payload, grace = adapter.build(fl)
    assert path == "/v1/videos/generations/i2v" and grace == 180
    assert [p["frame_pos"] for p in payload["image_prompts"]] == [0, -1]
    assert base64.b64decode(payload["image_prompts"][0]["image"]).startswith(
        b"\xff\xd8\xff"
    )
    ref = {"id": "REF2VA-L", "task": "ref2va", "duration_s": 5, "steps": 50, "seed": 42, "prompt": "prompt_min.txt",
           "images": ["img_min_256px_scientist.jpg"], "videos": ["vid_min_2s_city_skyline.mp4"],
           "audios": ["aud_min_2s_score.wav"]}  # fmt: skip
    path, payload, grace = adapter.build(ref)
    assert path == "/v1/videos/generations/ref2va" and grace == 600
    assert set(payload["references"]) == {"images", "videos", "audios"}


def test_oversize_keyframe_is_a_local_capability_refusal(pack):
    adapter = A.TenstorrentH3({"all": "http://127.0.0.1:1"}, api_key="k")
    adapter._caps_cache["http://127.0.0.1:1"] = {"image": 10, "media": 80_000_000}
    fl = {"id": "FL2VA-H", "task": "fl2va", "duration_s": 15, "steps": 50, "seed": 42, "prompt": "prompt_max_7000chars.txt",
          "images": ["img_max_27mb_astronaut.jpg", "img_min_256px_scientist.jpg"]}  # fmt: skip
    path, (code, body), grace = adapter.build(fl)
    assert path is None and code == 400 and grace == 0
    msg = body["error"]["message"]
    assert msg.startswith(M.CLIENT_REJECT) and "over this endpoint's 10-char cap" in msg
    rec = {"outcome": "submit_failed", "submit_http": 400, "error_message": msg}
    assert M.classify_failure(rec) == "client_capability"
    assert M.is_transient(rec) is False


def test_caps_are_read_from_openapi(pack, monkeypatch):
    spec = {"components": {"schemas": {
        "ImagePromptEntry": {"properties": {"image": {"maxLength": 42_000_000}}},
        "MediaSource": {"properties": {"b64": {"anyOf": [{"type": "string", "maxLength": 90_000_000}, {"type": "null"}]}}},
    }}}  # fmt: skip
    monkeypatch.setattr(A, "http", lambda method, url, **kw: (200, spec))
    adapter = A.TenstorrentH3({"t2va": "http://127.0.0.1:1"}, api_key="k")
    assert adapter.caps("t2va") == {"image": 42_000_000, "media": 90_000_000}


def test_synchronous_build_is_refused_not_retried(pack, monkeypatch):
    monkeypatch.setattr(
        A,
        "http",
        lambda method, url, **kw: (200, b"\x00\x00\x00\x18ftypisom" + b"0" * 5000),
    )
    adapter = A.TenstorrentH3({"t2va": "http://127.0.0.1:1"}, api_key="k")
    case = {
        "id": "SMOKE",
        "task": "t2va",
        "duration_s": 5,
        "steps": 8,
        "seed": 42,
        "prompt": "prompt_min.txt",
    }
    (code, body), task = adapter.post(case)
    assert code == 400 and "synchronous deployment" in body["error"]["message"]


def test_api_key_resolution_order(monkeypatch):
    for name in ("API_KEY", "MINIMAX_API_KEY", "TT_MINIMAX_API_KEY", "H3_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert A.resolve_api_key() == A.DEFAULT_API_KEY
    monkeypatch.setenv("TT_MINIMAX_API_KEY", "tt")
    assert A.resolve_api_key() == "tt"
    monkeypatch.setenv("API_KEY", "ci")
    assert A.resolve_api_key() == "ci"


# ---------------------------------------------------------------- classification / retry / budgets
@pytest.mark.parametrize(
    "rec,klass,transient",
    [
        ({"outcome": "timeout"}, "timeout", False),
        (
            {
                "outcome": "submit_failed",
                "submit_http": 0,
                "error_message": "URLError: refused",
            },
            "transport",
            True,
        ),
        (
            {
                "outcome": "submit_failed",
                "submit_http": 0,
                "error_message": "timed out",
            },
            "transport",
            False,
        ),
        ({"outcome": "unreachable"}, "transport", True),
        ({"outcome": "content_failed", "content_http": 0}, "transport", True),
        ({"outcome": "content_failed", "content_http": 404}, "content_missing", False),
        (
            {
                "outcome": "failed",
                "error_message": "duration_seconds must be an integer from 4 to 15",
            },
            "engine_validation",
            False,
        ),
        (
            {
                "outcome": "failed",
                "error_message": "unknown field(s) for MiniMax-H3 t2va: resolution",
            },
            "engine_validation",
            False,
        ),
        (
            {"outcome": "failed", "error_message": "Internal Server Error"},
            "engine_internal",
            True,
        ),
        (
            {
                "outcome": "failed",
                "error_message": "TT_THROW device timeout in fetch queue wait",
            },
            "engine_internal",
            False,
        ),
        (
            {
                "outcome": "failed",
                "job_id": "j",
                "last_status": "in_progress",
                "error_message": "bad gateway",
            },
            "engine_internal",
            False,
        ),
        ({"outcome": "ok"}, None, False),
    ],
)
def test_failure_classes_and_retry_policy(rec, klass, transient):
    assert M.classify_failure(rec) == klass
    assert M.is_transient(rec) is transient


def test_budgets_come_from_the_table_then_fallbacks(monkeypatch, tmp_path):
    M.configure(out_dir=str(tmp_path))
    monkeypatch.delenv("H3_TIMEOUT", raising=False)
    cases = {c["id"]: c for c in M.load_cases()["cases"]}
    assert M.case_timeout_s(cases["T2VA-L"], "BH1X") == 600
    assert M.case_timeout_s(cases["REF2VA-H"], "BH1X") == 4200
    assert M.case_timeout_s(cases["T2VA-L"], "T1") == 900
    assert (
        M.case_timeout_s(cases["REF2VA-H"], "T1")
        == M.TIMEOUT_TABLE_S["TT-SJC3"]["REF2VA-H"]
    )
    synthetic = {"id": "NEW", "duration_s": 5, "steps": 50}
    assert M.case_timeout_s(synthetic, "BH1X") == M.TIMEOUT_FLOOR_S
    monkeypatch.setenv("H3_TIMEOUT", "77")
    assert M.case_timeout_s(cases["T2VA-L"], "BH1X") == 77


def test_default_retry_budget_is_the_original_two(monkeypatch):
    assert (
        M.MAX_RETRIES == 2
    )  # H3_MAX_RETRIES overrides; is_transient() fences what is retried


def test_every_case_has_a_bh1x_budget():
    ids = {c["id"] for c in M.load_cases()["cases"]}
    missing = ids - set(M.TIMEOUT_TABLE_S["BH1X"])
    assert not missing, missing


def test_cases_json_matches_the_h3_benchmark_contract():
    cfg = M.load_cases()
    ids = [c["id"] for c in cfg["cases"]]
    assert len(ids) == 21 and len(set(ids)) == 21
    assert {c["task"] for c in cfg["cases"]} == {"t2va", "fl2va", "ref2va"}
    assert {c["duration_s"] for c in cfg["cases"]} == {5, 10, 15}
    assert all(
        c["steps"] == 50 and c["seed"] == 42 and c["aspect_ratio"] == "16:9"
        for c in cfg["cases"]
    )
    assert cfg["smoke"]["id"] == "SMOKE" and cfg["smoke"]["task"] == "t2va"


# ---------------------------------------------------------------- assets
def test_manifest_verification(pack):
    assert M.verify_assets(["prompt_min.txt", "img_min_256px_scientist.jpg"]) == []
    (pack / "prompt_std.txt").write_text("tampered")
    problems = M.verify_assets(["prompt_std.txt", "nope.jpg"])
    assert any("sha256 differs" in p for p in problems) and any(
        "missing asset" in p for p in problems
    )


def test_repo_pack_pins_the_prompts_and_manifests():
    M.configure(assets_dir=M.REPO_ASSETS)
    try:
        assert (
            M.verify_assets(
                ["prompt_min.txt", "prompt_std.txt", "prompt_max_7000chars.txt"]
            )
            == []
        )
        assert len(M.read_prompt("prompt_max_7000chars.txt")) == 7000
    finally:
        M._prompt_cache.clear()


# ---------------------------------------------------------------- resume + verdicts
def test_resume_bookkeeping_never_reuses_a_tag(pack):
    M.ensure_dirs()
    rows = [{"combo": "C", "case": "T2VA-L", "tag": "warmup", "measured": False, "outcome": "ok"},
            {"combo": "C", "case": "T2VA-L", "tag": "r1", "measured": True, "outcome": "failed"},
            {"combo": "C", "case": "T2VA-L", "tag": "r2", "measured": True, "outcome": "ok"}]  # fmt: skip
    with open(M.results_path(), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    assert R.existing_ok_counts("C") == {"T2VA-L": 1}
    assert R.next_free_tags(R.existing_tags("C")["T2VA-L"], 2) == ["r3", "r4"]
    assert [r["tag"] for r in R.newest_ok_rows("C", "T2VA-L", 3)] == ["r2"]


def test_case_verdict_rc(pack):
    M.ensure_dirs()
    case = {"id": "T2VA-L"}
    ok = {"outcome": "ok"}
    bad = {
        "outcome": "failed",
        "failure_class": "engine_internal",
        "error_message": "boom",
    }
    assert R.case_verdict("C", case, 3, 0, ok, [ok, ok, ok]) == 0
    assert R.case_verdict("C", case, 3, 1, ok, [ok, bad]) == 1
    assert R.case_verdict("C", case, 3, 0, bad, [bad]) == 2


def test_primary_metric_prefers_engine_time_then_gen():
    assert R.primary_metric({"engine_inference_s": 60, "gen_s": 70}) == 60.0
    assert R.primary_metric({"gen_s": 70, "e2e_s": 75}) == 70.0
    assert R.primary_metric({"e2e_s": 75}) == 75


# ---------------------------------------------------------------- host state machine
class _FakeAdapter:
    def __init__(self, health=200):
        self.health = health
        self.cancelled = []
        self.on_submit = None

    def get(self, task, path, key=True, timeout=30):
        if path == "/health":
            return self.health, {}
        return 404, {"detail": "Video job not found"}

    def cancel(self, task, job_id):
        self.cancelled.append(job_id)
        return 200

    def base_url(self, task):
        return "http://127.0.0.1:1"


def _ep():
    return H.Endpoint(task="t2va", url="http://127.0.0.1:1", name="box", reachable=True)


def test_after_run_ok_clears_strikes_and_capability_refusal_is_neutral():
    ad = _FakeAdapter()
    tracker = H.JobTracker(ad)
    ep = _ep()
    ep.strike(2)
    assert (
        H.after_run(
            ad,
            tracker,
            ep,
            {"case": "T2VA-L", "tag": "r1", "outcome": "ok", "job_id": "j1"},
        )
        == ""
    )
    assert ep.strikes == 0
    rec = {"case": "FL2VA-H", "tag": "r1", "outcome": "submit_failed", "submit_http": 400,
           "failure_class": "client_capability", "error_message": M.CLIENT_REJECT + "x: over cap"}  # fmt: skip
    assert (
        H.after_run(ad, tracker, ep, rec) == "" and ep.strikes == 0 and not ep.stopped
    )


def test_after_run_synchronous_build_stops_the_host():
    ad = _FakeAdapter()
    tracker = H.JobTracker(ad)
    ep = _ep()
    rec = {"case": "SMOKE", "tag": "smoke", "outcome": "submit_failed", "submit_http": 400,
           "failure_class": "client_capability", "error_message": M.CLIENT_REJECT + "synchronous deployment: POST answered 200"}  # fmt: skip
    assert H.after_run(ad, tracker, ep, rec).startswith("contract mismatch")
    assert ep.stopped


def test_after_run_three_device_faults_stop_the_host():
    ad = _FakeAdapter()
    tracker = H.JobTracker(ad)
    ep = _ep()
    rec = {"case": "SIZE-V", "tag": "r1", "outcome": "failed", "job_id": "j", "last_status": "failed",
           "failure_class": "engine_internal",
           "error_message": "Worker 0 execution error: TT_THROW device timeout in fetch queue wait, potential hang detected"}  # fmt: skip
    assert H.after_run(ad, tracker, ep, rec) == "" and ep.strikes == 1
    assert H.after_run(ad, tracker, ep, dict(rec, tag="r2")) == "" and ep.strikes == 2
    stop = H.after_run(ad, tracker, ep, dict(rec, tag="r3"))
    assert "3 device-trouble failures" in stop and ep.stopped == stop


def test_after_run_timeout_in_progress_wedges_unless_keep_going():
    ad = _FakeAdapter()
    ep = _ep()
    tracker = H.JobTracker(ad)
    rec = {"case": "T2VA-H", "tag": "r1", "outcome": "timeout", "job_id": "j9", "last_status": "in_progress",
           "timeout_s": 1200, "failure_class": "timeout"}  # fmt: skip
    assert H.after_run(ad, tracker, ep, rec, keep_going=True) == "" and not ep.stopped
    assert ad.cancelled == ["j9"]  # the live job is cancelled either way
    assert H.after_run(ad, tracker, ep, dict(rec, job_id="j10")).startswith(
        "endpoint wedged"
    )
    assert ep.stopped.startswith("wedged")


def test_after_run_still_queued_timeout_is_a_strike_not_a_wedge():
    ad = _FakeAdapter()
    ep = _ep()
    tracker = H.JobTracker(ad)
    rec = {
        "case": "T2VA-L",
        "tag": "r1",
        "outcome": "timeout",
        "job_id": "j",
        "last_status": "queued",
        "timeout_s": 600,
    }
    assert (
        H.after_run(ad, tracker, ep, rec) == "" and ep.strikes == 1 and not ep.stopped
    )


def test_after_run_transport_failure_with_dead_health_stops_the_host():
    ad = _FakeAdapter(health=0)
    ep = _ep()
    tracker = H.JobTracker(ad)
    rec = {"case": "T2VA-L", "tag": "r1", "outcome": "submit_failed", "submit_http": 0, "error_message": "URLError: refused",
           "failure_class": "transport"}  # fmt: skip
    assert H.after_run(ad, tracker, ep, rec).startswith("host unreachable")


def test_job_tracker_cancels_what_is_left_live():
    ad = _FakeAdapter()
    tracker = H.JobTracker(ad)
    tracker.record("t2va", "live-1")
    tracker.mark({"job_id": "done-1", "outcome": "ok"})
    assert tracker.cancel_leftovers() == []
    assert ad.cancelled == ["live-1"]


def test_smoke_cases_mirror_the_minimum_bar():
    cfg = M.load_cases()
    assert H.smoke_case("t2va", cfg)["id"] == "SMOKE"
    fl = H.smoke_case("fl2va", cfg)
    assert fl["images"] == ["img_min_256px_scientist.jpg"] and fl["duration_s"] == 5
    ref = H.smoke_case("ref2va", cfg)
    assert ref["videos"] == ["vid_min_2s_city_skyline.mp4"] and ref["audios"] == [
        "aud_min_2s_score.wav"
    ]


def test_probe_checks_read_the_endpoint():
    ep = _ep()
    ep.health = 200
    ep.liveness_status = 200
    ep.liveness = {"model_ready": False}
    ep.routes = {"/v1/videos/generations", "/v1/videos/generations/{job_id}"}
    ep.gates = {"t2va": "refused"}
    ep.gate_detail = {"t2va": "422 this deployment serves fl2va"}
    problems = H.probe_checks(ep)
    assert any("model_ready" in p for p in problems)
    assert any("/download" in p for p in problems)
    assert any("refuses t2va" in p for p in problems)


def test_ci_mode_picks_the_bounded_plan():
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    assert T._ci_mode(None) is True
    assert (
        T._ci_mode(
            SimpleNamespace(
                runtime_config=SimpleNamespace(ci_mode=True, limit_samples_mode=None)
            )
        )
        is True
    )
    assert (
        T._ci_mode(
            SimpleNamespace(
                runtime_config=SimpleNamespace(
                    ci_mode=False, limit_samples_mode="ci-nightly"
                )
            )
        )
        is True
    )
    assert (
        T._ci_mode(
            SimpleNamespace(
                runtime_config=SimpleNamespace(ci_mode=False, limit_samples_mode=None)
            )
        )
        is False
    )
