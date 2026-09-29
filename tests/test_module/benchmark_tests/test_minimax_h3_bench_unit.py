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
import sys
import tempfile
from types import SimpleNamespace

import pytest

from test_module._test_common.minimax_h3_bench import adapters as A
from test_module._test_common.minimax_h3_bench import host as H
from test_module._test_common.minimax_h3_bench import judge as J
from test_module._test_common.minimax_h3_bench import models as M
from test_module._test_common.minimax_h3_bench import runner as R

FFMPEG = M.ffmpeg_binary()


@pytest.fixture
def _ffmpeg_gate():
    """Skip without ffmpeg/ffprobe on a laptop; FAIL on a CI runner, where a skip would
    silently drop every judge test on real clips (test-gate.yml installs ffmpeg)."""
    if FFMPEG is not None and shutil.which("ffprobe") is not None:
        return
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(
            "ffmpeg/ffprobe missing on the CI runner; see the ffmpeg step in "
            ".github/workflows/test-gate.yml",
            pytrace=False,
        )
    pytest.skip("ffmpeg + ffprobe required")


needs_ffmpeg = pytest.mark.usefixtures("_ffmpeg_gate")
MEDIA = ("*.jpg", "*.mp4", "*.wav")  # the part of the input pack that is not in git


@pytest.fixture(autouse=True)
def _no_staged_packs(monkeypatch, tmp_path_factory):
    """Hermetic asset lookup: a pack staged on this machine (H3_ASSETS, the CI volumes)
    must not stand in for a file a test expects to be missing or to come from its pack."""
    for name in ("H3_ASSETS", "PERSISTENT_VOLUME_ROOT"):
        monkeypatch.delenv(name, raising=False)
    absent = tmp_path_factory.mktemp("no-staged-pack")
    monkeypatch.setattr(M, "SHARED_ASSETS", str(absent / "shared"))
    monkeypatch.setattr(M, "LOCALDEV_ASSETS", str(absent / "localdev"))


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
    monkeypatch.setattr(
        M, "MAX_RETRIES", 0
    )  # read at import: the env var would not apply
    monkeypatch.setattr(M, "RETRY_BACKOFF_BASE_S", 0.0)
    # the throwaway pack pins against its own manifest; the default is the repo's
    monkeypatch.setattr(M, "PIN_MANIFESTS", (str(assets / "sha256s-bundle.txt"),))
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
    from test_module._test_common import minimax_h3_client as C

    # one list for every H3 test in the repo
    assert (
        A.API_KEY_ENV_VARS
        == C.API_KEY_ENV_VARS
        == ("API_KEY", "MINIMAX_API_KEY", "TT_MINIMAX_API_KEY")
    )
    for name in A.API_KEY_ENV_VARS + ("H3_API_KEY",):
        monkeypatch.delenv(name, raising=False)
    assert A.resolve_api_key() == A.DEFAULT_API_KEY == C.DEFAULT_API_KEY
    monkeypatch.setenv("H3_API_KEY", "not a name we read")
    assert A.resolve_api_key() == A.DEFAULT_API_KEY
    monkeypatch.setenv("TT_MINIMAX_API_KEY", "tt")
    assert A.resolve_api_key() == C.resolve_server_api_key() == "tt"
    monkeypatch.setenv("API_KEY", "ci")
    assert A.resolve_api_key() == C.resolve_server_api_key() == "ci"


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


# -- media probes never put a file name on a command line ----------------------------------


def test_media_arg_hands_the_file_over_as_a_descriptor(tmp_path):
    src = tmp_path / "clip.bin"
    src.write_bytes(b"ftyp-ish payload")
    with M.media_arg(str(src)) as (arg, fds):
        assert arg.startswith("/dev/fd/") and str(src) not in arg
        (fd,) = fds["pass_fds"]
        assert arg == f"/dev/fd/{fd}"
        with open(arg, "rb") as fh:  # what ffmpeg/ffprobe do with the argument
            assert fh.read() == b"ftyp-ish payload"
    with pytest.raises(OSError):
        os.fstat(fd)  # closed on exit
    with pytest.raises(OSError):
        with M.media_arg(str(tmp_path / "missing.mp4")):
            pass


def test_probes_on_a_missing_clip_answer_like_a_failed_probe(tmp_path):
    missing = str(tmp_path / "gone.mp4")
    assert M.has_audio(missing) in (
        False,
        None,
    )  # None only when no probe binary exists
    meta = J.ffprobe_json(missing)
    assert meta is None or "FileNotFoundError" in meta.get("probe_error", "")
    assert J.audio_stats(missing) == (None, None, None)


# -- review round 2: assets, metrics, probes, verdicts ---------------------------------------


def test_assets_resolve_per_file_and_pin_against_the_repo_manifest(
    tmp_path, monkeypatch
):
    # the git part of the in-repo pack only, so this holds whether or not media is
    # committed there; the pins stay the real repo manifests (PIN_MANIFESTS)
    repo = tmp_path / "repo"
    shutil.copytree(M.REPO_ASSETS, repo, ignore=shutil.ignore_patterns(*MEDIA))
    monkeypatch.setattr(M, "REPO_ASSETS", str(repo))
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "img_std_old_man_portrait.jpg").write_bytes(b"\xff\xd8\xff" + b"9" * 50)
    (shared / "prompt_min.txt").write_text("tampered\n")
    (shared / "sha256s-bundle.txt").write_text(  # vouches for both of its files
        "".join(
            f"{M.sha256(str(shared / n))}  {n}\n"
            for n in ("prompt_min.txt", "img_std_old_man_portrait.jpg")
        )
    )
    M.configure(assets_dir=str(shared), out_dir=str(tmp_path / "out"))
    try:
        assert M.asset_dirs() == [str(shared), str(repo)]
        assert M.asset_path("img_std_old_man_portrait.jpg") == str(
            shared / "img_std_old_man_portrait.jpg"
        )
        # a file the staged directory lacks comes from the repo pack, per file
        assert M.asset_path("prompt_std.txt") == str(repo / "prompt_std.txt")
        # the named directory's copy is the one used, pinned or not ...
        assert M.asset_path("prompt_min.txt") == str(shared / "prompt_min.txt")
        assert M.asset_path("nope.txt") is None
        problems = M.verify_assets(
            [
                "img_std_old_man_portrait.jpg",
                "prompt_min.txt",
                "prompt_std.txt",
                "nope.txt",
            ]
        )
        # ... and the shared directory's own manifest pins nothing: the repo manifest rules
        for name in ("img_std_old_man_portrait.jpg", "prompt_min.txt"):
            assert any(p.startswith(f"{name}: sha256 differs") for p in problems), (
                problems
            )
        assert not any("prompt_std.txt" in p for p in problems), problems
        assert any(p.startswith("missing asset nope.txt") for p in problems), problems
        # found rather than named (the shared volume, say): the pinned copy in git wins
        M.configure(assets_dir=str(shared), explicit=False)
        assert M.asset_path("prompt_min.txt") == str(repo / "prompt_min.txt")
    finally:
        M.configure(
            assets_dir=M.REPO_ASSETS,
            out_dir=os.path.join(tempfile.gettempdir(), "minimax_h3_bench"),
        )
        M._prompt_cache.clear()


def test_metric_takes_the_newest_matching_series():
    ts = "tt_media_server_video_last_generation_timestamp"
    text = (
        f'{ts}{{request_type="i2v",status="success"}} 100\n'
        f'{ts}{{request_type="t2v",status="success"}} 250\n'
        f'{ts}{{request_type="t2v",status="failure"}} 200\n'
        f'{ts}_created{{status="success"}} 999\n'
        'tt_canary_state{state="dead"} 0\n'
        'tt_canary_state{state="alive"} 1\n'
        "plain_gauge 7\n"
    )
    assert H._metric(text, ts, 'status="success"') == 250  # not the first line (100)
    assert H._metric(text, ts, 'status="failure"') == 200
    assert H._metric(text, "tt_canary_state", 'state="dead"') == 0
    assert H._metric(text, "tt_canary_state", 'state="alive"') == 1
    assert H._metric(text, "tt_canary_state", 'tate="dead"') is None  # whole pair only
    assert H._metric(text, "plain_gauge") == 7
    assert H._metric(text, "absent") is None


def test_probe_failure_fails_the_clip(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 300_000)
    monkeypatch.setattr(J, "ffprobe_json", lambda _p: {"probe_error": "boom"})
    problems, notes = J.judge(str(clip), 5)
    assert problems[0] == "probe failed: boom"
    assert any(p.startswith("duration") for p in problems)  # mvhd fallback still judged


@needs_ffmpeg
def test_a_corrupt_container_is_a_probe_failure_not_a_pass(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 300_000)
    meta = J.ffprobe_json(str(clip))
    assert meta is not None and (meta.get("probe_error") or not meta.get("streams"))
    problems, _ = J.judge(str(clip), 5)
    assert problems and all(
        "stream" in p or "probe" in p or "duration" in p for p in problems
    )
    assert M.has_audio(str(clip)) in (None, False)


def test_case_result_counts_only_this_runs_rows(pack):
    M.ensure_dirs()
    case = {"id": "T2VA-L", "task": "t2va", "duration_s": 5, "steps": 50}
    stale = {"combo": "C", "case": "T2VA-L", "tag": "r1", "measured": True, "outcome": "ok",
             "gen_s": 5.0, "e2e_s": 6.0, "out_file": "/nonexistent/r1.mp4",
             "mvhd_duration_s": 5.167, "has_audio": True}  # fmt: skip
    with open(M.results_path(), "w") as fh:
        fh.write(json.dumps(stale) + "\n")
    fresh = dict(
        stale, tag="r2", gen_s=20.0, e2e_s=21.0, out_file="/nonexistent/r2.mp4"
    )
    warm = dict(stale, tag="warmup", measured=False, gen_s=19.0)
    outcome = {
        "resumed": False,
        "have": 0,
        "rc": 0,
        "warm": warm,
        "done": [fresh],
        "stop": "",
    }
    entry = H.case_result("C", case, 1, outcome)
    assert entry["status"] == "pass" and entry["runs_ok"] == 1
    assert entry["median_s"] == 20.0  # this run's row, not the stale 5.0 on disk
    # a resumed case (CLI only) is judged from what is on disk
    resumed = {
        "resumed": True,
        "have": 1,
        "rc": 0,
        "warm": None,
        "done": [],
        "stop": "",
    }
    assert H.case_result("C", case, 1, resumed)["median_s"] == 5.0
    # a fresh run with no ok row has no timing and fails on its rc
    failed = {"resumed": False, "have": 0, "rc": 1, "warm": warm, "stop": "",
              "done": [dict(fresh, outcome="timeout", failure_class="timeout")]}  # fmt: skip
    entry = H.case_result("C", case, 1, failed)
    assert (
        entry["status"] == "fail"
        and entry["median_s"] is None
        and entry["runs_ok"] == 0
    )


def test_median_is_the_statistical_median_for_even_counts(pack):
    M.ensure_dirs()
    case = {"id": "T2VA-L", "task": "t2va", "duration_s": 5, "steps": 50}
    rows = [{"combo": "C", "case": "T2VA-L", "tag": t, "measured": True, "outcome": "ok",
             "gen_s": g, "out_file": "/nonexistent/x.mp4", "mvhd_duration_s": 5.167, "has_audio": True}
            for t, g in (("r1", 10.0), ("r2", 30.0))]  # fmt: skip
    outcome = {
        "resumed": False,
        "have": 0,
        "rc": 0,
        "warm": None,
        "done": rows,
        "stop": "",
    }
    assert H.case_result("C", case, 2, outcome)["median_s"] == 20.0


# -- per-task plumbing: the task from the spec, per-task plans, out dirs, smoke, assets ------


def _ctx(runner=None, ci=True, output_path="/nonexistent"):
    env = {"MODEL_RUNNER": runner} if runner else {}
    return SimpleNamespace(
        model_spec=SimpleNamespace(env_vars=env),
        runtime_config=SimpleNamespace(ci_mode=ci, limit_samples_mode=None),
        output_path=output_path,
        service_port=8000,
        base_url="http://127.0.0.1:1",
    )


def _bench(targets, ctx=None):
    from test_module._test_common import TestConfig
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    config = TestConfig({"timeout": 900, "retry_attempts": 0, "retry_delay": 0,
                         "break_on_failure": False})  # fmt: skip
    return T.MiniMaxH3BenchmarkTest(config, targets, ctx=ctx)


def _plan_ids(plan):
    return [cid for item in plan for cid in item["cases"]]


@pytest.mark.parametrize("task", A.TASKS)
def test_default_plans_cover_every_case_of_their_task(task):
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    ids = [c["id"] for c in M.load_cases()["cases"] if c["task"] == task]
    for plan in (T.DEFAULT_PLAN_CI[task], T.DEFAULT_PLAN_FULL[task]):
        assert sorted(_plan_ids(plan)) == sorted(ids)  # each case exactly once
    assert all(item["runs"] == 3 for item in T.DEFAULT_PLAN_FULL[task])
    assert _plan_ids(T.DEFAULT_PLAN_FULL[task]) == _plan_ids(T.DEFAULT_PLAN_CI[task])


def test_default_plans_per_task():
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    assert T.DEFAULT_PLAN_CI["t2va"] == [
        {"cases": ["T2VA-L"], "runs": 3},
        {"cases": ["T2VA-M", "T2VA-H"], "runs": 1},
    ]
    assert T.DEFAULT_PLAN_FULL["t2va"] == [
        {"cases": ["T2VA-L", "T2VA-M", "T2VA-H"], "runs": 3}
    ]
    assert T.DEFAULT_PLAN_CI["fl2va"] == [
        {"cases": ["FL2VA-L"], "runs": 3},
        {
            "cases": ["FL2VA-M", "FL2VA-H1", "FL2VA-L2", "FL2VA-M2", "FL2VA-H"],
            "runs": 1,
        },
    ]
    (ref,) = T.DEFAULT_PLAN_CI["ref2va"]
    assert ref["runs"] == 1
    by_id = {c["id"]: c for c in M.load_cases()["cases"]}
    order = ref["cases"]
    h_family = [cid for cid in order if cid.startswith("REF2VA-H")]
    assert (
        order[-len(h_family) :] == h_family == ["REF2VA-H5", "REF2VA-H10", "REF2VA-H"]
    )
    rest = order[: -len(h_family)]
    durations = [by_id[cid]["duration_s"] for cid in rest]
    assert durations == sorted(durations) and durations[:3] == [5, 5, 5]
    # within a duration the cheaper inputs go first (the BH1X budget is the risk proxy)
    for i in range(0, len(rest), 3):
        budgets = [M.TIMEOUT_TABLE_S["BH1X"][cid] for cid in rest[i : i + 3]]
        assert budgets == sorted(budgets), rest[i : i + 3]


def test_the_task_comes_from_the_model_runner():
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    for task in A.TASKS:
        assert T.deployment_task(_ctx(f"tt-minimax-h3-{task}")) == task
        assert T.deployment_task(_ctx(f"tt-minimax-h3-{task}"), task) == task
    # no spec (CLI, hardware-free tests): the target, else t2va as before
    assert T.deployment_task(None) == "t2va"
    assert T.deployment_task(None, "ref2va") == "ref2va"
    assert T.deployment_task(_ctx(None), "fl2va") == "fl2va"
    with pytest.raises(ValueError, match="wrong task"):
        T.deployment_task(_ctx("tt-minimax-h3-fl2va"), "t2va")
    with pytest.raises(ValueError, match="not a MiniMax-H3 runner"):
        T.deployment_task(_ctx("tt-wan2.2"))
    assert T.deployment_task(_ctx("tt-wan2.2"), "fl2va") == "fl2va"  # the target says
    with pytest.raises(ValueError, match="not one of"):
        T.deployment_task(None, "i2v")


def test_the_wrapper_runs_the_default_plan_of_the_derived_task():
    ctx = _ctx("tt-minimax-h3-ref2va")
    test = _bench({}, ctx)
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    assert test._task() == "ref2va"
    assert test._plan("ref2va") == T.DEFAULT_PLAN_CI["ref2va"]
    full = _bench({}, _ctx("tt-minimax-h3-fl2va", ci=False))
    assert full._plan(full._task()) == T.DEFAULT_PLAN_FULL["fl2va"]
    own = [{"cases": ["FL2VA-L"], "runs": 1}]
    assert _bench({"plan_ci": own}, _ctx("tt-minimax-h3-fl2va"))._plan("fl2va") == own


def test_a_contradicting_task_fails_before_anything_is_sent(monkeypatch):
    import asyncio

    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    def never(**_kw):
        raise AssertionError("run_benchmark must not start")

    monkeypatch.setattr(T, "run_benchmark", never)
    test = _bench({"task": "t2va"}, _ctx("tt-minimax-h3-fl2va"))
    with pytest.raises(
        ValueError, match="MODEL_RUNNER='tt-minimax-h3-fl2va' serves fl2va"
    ):
        asyncio.run(test._run_specific_test_async())


def test_each_entry_gets_its_own_output_directory(monkeypatch, tmp_path):
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    monkeypatch.setattr(T, "_OUT_DIR_OWNERS", {})
    base = tmp_path / "minimax_h3_bench"
    ctx = _ctx("tt-minimax-h3-ref2va", output_path=str(tmp_path))
    first = [{"cases": ["REF2VA-L", "REF2VA-M5"], "runs": 1}]
    second = [{"cases": ["REF2VA-H5"], "runs": 1}]
    # the single-entry layout: the first entry keeps <output>/minimax_h3_bench
    assert _bench({}, ctx)._out_dir(first) == str(base)
    assert _bench({}, ctx)._out_dir(first) == str(base)  # the same entry again (retry)
    # a later entry with other cases writes below it, named after its cases
    assert _bench({}, ctx)._out_dir(second) == str(base / "REF2VA-H5")
    assert _bench({"out_subdir": "h-family"}, ctx)._out_dir(second) == str(
        base / "h-family"
    )
    assert _bench({"out_dir": str(tmp_path / "x"), "out_subdir": "a/b"}, ctx)._out_dir(
        second
    ) == str(tmp_path / "x" / "a" / "b")
    for bad in ("../elsewhere", "/abs"):
        with pytest.raises(ValueError, match="relative path"):
            _bench({"out_subdir": bad}, ctx)._out_dir(second)
    # two entries may not share an explicit out_subdir; the same entry again may
    with pytest.raises(ValueError, match="already another entry's directory"):
        _bench({"out_subdir": "h-family", "skip_smoke": True}, ctx)._out_dir(second)
    assert _bench({"out_subdir": "h-family"}, ctx)._out_dir(second) == str(
        base / "h-family"
    )
    # a long case list gets a bounded, stable name
    slug = T._cases_slug(T.DEFAULT_PLAN_CI["ref2va"])
    assert len(slug) <= 64 and slug == T._cases_slug(T.DEFAULT_PLAN_CI["ref2va"])
    assert slug.startswith("REF2VA-L_to_REF2VA-H_12cases_")


def test_entries_that_run_the_same_plan_still_get_their_own_directory(
    monkeypatch, tmp_path
):
    # run-full-evals: entries that set only plan_ci all fall back to the same full plan
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    monkeypatch.setattr(T, "_OUT_DIR_OWNERS", {})
    base = tmp_path / "minimax_h3_bench"
    ctx = _ctx("tt-minimax-h3-ref2va", ci=False, output_path=str(tmp_path))
    entries = [
        _bench({"plan_ci": [{"cases": [cid], "runs": 1}]}, ctx)
        for cid in ("REF2VA-L", "REF2VA-H", "SIZE-V")
    ]
    plans = [e._plan(e._task()) for e in entries]
    assert plans == [T.DEFAULT_PLAN_FULL["ref2va"]] * 3
    slug = T._cases_slug(plans[0])
    dirs = [e._out_dir(plan) for e, plan in zip(entries, plans)]
    assert dirs == [str(base), str(base / slug), str(base / f"{slug}-2")]
    # each keeps its own directory when it runs again (a retry)
    assert [e._out_dir(plan) for e, plan in zip(entries, plans)] == dirs


def test_plan_estimate_is_the_budgets_over_their_safety_factor(pack, tmp_path):
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    bh1x = M.TIMEOUT_TABLE_S["BH1X"]
    plan = [{"cases": ["REF2VA-L"], "runs": 2}, {"cases": ["SIZE-V"], "runs": 1}]
    gens = bh1x["SMOKE-REF2VA"] + 3 * bh1x["REF2VA-L"] + 2 * bh1x["SIZE-V"]
    assert T.plan_estimate_s("ref2va", plan) == round(gens / T.BUDGET_FACTOR)
    assert T.plan_estimate_s("ref2va", plan, skip_smoke=True) == round(
        (gens - bh1x["SMOKE-REF2VA"]) / T.BUDGET_FACTOR
    )
    # the defaults: t2va and fl2va fit the template's 14400 s entry, ref2va does not
    deadline = 14400 - 600
    assert T.plan_estimate_s("t2va", T.DEFAULT_PLAN_CI["t2va"]) == 3200
    assert T.plan_estimate_s("t2va", T.DEFAULT_PLAN_FULL["t2va"]) == 5400
    assert T.plan_estimate_s("fl2va", T.DEFAULT_PLAN_CI["fl2va"]) == 4600
    assert T.plan_estimate_s("fl2va", T.DEFAULT_PLAN_FULL["fl2va"]) <= deadline
    assert T.plan_estimate_s("ref2va", T.DEFAULT_PLAN_CI["ref2va"]) > deadline
    # a run whose plan cannot fit its deadline says so up front
    result = T.run_benchmark(
        base_url="http://127.0.0.1:1", task="ref2va", out_dir=str(tmp_path / "o"),
        assets_dir=str(pack), verify_manifest=False, deadline_s=deadline,
    )  # fmt: skip
    assert result["estimate_s"] == T.plan_estimate_s(
        "ref2va", T.DEFAULT_PLAN_CI["ref2va"]
    )
    log = (tmp_path / "o" / "run.log").read_text()
    assert f"~{result['estimate_s']}s at the BH1X pace, over the {deadline}s" in log


def test_a_named_asset_directory_wins_over_a_pinned_copy(monkeypatch, tmp_path):
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    repo = tmp_path / "repo"
    shutil.copytree(M.REPO_ASSETS, repo, ignore=shutil.ignore_patterns(*MEDIA))
    monkeypatch.setattr(M, "REPO_ASSETS", str(repo))
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "prompt_min.txt").write_text("an edited prompt\n")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "prompt_std.txt").write_text("a stale staged prompt\n")
    monkeypatch.setenv("H3_ASSETS", str(staged))
    plan = [{"cases": ["T2VA-L", "T2VA-M"], "runs": 1}]
    kw = dict(base_url="http://127.0.0.1:1", task="t2va", plan=plan)
    try:
        # named: its unpinned prompt is the one used, and the probe says it is not pinned
        ran = T.run_benchmark(out_dir=str(tmp_path / "a"), assets_dir=str(mine), **kw)
        assert M.asset_path("prompt_min.txt") == str(mine / "prompt_min.txt")
        assert (
            "prompt_min.txt: sha256 differs from the manifest (not the pinned file)"
            in (ran["probe"])
        )
        # among found locations the pinned copy in git still beats the stale staged one
        assert M.asset_path("prompt_std.txt") == str(repo / "prompt_std.txt")
        # --no-manifest: no pin preference at all, the first copy found
        T.run_benchmark(out_dir=str(tmp_path / "b"), assets_dir=str(mine),
                        verify_manifest=False, **kw)  # fmt: skip
        assert M.asset_path("prompt_min.txt") == str(mine / "prompt_min.txt")
        assert M.asset_path("prompt_std.txt") == str(staged / "prompt_std.txt")
        # a named directory that does not exist is not named: the staged pack is found
        T.run_benchmark(out_dir=str(tmp_path / "c"), assets_dir=str(tmp_path / "gone"),
                        **kw)  # fmt: skip
        assert M.assets_dir() == str(staged) and not M._STATE["explicit_assets"]
        assert M.asset_path("prompt_std.txt") == str(repo / "prompt_std.txt")
    finally:
        M.configure(
            assets_dir=M.REPO_ASSETS,
            out_dir=os.path.join(tempfile.gettempdir(), "minimax_h3_bench"),
        )
        M._prompt_cache.clear()


@pytest.mark.parametrize(
    "argv,plan",
    [
        ([], [{"cases": ["T2VA-L"], "runs": 3}]),  # unchanged for t2va
        (["--task", "fl2va"], [{"cases": ["FL2VA-L"], "runs": 3}]),
        (["--task", "ref2va", "--runs", "1"], [{"cases": ["REF2VA-L"], "runs": 1}]),
        (
            ["--task", "ref2va", "--cases", "SIZE-V, REF2VA-M"],
            [{"cases": ["SIZE-V", "REF2VA-M"], "runs": 3}],
        ),
    ],
)
def test_cli_defaults_to_the_first_case_of_its_task(monkeypatch, argv, plan):
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"success": True}

    monkeypatch.setattr(T, "run_benchmark", fake)
    assert T.main(["--base-url", "http://127.0.0.1:1", *argv]) == 0
    assert seen["plan"] == plan


def test_skip_smoke_drops_the_smoke_assets_from_the_probe(pack, monkeypatch, tmp_path):
    from test_module.load_param_tests import minimax_h3_benchmark_test as T

    # SIZE-V needs prompt_std.txt + the 47 MB video; the ref2va smoke needs four other files
    (pack / "vid_max_8s_47mb_robot_street.mp4").write_bytes(b"\x00" * 64)
    (pack / "sha256s-bundle.txt").write_text(
        "".join(f"{M.sha256(str(p))}  {p.name}\n" for p in sorted(pack.iterdir()))
    )
    for name in ("prompt_min.txt", "vid_min_2s_city_skyline.mp4"):
        os.remove(pack / name)
    monkeypatch.setattr(M, "REPO_ASSETS", str(tmp_path / "no-repo-pack"))
    plan = [{"cases": ["SIZE-V"], "runs": 1}]
    kw = dict(
        base_url="http://127.0.0.1:1", task="ref2va", plan=plan, assets_dir=str(pack)
    )
    ran = T.run_benchmark(out_dir=str(tmp_path / "a"), **kw)
    assert sorted(p.split(" (")[0] for p in ran["probe"]) == [
        "missing asset prompt_min.txt",
        "missing asset vid_min_2s_city_skyline.mp4",
    ]
    skipped = T.run_benchmark(out_dir=str(tmp_path / "b"), skip_smoke=True, **kw)
    assert skipped["probe"] == []
    assert skipped["pregate"] and "unreachable" in skipped["pregate"][0]


def _poll_settings(**env):
    """(POLL_S, LOST_POLLS, TRANSPORT_POLLS) as a fresh import under ``env`` computes them."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("H3_")} | env
    code = ("from test_module._test_common.minimax_h3_bench import models as M; "
            "print(M.POLL_S, M.LOST_POLLS, M.TRANSPORT_POLLS)")  # fmt: skip
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(M.BASE))),
    )  # fmt: skip
    assert out.returncode == 0, out.stderr
    poll, lost, transport = out.stdout.split()
    return float(poll), int(lost), int(transport)


def test_poll_interval_keeps_the_give_up_windows_and_the_ref2va_smoke_budget():
    assert M.TIMEOUT_TABLE_S["BH1X"]["SMOKE-REF2VA"] == 2400
    assert _poll_settings() == (2.0, 30, 150)  # 60 s lost, 300 s unreachable
    # a faster or slower poll changes the counts, not the windows
    assert _poll_settings(H3_POLL_S="0.5") == (0.5, 120, 600)
    assert _poll_settings(H3_POLL_S="10") == (10.0, 6, 30)
    assert _poll_settings(H3_POLL_S="0") == (0.1, 600, 3000)  # never a busy loop
    # H3_LOST_POLLS is a count and still wins when set
    assert _poll_settings(H3_POLL_S="0.5", H3_LOST_POLLS="7") == (0.5, 7, 600)


def test_asset_search_order(monkeypatch, tmp_path):
    names = ("explicit", "env", "shared", "volume", "localdev", "repo")
    dirs = {k: tmp_path / k for k in names}
    dirs["volume"] = tmp_path / "pv" / "h3-assets"
    for d in dirs.values():
        d.mkdir(parents=True)
    monkeypatch.setenv("H3_ASSETS", str(dirs["env"]))
    monkeypatch.setattr(M, "SHARED_ASSETS", str(dirs["shared"]))
    monkeypatch.setenv("PERSISTENT_VOLUME_ROOT", str(tmp_path / "pv"))
    monkeypatch.setattr(M, "LOCALDEV_ASSETS", str(dirs["localdev"]))
    monkeypatch.setattr(M, "REPO_ASSETS", str(dirs["repo"]))
    monkeypatch.setattr(M, "PIN_MANIFESTS", (str(dirs["repo"] / "sha256s-bundle.txt"),))
    staged = [str(dirs[k]) for k in ("env", "shared", "volume", "localdev")]
    assert M.staged_asset_dirs() == staged
    assert H.resolve_assets_dir(str(dirs["explicit"])) == str(dirs["explicit"])
    assert H.resolve_assets_dir(str(tmp_path / "missing")) == str(dirs["env"])
    monkeypatch.delenv("H3_ASSETS")
    assert H.resolve_assets_dir(None) == str(dirs["shared"])
    monkeypatch.setenv("H3_ASSETS", str(dirs["env"]))
    try:
        M.configure(assets_dir=str(dirs["explicit"]))
        assert M.asset_dirs() == [str(dirs["explicit"]), *staged, str(dirs["repo"])]
        # per file, first hit wins: each location holds one file only it has
        for key in dirs:
            (dirs[key] / f"only_{key}.jpg").write_bytes(key.encode())
            assert M.asset_path(f"only_{key}.jpg") == str(dirs[key] / f"only_{key}.jpg")
        # the in-repo pack (media committed there) outranks a stale staged copy ...
        (dirs["repo"] / "vid.mp4").write_bytes(b"pinned")
        (dirs["shared"] / "vid.mp4").write_bytes(b"stale")
        (dirs["repo"] / "sha256s-bundle.txt").write_text(
            f"{M.sha256(str(dirs['repo'] / 'vid.mp4'))}  vid.mp4\n"
        )
        assert M.asset_path("vid.mp4") == str(dirs["repo"] / "vid.mp4")
        assert M.verify_assets(["vid.mp4"]) == []
        # ... but not a staged copy that is pinned too: the earlier location still wins
        (dirs["env"] / "vid.mp4").write_bytes(b"pinned")
        assert M.asset_path("vid.mp4") == str(dirs["env"] / "vid.mp4")
        # no copy matches: the first one found, and the probe says why
        (dirs["repo"] / "sha256s-bundle.txt").write_text(f"{'0' * 64}  vid.mp4\n")
        assert M.asset_path("vid.mp4") == str(dirs["env"] / "vid.mp4")
        assert M.verify_assets(["vid.mp4"]) == [
            "vid.mp4: sha256 differs from the manifest (not the pinned file)"
        ]
        # a directory that does not exist is not searched; the repo pack is last ...
        monkeypatch.setattr(M, "SHARED_ASSETS", str(tmp_path / "not-mounted"))
        assert str(tmp_path / "not-mounted") not in M.asset_dirs()
        M.configure(assets_dir=str(dirs["repo"]), explicit=False)
        assert M.asset_dirs()[-1] == str(dirs["repo"]) and M.asset_dirs()[0] == str(
            dirs["env"]
        )
        # ... unless it is the one named
        M.configure(assets_dir=str(dirs["repo"]))
        assert M.asset_dirs()[0] == str(dirs["repo"])
        assert M.asset_path("vid.mp4") == str(dirs["repo"] / "vid.mp4")
    finally:
        M.configure(
            assets_dir=M.REPO_ASSETS,
            out_dir=os.path.join(tempfile.gettempdir(), "minimax_h3_bench"),
        )
