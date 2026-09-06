# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Live MiniMax-H3 shape coverage: every published duration and aspect ratio, served from ONE worker
process in the orders that exposed bugs on OM Quad1 (2026-09-05), with every output judged for
content (audio clipping / noise, garbage frames, duration and canvas echo) -- see h3_media_checks.

Tier 2 (H3_LIVE_URL + deployment control): ~45 min for t2va, ~30 min for fl2va.  Run one class:

    bash tests/run_live_test.sh tests/test_minimax_h3_live_shapes.py -k T2va

Known-open bugs are ``xfail(strict=True)`` with the symptom in the reason, so they flip to XPASS
(and fail the run, asking to be un-marked) when fixed:

* audio corruption on rung replays -- a t2va request whose duration first appears AFTER the rung's
  trace was captured comes back with full-scale-noise audio (video intact), deterministically;
* fl2va with both keyframes at a 1008-rows/frame canvas exceeds the prompt arena cap (2053 > 2048).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_live_common as live  # noqa: E402
from h3_live_common import _fresh_or_skip, _need_task, require_live  # noqa: E402
from h3_live_common import assets, deployment, report, served_task  # noqa: E402,F401  (fixtures)
import h3_media_checks as media  # noqa: E402
from h3_live_sequences import ASPECT_RATIOS, DURATIONS_S, Spec, count_side_files, failures, replay_audio_failures, run_one, run_sequence  # noqa: E402

require_live()
pytestmark = [pytest.mark.live, pytest.mark.h3_tier2]

OUT = Path(os.environ.get("H3_LIVE_OUT_DIR") or os.path.join(tempfile.gettempdir(), f"h3-live-{os.environ.get('USER', 'x')}")) / "shapes"

AUDIO_REPLAY_BUG = pytest.mark.xfail(
    strict=False,   # non-strict since metal 8fb0c4c0483 (vocoder rebuilds the tpad mask per decode): an XPASS here is the fix landing
    reason="rung replay audio corruption: a request whose duration first appears after the rung's trace "
    "capture returns full-scale-noise audio (max_volume 0.0 dB), video intact, byte-identical on repeat "
    "(quad1 2026-09-05, metal 439b4bb8d5b; the audio decoder allocates per length under a live capture)",
)
T2VA_LADDER = [Spec("t2va", "16:9", s) for s in DURATIONS_S]
FL2VA_LADDER = [Spec("fl2va", "16:9", s, keyframes=(0,)) for s in DURATIONS_S]
ASPECT_SPECS = [Spec("t2va", a, 5) for a in ASPECT_RATIOS] + [Spec("t2va", a, 15) for a in ("21:9", "9:16", "1:1")]
# H3_LIVE_SKIP_ASPECTS="1:1,3:4" removes canvases from the fl2va specs below (aspects, two keyframes, full matrix).
# 2026-09-06, metal 8fb0c4c0483: the fl2va conditioner (text encoder, model_qwen3vl qkv_proj) hangs the mesh
# intermittently -- 3 of ~30 requests, seen at 1:1 (594 and 1205 presentation tokens) and 3:4 (1589) -- and a hung
# mesh costs the 300 s op timeout per request and a chip reset.  The knob keeps the rest of the canvas coverage
# runnable while that is open; nothing is skipped by default.
SKIP_ASPECTS = tuple(a.strip() for a in os.environ.get("H3_LIVE_SKIP_ASPECTS", "").split(",") if a.strip())
FL2VA_ASPECTS = tuple(a for a in ASPECT_RATIOS if a not in SKIP_ASPECTS)
FL2VA_ASPECT_SPECS = [Spec("fl2va", a, 5, keyframes=(0,)) for a in FL2VA_ASPECTS] + [Spec("fl2va", a, 15, keyframes=(0,)) for a in ("21:9", "1:1") if a in FL2VA_ASPECTS]

_state: dict = {"t2va_ladder_process": False}


def _assert_all_ok(results, what: str) -> None:
    bad = failures(results)
    assert not bad, f"{what}: {len(bad)}/{len(results)} requests bad:\n  " + "\n  ".join(bad)


class TestT2vaShapes:
    """One t2va process walks every duration (4 rungs), then every aspect ratio, then the bug orders."""

    def test_duration_ladder_4_to_15s(self, assets, served_task, deployment, report):
        """Ascending durations in a fresh process: each new rung is bound then captured; no OOM, valid content."""
        _fresh_or_skip("t2va", served_task, deployment)
        results = run_sequence(T2VA_LADDER, assets, deployment, report, OUT / "t2va-ladder")
        _state["t2va_ladder_process"] = all(r.ok for r in results)
        _assert_all_ok(results, "t2va ladder pass 1")
        rungs = sorted({r.rung for r in results if r.rung})
        assert rungs == [44032, 61440, 86016, 119808] or not any(r.rung for r in results), f"unexpected rungs {rungs}"

    @AUDIO_REPLAY_BUG
    def test_duration_ladder_second_pass_replays(self, assets, served_task, deployment, report):
        """The same ladder again in the SAME process: every rung is a traced replay now.  Exposes the
        audio corruption (5/12 outputs on 2026-09-05).  Reuses the process the previous test left
        when it ran; otherwise walks the ladder once first."""
        if not _state["t2va_ladder_process"]:
            _fresh_or_skip("t2va", served_task, deployment)
            _assert_all_ok(run_sequence(T2VA_LADDER, assets, deployment, report, OUT / "t2va-ladder-p1"), "t2va ladder pass 1")
        else:
            _need_task("t2va", served_task, deployment)
        results = run_sequence(T2VA_LADDER, assets, deployment, report, OUT / "t2va-ladder-p2")
        _assert_all_ok(results, "t2va ladder pass 2 (replays)")

    def test_aspect_ratios_echo_the_canvas(self, assets, served_task, deployment, report):
        """All six ratios at 5 s and the three canvases at 15 s: correct WxH, valid content, no OOM
        (binds rungs 22528/31744/44032/119808/86016 in one process)."""
        _fresh_or_skip("t2va", served_task, deployment)
        results = run_sequence(ASPECT_SPECS, assets, deployment, report, OUT / "t2va-aspects")
        _assert_all_ok(results, "t2va aspect ratios")

    @AUDIO_REPLAY_BUG
    def test_audio_survives_replay_of_a_length_seen_after_capture(self, assets, served_task, deployment, report):
        """Minimal repro: 12 s binds rung 119808, 13 s captures it, 13 s again replays -> bad audio (two replays
        so one lucky replay cannot flip the strict xfail)."""
        _fresh_or_skip("t2va", served_task, deployment)
        results = run_sequence([Spec("t2va", "16:9", 12), Spec("t2va", "16:9", 13), Spec("t2va", "16:9", 13), Spec("t2va", "16:9", 13)],
                               assets, deployment, report, OUT / "t2va-audio-repro")
        _assert_all_ok(results, "12 s, 13 s, 13 s, 13 s")

    def test_switch_back_to_a_smaller_rung(self, assets, served_task, deployment, report):
        """5 s -> 9 s -> 5 s: the second 5 s replays rung 44032 after 86016 was bound.  Crashed in
        ttnn.copy before upstream keyed _timestep_idx_state per pad_to; must stay clean."""
        _fresh_or_skip("t2va", served_task, deployment)
        results = run_sequence([Spec("t2va", "16:9", 5), Spec("t2va", "16:9", 9), Spec("t2va", "16:9", 5)],
                               assets, deployment, report, OUT / "t2va-switch-back")
        _assert_all_ok(results, "5 s, 9 s, 5 s")

    def test_same_seed_is_byte_identical(self, assets, served_task, deployment, report):
        """t2va is deterministic on the quad: the same body twice (bind, then capture) gives the same mp4."""
        _fresh_or_skip("t2va", served_task, deployment)
        a, b, c = run_sequence([Spec("t2va", "16:9", 5), Spec("t2va", "16:9", 5), Spec("t2va", "16:9", 5, seed=8)],
                               assets, deployment, report, OUT / "t2va-determinism")
        _assert_all_ok([a, b, c], "5 s twice + seed 8")
        assert a.sha256 == b.sha256, f"non-deterministic output: {a.mp4_bytes} B vs {b.mp4_bytes} B"
        assert c.sha256 != a.sha256, "a different seed produced the identical mp4: the seed is ignored"


class TestFl2vaShapes:
    """fl2va with one keyframe over the same space, then both keyframes at every canvas."""

    def test_one_keyframe_duration_ladder(self, assets, served_task, deployment, report):
        _fresh_or_skip("fl2va", served_task, deployment)
        _assert_all_ok(run_sequence(FL2VA_LADDER, assets, deployment, report, OUT / "fl2va-ladder"), "fl2va first-keyframe ladder")

    def test_one_keyframe_aspect_ratios(self, assets, served_task, deployment, report):
        """aspect_ratio decides the canvas (the keyframe is stretched to it): all six at 5 s, two at 15 s."""
        _fresh_or_skip("fl2va", served_task, deployment)
        _assert_all_ok(run_sequence(FL2VA_ASPECT_SPECS, assets, deployment, report, OUT / "fl2va-aspects"), "fl2va first-keyframe aspects")

    LONG_PROMPT = ("A red fox trotting through fresh snow at dawn, soft golden light, cinematic, shallow depth "
                   "of field, gentle camera drift, distant pine trees, sparkling frost, calm and quiet mood")  # >= 33 tokens

    @pytest.mark.parametrize("prompt_len", ["short", "long"])
    @pytest.mark.parametrize("aspect", ASPECT_RATIOS)
    def test_first_and_last_keyframes(self, aspect, prompt_len, assets, served_task, deployment, report):
        """Both keyframes (frame_pos 0 and -1), the API's documented fl2va request, at every canvas, with the
        suite's short prompt (~26 tokens) and a long one (>= 33).  Two keyframes at a 1008-rows/frame canvas
        are ~2016 vision tokens: until metal cfcb53a2404 (prompt cap 2048 -> 4160) the long prompt was refused
        (2053 > 2048) and the short one was admitted but its text, padded to 3072 rows, overran the prompt
        arena into the audio rows (noise audio).  Each symptom stays an xfail with its own reason; a clean
        completion passes (2026-09-06 on 8fb0c4c0483: 21:9, 16:9, 4:3, 3:4, 9:16 pass); anything else fails."""
        if aspect in SKIP_ASPECTS:
            pytest.skip(f"{aspect} skipped via H3_LIVE_SKIP_ASPECTS (fl2va conditioner hang)")
        _need_task("fl2va", served_task, deployment)
        prompt = live.PROMPT if prompt_len == "short" else self.LONG_PROMPT
        spec = Spec("fl2va", aspect, 5, keyframes=(0, -1), prompt=prompt, label=f"fl2va {aspect} 5s+first+last ({prompt_len} prompt)")
        res = run_one(spec, assets, deployment, OUT / "fl2va-two-keyframes")
        report.add(**res.row())
        if res.ok:
            return
        if res.status == "failed" and res.error and "arena caps" in res.error and "prompt tokens" in res.error:
            deployment.poisoned = False   # an admission refusal frees everything; no reset needed
            pytest.xfail(f"fl2va first+last at {aspect} refused by the prompt arena cap (MiniMaxH3ArenaCaps.prompt 2048): {res.error[:120]}")
        if res.status == "completed" and res.verdict and any("audio looks like noise" in x for x in res.verdict.reasons):
            pytest.xfail(f"fl2va first+last at {aspect} admitted under the cap but the text overruns the prompt arena into the audio rows: {res.verdict.summary()[:160]}")
        pytest.fail(res.describe())


@pytest.mark.h3_tier3
class TestFullMatrix:
    """Every duration x every aspect ratio in ONE process per task (72 requests, ~60-75 min each).
    Strict on completion, duration/canvas echo, video content, and audio on binds/captures; noise audio
    on traced REPLAYS is the known bug and is reported as an xfail instead of failing the matrix."""

    MATRIX = [(a, s) for a in ASPECT_RATIOS for s in DURATIONS_S]

    def _run(self, task, keyframes, assets, served_task, deployment, report):
        _fresh_or_skip(task, served_task, deployment)
        aspects = FL2VA_ASPECTS if task == "fl2va" else ASPECT_RATIOS
        specs = [Spec(task, a, s, keyframes=keyframes) for a, s in self.MATRIX if a in aspects]
        results = run_sequence(specs, assets, deployment, report, OUT / f"{task}-full-matrix")
        hard = failures(results, ignore_replay_audio=True)
        assert not hard, f"{task} full matrix: {len(hard)}/{len(results)} requests bad (status/echo/video, or noise audio on a bind/capture):\n  " + "\n  ".join(hard)
        audio = replay_audio_failures(results)
        if audio:
            pytest.xfail(f"{task} full matrix: {len(audio)}/{len(results)} replays with noise audio (rung replay bug):\n  " + "\n  ".join(audio[:6]))

    def test_t2va(self, assets, served_task, deployment, report):
        self._run("t2va", (), assets, served_task, deployment, report)

    def test_fl2va_one_keyframe(self, assets, served_task, deployment, report):
        self._run("fl2va", (0,), assets, served_task, deployment, report)


@pytest.mark.h3_tier2
class TestRef2vaShapes:
    def test_duration_and_aspect_echo(self, assets, served_task, deployment, report):
        """ref2va (one image reference) must honour duration_seconds and aspect_ratio like the other tasks:
        9 s at 9:16 then 5 s at 16:9, served canvas and duration checked, content judged."""
        _fresh_or_skip("ref2va", served_task, deployment)
        results = run_sequence([Spec("ref2va", "9:16", 9, ref_images=1), Spec("ref2va", "16:9", 5, ref_images=1)],
                               assets, deployment, report, OUT / "ref2va-echo")
        _assert_all_ok(results, "ref2va duration/aspect echo")


@pytest.mark.h3_tier3
class TestRungAudioMap:
    """Which rungs corrupt audio on the replay of their capture duration: bind, capture, replay the same
    length twice, per rung.  Strict on status/echo/video; replay audio noise is the known bug (xfail)."""

    @pytest.mark.parametrize(("rung", "bind_s", "capture_s"), [
        pytest.param(44032, 4, 5, id="44032"), pytest.param(61440, 6, 7, id="61440"),
        pytest.param(86016, 9, 10, id="86016"), pytest.param(119808, 12, 13, id="119808"),
    ])
    def test_replay_of_capture_duration(self, rung, bind_s, capture_s, assets, served_task, deployment, report):
        _fresh_or_skip("t2va", served_task, deployment)
        specs = [Spec("t2va", "16:9", bind_s), Spec("t2va", "16:9", capture_s), Spec("t2va", "16:9", capture_s), Spec("t2va", "16:9", capture_s)]
        results = run_sequence(specs, assets, deployment, report, OUT / f"t2va-rung-audio-{rung}")
        hard = failures(results, ignore_replay_audio=True)
        assert not hard, f"rung {rung}: " + "; ".join(hard)
        audio = replay_audio_failures(results)
        if audio:
            pytest.xfail(f"rung {rung}: {len(audio)}/2 replays of the capture duration have noise audio: " + "; ".join(audio))


@pytest.mark.h3_tier2
class TestRequestFields:
    """Prompt and field variants a real client sends."""

    def test_prompt_variants(self, assets, served_task, deployment, report):
        _fresh_or_skip("t2va", served_task, deployment)
        long_prompt = " ".join(["a red fox trots through fresh snow at dawn while golden light glints on frost"] * 60)   # ~1000 tokens
        specs = [
            Spec("t2va", "16:9", 5, prompt="黎明时分一只红狐穿过新雪，柔和的金色光线，电影感", label="unicode prompt"),
            Spec("t2va", "16:9", 5, prompt=live.PROMPT + " --neg", label="with negative_prompt"),
            Spec("t2va", "16:9", 5, prompt=long_prompt, label="~1000-token prompt"),
        ]
        bodies = {"with negative_prompt": {"negative_prompt": "blurry, text, watermark"}}
        results = []
        for spec in specs:
            res = run_one(spec, assets, deployment, OUT / "t2va-fields") if spec.label not in bodies else _run_with_extra(spec, bodies[spec.label], assets, deployment)
            report.add(**res.row())
            results.append(res)
        _assert_all_ok(results, "prompt / field variants")

    @pytest.mark.xfail(strict=True, reason="the SHM wire holds 2048 prompt bytes: a prompt and the same prompt with words appended "
                       "past byte 2048 give byte-identical videos (silent truncation), instead of different videos or a 422")
    def test_long_prompt_is_delivered_whole(self, assets, served_task, deployment, report):
        _need_task("t2va", served_task, deployment)
        base = ("a red fox trots through fresh snow at dawn while golden light glints on frost, " * 26)[:2048]   # exactly 2048 bytes (ASCII)
        a, b = run_sequence([Spec("t2va", "16:9", 5, prompt=base, label="2048-byte prompt"),
                             Spec("t2va", "16:9", 5, prompt=base + " a purple elephant stands on the beach at night", label="prompt + tail past byte 2048")],
                            assets, deployment, report, OUT / "t2va-prompt-bytes")
        _assert_all_ok([a, b], "prompt byte-cap probe")
        assert a.sha256 != b.sha256, "text past byte 2048 changed nothing: the prompt was truncated on the wire"

    @pytest.mark.xfail(strict=True, reason="negative_prompt never reaches the H3 pipeline (guidance-distilled, runner passes "
                       "prompt/shape/steps/seed only): with and without it the output is byte-identical")
    def test_negative_prompt_changes_the_output(self, assets, served_task, deployment, report):
        _need_task("t2va", served_task, deployment)
        plain = run_one(Spec("t2va", "16:9", 5, label="no negative_prompt"), assets, deployment, OUT / "t2va-negprompt")
        report.add(**plain.row())
        neg = _run_with_extra(Spec("t2va", "16:9", 5, label="negative_prompt=blurry"), {"negative_prompt": "blurry, text, watermark, snow"}, assets, deployment)
        report.add(**neg.row())
        _assert_all_ok([plain, neg], "negative_prompt probe")
        assert plain.sha256 != neg.sha256, "negative_prompt is a silent no-op"


@pytest.mark.h3_tier2
class TestApiBehaviour:
    """HTTP behaviours around a job's life that clients hit every day."""

    @pytest.mark.xfail(strict=True, reason="the 202 body, GET /generations/{id} and GET /jobs echo request_parameters including the "
                       "full base64 media (job_manager.to_public_dict): an fl2va submit answers with megabytes")
    def test_submit_response_does_not_echo_media(self, assets, served_task, deployment):
        _need_task("fl2va", served_task, deployment)
        body = Spec("fl2va", "16:9", 5, keyframes=(0,)).body(assets)
        code, resp = live.http("POST", live.ENDPOINT["fl2va"], body, timeout=120)
        assert code in (200, 202) and isinstance(resp, dict) and resp.get("id"), f"submit -> {code}"
        size = len(json.dumps(resp))
        live._wait_terminal(resp["id"], time.time(), deployment)
        live.http("DELETE", f"/v1/videos/generations/{resp['id']}", timeout=60)
        assert size < 65536, f"202 body is {size} bytes (echoes the keyframe base64)"

    @pytest.mark.xfail(strict=True, reason="/download of a job that is still queued/in_progress is a 404, indistinguishable from an "
                       "unknown id; the contract is 409/425 'not ready yet'")
    def test_download_while_running_is_not_a_404(self, assets, served_task, deployment):
        _need_task("t2va", served_task, deployment)
        code, resp = live.http("POST", live.ENDPOINT["t2va"], Spec("t2va", "16:9", 5).body(assets), timeout=120)
        assert code in (200, 202) and resp.get("id")
        dcode, _ = live.http("GET", f"/v1/videos/generations/{resp['id']}/download", timeout=30)
        live._wait_terminal(resp["id"], time.time(), deployment)
        live.http("DELETE", f"/v1/videos/generations/{resp['id']}", timeout=60)
        assert dcode in (409, 425), f"download while running -> {dcode}"

    @pytest.mark.xfail(strict=True, reason="POST /cancel on a finished job answers 404 'Video job not found' although GET still "
                       "returns the job; DELETE models the same case as 409")
    def test_cancel_of_a_finished_job_is_a_409(self, assets, served_task, deployment):
        _need_task("t2va", served_task, deployment)
        res = run_one(Spec("t2va", "16:9", 5, label="job to cancel after completion"), assets, deployment, OUT / "api", delete=False)
        assert res.ok, res.describe()
        code, _ = live.http("POST", f"/v1/videos/generations/{res.job_id}/cancel", timeout=30)
        live.http("DELETE", f"/v1/videos/generations/{res.job_id}", timeout=60)
        assert code == 409, f"cancel of a completed job -> {code}"


@pytest.mark.h3_tier2
class TestOutputContract:
    @pytest.mark.xfail(strict=True, reason="the encoder writes untagged yuv420p (BT.601 matrix, no color_space/primaries/transfer "
                       "tags); HD players assume BT.709 so every clip is colour-shifted")
    def test_colour_is_tagged(self, assets, served_task, deployment, report):
        _need_task("t2va", served_task, deployment)
        res = run_one(Spec("t2va", "16:9", 5, label="colour-tag probe"), assets, deployment, OUT / "output-contract")
        report.add(**res.row())
        assert res.ok, res.describe()
        p = res.verdict.probe
        assert p.color_space and p.color_primaries and p.color_transfer, f"untagged colour: space={p.color_space} primaries={p.color_primaries} transfer={p.color_transfer}"

    @pytest.mark.xfail(strict=True, reason="the mux uses -shortest and the 40 Hz audio grid is shorter than 17n+5 frames at 4, 6 and "
                       "15 s, so those clips carry 17n+4 video frames (the last frame -- an fl2va last keyframe -- is dropped)")
    def test_all_frames_are_delivered_at_4s(self, assets, served_task, deployment, report):
        _need_task("t2va", served_task, deployment)
        res = run_one(Spec("t2va", "16:9", 4, label="frame-count probe 4 s"), assets, deployment, OUT / "output-contract")
        report.add(**res.row())
        assert res.ok, res.describe()
        dropped = media.frames_dropped(res.verdict.probe, 4)
        assert dropped == 0, f"{dropped} of {media.expected_frames(4)} frames missing from the container"


@pytest.mark.h3_tier2
class TestFl2vaConditioning:
    def test_keyframes_are_honoured(self, assets, served_task, deployment, report):
        """fl2va at 4:3 with first+last keyframes: the first decoded frame must resemble the FIRST keyframe
        more than the last one and vice versa (SSIM), so ignored, swapped or dropped keyframes fail."""
        _need_task("fl2va", served_task, deployment)
        res = run_one(Spec("fl2va", "4:3", 5, keyframes=(0, -1), label="keyframe fidelity 4:3"), assets, deployment, OUT / "fl2va-fidelity", delete=False)
        report.add(**res.row())
        assert res.ok, res.describe()
        dur = res.verdict.probe.duration_s or 5.0
        f_first, f_last = media.keyframe_ssim(res.mp4, assets.key_first, 0.0), media.keyframe_ssim(res.mp4, assets.key_last, 0.0)
        l_first, l_last = media.keyframe_ssim(res.mp4, assets.key_first, max(0.0, dur - 0.1)), media.keyframe_ssim(res.mp4, assets.key_last, max(0.0, dur - 0.1))
        live.http("DELETE", f"/v1/videos/generations/{res.job_id}", timeout=60)
        report.add(combo="keyframe SSIM", task="fl2va", request="ssim", status="measured", wall_s=0.0, job_id=res.job_id,
                   first_vs_first=f_first, first_vs_last=f_last, last_vs_first=l_first, last_vs_last=l_last)
        assert None not in (f_first, f_last, l_first, l_last), "ssim measurement failed"
        assert f_first > f_last, f"first frame is closer to the LAST keyframe ({f_first:.3f} vs {f_last:.3f}): swapped or ignored"
        assert l_last > l_first, f"last frame is closer to the FIRST keyframe ({l_last:.3f} vs {l_first:.3f}): swapped or dropped"

    @pytest.mark.xfail(strict=True, reason="fl2va is not deterministic for a fixed seed across bind/capture in one process "
                       "(identical bodies: md5 differ, PSNR 22.6 dB / SSIM 0.84 on 2026-09-05 sweepB r02 vs r13)")
    def test_same_seed_is_byte_identical(self, assets, served_task, deployment, report):
        _fresh_or_skip("fl2va", served_task, deployment)
        a, b = run_sequence([Spec("fl2va", "16:9", 5, keyframes=(0,)), Spec("fl2va", "16:9", 5, keyframes=(0,))], assets, deployment, report, OUT / "fl2va-determinism")
        _assert_all_ok([a, b], "fl2va 5 s twice")
        assert a.sha256 == b.sha256, f"fl2va outputs differ for identical requests: {a.mp4_bytes} B vs {b.mp4_bytes} B"


def _run_with_extra(spec, extra: dict, assets, deployment):
    """run_one with extra body fields (negative_prompt ...): submit by hand, then reuse run_one's polling path."""
    import h3_live_sequences as seq
    body = spec.body(assets); body.update(extra)
    orig = seq.Spec.body
    try:
        seq.Spec.body = lambda self, a, _b=body: _b   # one-off override for this spec
        return run_one(spec, assets, deployment, OUT / "t2va-fields")
    finally:
        seq.Spec.body = orig


@pytest.mark.h3_tier2
class TestQueue:
    def test_three_queued_jobs_complete_in_order(self, assets, served_task, deployment, report):
        """Three back-to-back submits: all accepted (202), queue_size reflects them, they complete in
        submission order and every output is valid."""
        _need_task("t2va", served_task, deployment)
        t0 = time.time()
        ids = []
        for i, sec in enumerate((5, 6, 5)):
            code, resp = live.http("POST", live.ENDPOINT["t2va"], Spec("t2va", "16:9", sec, seed=10 + i).body(assets), timeout=120)
            assert code in (200, 202) and resp.get("id"), f"submit {i} -> {code} {resp}"
            ids.append(resp["id"])
        _, lv = live.http("GET", "/tt-liveness", timeout=15)
        queued_seen = lv.get("queue_size") if isinstance(lv, dict) else None
        done_at: dict = {}
        while len(done_at) < 3 and time.time() - t0 < 900:
            for jid in ids:
                if jid in done_at:
                    continue
                _, job = live.http("GET", f"/v1/videos/generations/{jid}", timeout=30)
                st = job.get("status") if isinstance(job, dict) else None
                if st in ("completed", "failed", "cancelled"):
                    done_at[jid] = (time.time(), st)
            time.sleep(3)
        order = sorted(done_at, key=lambda j: done_at[j][0])
        report.add(combo="3 queued jobs", task="t2va", request="queue", status="+".join(done_at[j][1] for j in ids) if len(done_at) == 3 else "incomplete",
                   wall_s=round(time.time() - t0, 1), job_id=",".join(i[:8] for i in ids), queue_size_seen=queued_seen)
        assert len(done_at) == 3, f"only {len(done_at)}/3 jobs finished in 900 s"
        assert all(done_at[j][1] == "completed" for j in ids), {j[:8]: done_at[j][1] for j in ids}
        assert order == ids, f"completion order {[o[:8] for o in order]} != submission order {[i[:8] for i in ids]}"
        for jid in ids:
            dl = OUT / "queue" / f"{jid[:8]}.mp4"; dl.parent.mkdir(parents=True, exist_ok=True)
            code, _ = live.http("GET", f"/v1/videos/generations/{jid}/download", timeout=300, save_to=dl)
            assert code == 200
            v = media.judge(dl, expect_canvas=(1344, 768))
            assert v.ok, f"{jid[:8]}: {v.summary()}"
            live.http("DELETE", f"/v1/videos/generations/{jid}", timeout=60)

    def test_cancel_running_job_then_next_request_is_clean(self, assets, served_task, deployment, report):
        """Cancel a 12 s job mid-flight, then submit a 5 s one: the second must complete with valid
        content (the worker may still be finishing the abandoned run; its wall time is recorded)."""
        _need_task("t2va", served_task, deployment)
        code, resp = live.http("POST", live.ENDPOINT["t2va"], Spec("t2va", "16:9", 12).body(assets), timeout=120)
        assert code in (200, 202) and resp.get("id"), f"submit -> {code} {resp}"
        victim = resp["id"]
        time.sleep(8)
        code, resp = live.http("POST", f"/v1/videos/generations/{victim}/cancel", timeout=60)
        assert code == 200, f"cancel -> {code} {resp}"
        nxt = run_one(Spec("t2va", "16:9", 5, label="after cancelling a running job"), assets, deployment, OUT / "queue")
        report.add(**nxt.row())
        live.http("DELETE", f"/v1/videos/generations/{victim}", timeout=60)
        assert nxt.ok, nxt.describe()


@pytest.mark.h3_tier2
class TestSideFiles:
    def test_fl2va_requests_leave_no_side_files(self, assets, served_task, deployment, report):
        """Every fl2va request ships its keyframes through a tt_img_*.json side-file on tmpfs; the SP
        runner must unlink it after the response (a leak fills /dev/shm on the API host)."""
        before = count_side_files()
        if before is None:
            pytest.skip("counts /dev/shm side-files: run on the server host")
        _need_task("fl2va", served_task, deployment)
        results = run_sequence([Spec("fl2va", "16:9", 5, keyframes=(0,), seed=21), Spec("fl2va", "16:9", 5, keyframes=(0,), seed=22)],
                               assets, deployment, report, OUT / "fl2va-sidefiles")
        _assert_all_ok(results, "fl2va side-file requests")
        time.sleep(2)
        after = count_side_files()
        assert after == before, f"tt_img_* side-files: {before} before, {after} after two fl2va requests (leak)"


@pytest.mark.h3_tier3
class TestSoak:
    def test_identical_requests_stay_flat(self, assets, served_task, deployment, report):
        """Twelve identical 5 s requests in one process: every output valid and identical, and the
        wall time does not creep (max of requests 3..12 <= 2x their median) -- residual device memory or
        host-side growth shows up here first."""
        _fresh_or_skip("t2va", served_task, deployment)
        results = run_sequence([Spec("t2va", "16:9", 5) for _ in range(12)], assets, deployment, report, OUT / "t2va-soak")
        _assert_all_ok(results, "soak")
        shas = {r.sha256 for r in results}
        assert len(shas) == 1, f"{len(shas)} distinct outputs for identical requests"
        walls = sorted(r.wall_s for r in results[2:])
        median = walls[len(walls) // 2]
        assert max(walls) <= 2 * median, f"wall time creeps: median {median}s, max {max(walls)}s"


@pytest.mark.h3_tier3
class TestFl2vaRepeats:
    def test_two_keyframes_repeat_at_4x3(self, assets, served_task, deployment, report):
        """The two-keyframe request that fits under the cap, three times in one process (bind, capture,
        replay): no OOM, valid content each time."""
        _fresh_or_skip("fl2va", served_task, deployment)
        results = run_sequence([Spec("fl2va", "4:3", 5, keyframes=(0, -1)) for _ in range(3)], assets, deployment, report, OUT / "fl2va-two-keyframes-repeat")
        _assert_all_ok(results, "fl2va 4:3 two keyframes x3")


@pytest.mark.h3_tier3
class TestRef2vaReferences:
    def test_video_reference_with_duration(self, assets, served_task, deployment, report):
        """ref2va with one video reference at 9 s / 9:16 and one image reference at 5 s / 16:9."""
        _fresh_or_skip("ref2va", served_task, deployment)
        results = run_sequence([Spec("ref2va", "9:16", 9, ref_videos=1), Spec("ref2va", "16:9", 5, ref_images=1)],
                               assets, deployment, report, OUT / "ref2va-video-ref")
        _assert_all_ok(results, "ref2va video reference")
