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

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_live_common as live  # noqa: E402
from h3_live_common import _fresh_or_skip, _need_task, require_live  # noqa: E402
from h3_live_common import assets, deployment, report, served_task  # noqa: E402,F401  (fixtures)
from h3_live_sequences import ASPECT_RATIOS, DURATIONS_S, Spec, failures, replay_audio_failures, run_one, run_sequence  # noqa: E402

require_live()
pytestmark = [pytest.mark.live, pytest.mark.h3_tier2]

OUT = Path(os.environ.get("H3_LIVE_OUT_DIR") or os.path.join(tempfile.gettempdir(), f"h3-live-{os.environ.get('USER', 'x')}")) / "shapes"

AUDIO_REPLAY_BUG = pytest.mark.xfail(
    strict=True,
    reason="rung replay audio corruption: a request whose duration first appears after the rung's trace "
    "capture returns full-scale-noise audio (max_volume 0.0 dB), video intact, byte-identical on repeat "
    "(quad1 2026-09-05, metal 439b4bb8d5b; the audio decoder allocates per length under a live capture)",
)
T2VA_LADDER = [Spec("t2va", "16:9", s) for s in DURATIONS_S]
FL2VA_LADDER = [Spec("fl2va", "16:9", s, keyframes=(0,)) for s in DURATIONS_S]
ASPECT_SPECS = [Spec("t2va", a, 5) for a in ASPECT_RATIOS] + [Spec("t2va", a, 15) for a in ("21:9", "9:16", "1:1")]
FL2VA_ASPECT_SPECS = [Spec("fl2va", a, 5, keyframes=(0,)) for a in ASPECT_RATIOS] + [Spec("fl2va", a, 15, keyframes=(0,)) for a in ("21:9", "1:1")]

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
        assert rungs == [44032, 61440, 86016, 118784] or not any(r.rung for r in results), f"unexpected rungs {rungs}"

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
        (binds rungs 22528/31744/44032/118784/86016 in one process)."""
        _fresh_or_skip("t2va", served_task, deployment)
        results = run_sequence(ASPECT_SPECS, assets, deployment, report, OUT / "t2va-aspects")
        _assert_all_ok(results, "t2va aspect ratios")

    @AUDIO_REPLAY_BUG
    def test_audio_survives_replay_of_a_length_seen_after_capture(self, assets, served_task, deployment, report):
        """Minimal repro: 12 s binds rung 118784, 13 s captures it, 13 s again replays -> bad audio (two replays
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
        a, b = run_sequence([Spec("t2va", "16:9", 5), Spec("t2va", "16:9", 5)], assets, deployment, report, OUT / "t2va-determinism")
        _assert_all_ok([a, b], "5 s twice")
        assert a.sha256 == b.sha256, f"non-deterministic output: {a.mp4_bytes} B vs {b.mp4_bytes} B"


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
        are ~2016 vision tokens, so: long prompt -> the worker refuses the request (prompt cap 2048);
        short prompt -> admitted, but the text padded to 3072 rows overruns the 2048-row prompt arena
        into the audio rows and the audio comes back as noise.  Each symptom is an xfail with its own
        reason; a clean completion passes (the fix landed); any other outcome fails."""
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
        specs = [Spec(task, a, s, keyframes=keyframes) for a, s in self.MATRIX]
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
        pytest.param(86016, 9, 10, id="86016"), pytest.param(118784, 12, 13, id="118784"),
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
