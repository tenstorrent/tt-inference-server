# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tier-0 tests for h3_media_checks: synthetic clips built with ffmpeg's lavfi sources stand in for
clean and corrupted H3 outputs, so the thresholds are exercised without any model or device."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ holds the helper module
import h3_media_checks as mc  # noqa: E402

pytestmark = pytest.mark.skipif(not mc.have_ffmpeg(), reason="ffmpeg/ffprobe needed to synthesise clips")


def _clip(out: Path, seconds: float, video: str, audio: str, size: str = "1344x768") -> Path:
    """One mp4: a lavfi video source + a lavfi audio source, h264 + aac like the server emits."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           "-f", "lavfi", "-i", f"{video}=size={size}:rate=24:duration={seconds}",
           "-f", "lavfi", "-i", audio.format(dur=seconds),
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("h3media")
    return {
        # smooth moving pattern + a quiet tone: what a healthy H3 output looks like to the metrics
        "clean": _clip(d / "clean.mp4", 5.167, "testsrc2", "sine=frequency=440:sample_rate=32000:duration={dur},volume=-24dB"),
        # healthy video, audio replaced by full-scale white noise (the 2026-09-05 t2va corruption)
        "noise_audio": _clip(d / "noise_audio.mp4", 5.167, "testsrc2", "anoisesrc=color=white:sample_rate=32000:amplitude=1.0:duration={dur}"),
        # random-noise frames (the old-tree tiled garbage), audio fine
        "noise_video": _clip(d / "noise_video.mp4", 5.167, "nullsrc", "sine=frequency=440:sample_rate=32000:duration={dur},volume=-24dB"),
        "silent": _clip(d / "silent.mp4", 5.167, "testsrc2", "anullsrc=sample_rate=32000:duration={dur}"),
        # loud but legitimate: 440 Hz bursts at -0.9 dB peak, 20 % duty -> mean ~ -11 dB, nothing at the rails
        # (the 3:4 5 s drone the model produces on both metal trees: mean -13..-15 dB, max -0.8..-1.3 dB)
        "loud_tone": _clip(d / "loud_tone.mp4", 5.167, "testsrc2",
                           "aevalsrc='0.9*sin(440*2*PI*t)*lt(mod(t\\,1)\\,0.2)':sample_rate=32000:duration={dur}"),
        "portrait_9s": _clip(d / "portrait.mp4", 9.417, "testsrc2", "sine=frequency=440:sample_rate=32000:duration={dur},volume=-24dB", size="768x1344"),
    }


def _noise_video(path: Path) -> Path:
    """nullsrc is black; overwrite with per-pixel random noise via geq so every frame is garbage."""
    out = path.with_name("noise_video_geq.mp4")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                    "-vf", "geq=lum='random(1)*255':cb='random(2)*255':cr='random(3)*255'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)],
                   check=True, capture_output=True)
    return out


def test_expected_frames_rounds_up_to_17n_plus_5():
    # the served-duration policy the API documents: 4 s -> 107 frames (4.458 s), 8 s exact (192), 15 s -> 362
    assert mc.expected_frames(4) == 107
    assert mc.expected_frames(5) == 124
    assert mc.expected_frames(8) == 192
    assert mc.expected_frames(9) == 226
    assert mc.expected_frames(15) == 362
    assert abs(mc.expected_duration_s(9) - 9.4167) < 1e-3


def test_clean_clip_passes(clips):
    v = mc.judge(clips["clean"], expect_seconds=5, expect_canvas=(1344, 768))
    assert v.ok, v.summary()
    assert v.audio.max_db < mc.AUDIO_MAX_DB_CLIPPED and v.audio.mean_db < mc.AUDIO_MEAN_DB_LOUD
    assert all(not f.garbage for f in v.frames), v.summary()


def test_full_scale_noise_audio_is_rejected(clips):
    v = mc.judge(clips["noise_audio"], expect_seconds=5)
    assert not v.ok and any("audio looks like noise" in r for r in v.reasons), v.summary()
    assert v.audio.clipped and v.audio.railed, (v.audio.rail_share, v.audio.mean_db)


def test_loud_tonal_audio_is_not_noise(clips):
    """2026-09-06: a 3:4 5 s t2va soundtrack at mean -14.7 dB / max -1.3 dB was flagged as noise by the old
    loudness-only rule, although the same spec produced the same loud drone on the previous metal tree and
    the waveform never touches the rails.  Loud is a note, railed is the failure."""
    v = mc.judge(clips["loud_tone"], expect_seconds=5)
    assert v.ok, v.summary()
    assert v.audio.clipped or v.audio.loud, (v.audio.mean_db, v.audio.max_db)
    assert not v.audio.railed and (v.audio.rail_share or 0) < mc.AUDIO_RAIL_SHARE, v.audio.rail_share
    assert any("loud but not railed" in n for n in v.notes), v.notes


def test_noise_video_frames_are_rejected(clips):
    v = mc.judge(_noise_video(clips["noise_video"]), expect_seconds=5)
    assert not v.ok and any("looks like noise" in r and "frame" in r for r in v.reasons), v.summary()
    assert any(f.garbage for f in v.frames)


def test_silent_audio_is_reported(clips):
    v = mc.judge(clips["silent"], expect_seconds=5)
    assert not v.ok and any("silent" in r for r in v.reasons), v.summary()


def test_duration_and_canvas_echo_checks(clips):
    ok = mc.judge(clips["portrait_9s"], expect_seconds=9, expect_canvas=(768, 1344), check_video=False, check_audio=False)
    assert ok.ok, ok.summary()
    wrong_dur = mc.judge(clips["portrait_9s"], expect_seconds=5, check_video=False, check_audio=False)
    assert not wrong_dur.ok and "duration" in wrong_dur.reasons[0]
    wrong_canvas = mc.judge(clips["portrait_9s"], expect_canvas=(1344, 768), check_video=False, check_audio=False)
    assert not wrong_canvas.ok and "canvas" in wrong_canvas.reasons[0]


def test_missing_stream_fails_fast(tmp_path):
    p = tmp_path / "video_only.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True, capture_output=True)
    v = mc.judge(p)
    assert not v.ok and "audio=0" in v.reasons[0]


def test_probe_reports_fps_codecs_and_stream_durations(clips):
    p = mc.probe(clips["clean"])
    assert abs(p.fps - 24) < 0.01 and p.video_codec == "h264" and p.audio_codec == "aac"
    assert p.video_duration_s and p.audio_duration_s and abs(p.video_duration_s - p.audio_duration_s) < mc.AUDIO_VIDEO_DRIFT_S


def test_audio_video_length_mismatch_is_rejected(clips, tmp_path):
    """A soundtrack that stops halfway (or a missing one padded with silence) must not pass as valid."""
    short = tmp_path / "short_audio.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=5",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=2,volume=-24dB",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(short)], check=True, capture_output=True)
    v = mc.judge(short, check_video=False)
    assert not v.ok and any("out of step" in r for r in v.reasons), v.summary()
