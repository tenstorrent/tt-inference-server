# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Media validity checks for MiniMax-H3 outputs (ffmpeg/ffprobe based, no model in the loop).

Why these exist: the H3 pipeline can complete a job with a well-formed mp4 whose CONTENT is garbage --
2026-09-05 on OM Quad1 the audio track came back as full-scale noise (max volume 0.0 dB) on one in
four t2va requests while the video was byte-identical to a clean run, and an older metal tree produced
tiled-noise video frames. ``ffprobe`` stream checks ("has video + audio") pass on all of those, so the
live tests judge the decoded samples themselves:

* audio: ``volumedetect`` mean/max.  Clean H3 soundtracks measured mean -28..-53 dB, max -10..-35 dB
  (77 clean samples); corrupted ones mean -0.3..-13 dB, max 0.0 dB.  A track is *clipped* when the
  max is above ``AUDIO_MAX_DB_CLIPPED`` (-1.0 dB) and *loud* when the mean is above
  ``AUDIO_MEAN_DB_LOUD`` (-15 dB); either fails the check.  Pure silence (mean < -80) is reported too.
* video: per-frame ``entropy`` (ffmpeg ``entropy`` filter, normal mode, Y plane) and the size of a
  fixed-quality JPEG of the frame.  Calibrated 2026-09-05/06 on quad1 outputs: clean H3 frames measure
  0.068-0.115 bytes/pixel at ``-q:v 3`` (672 px wide; 16 clips, 6 canvases, two prompts) with Y-entropy
  5.9-7.55 bits -- a detailed scene alone can reach 7.5+, so entropy is never used on its own;
  tiled-noise garbage frames measure 0.29-0.46 B/px at 7.77-7.83 bits.  A frame is garbage when
  ``bytes_per_pixel > VIDEO_MAX_BYTES_PER_PIXEL`` (0.20), or when entropy > ``VIDEO_HIGH_ENTROPY``
  (7.7) while ``bytes_per_pixel > VIDEO_HIGH_ENTROPY_BYTES_PER_PIXEL`` (0.12).  Six frames spread over
  the clip are sampled.  The thresholds catch broken decodes, they do not grade aesthetics.
* container bit rate: ``bit_rate / (width * height * 24)`` -- clean H3 clips encode at <= 0.34
  bits/pixel/frame (CRF 23; short busy clips are the high end), tiled-noise garbage at 0.77-0.88;
  ``VIDEO_MAX_BITS_PER_PIXEL_FRAME`` (0.6) flags a clip from ffprobe alone, before any frame is decoded.
* geometry: served duration and canvas must echo the request (durations round UP to ``17n + 5``
  frames at 24 fps -- ``expected_frames`` / ``expected_duration_s`` reproduce the server's math).

Everything degrades to a *skipped* sub-check (not a failure) when ffmpeg/ffprobe are missing.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

FPS = 24
FRAMES_PER_CHUNK = 17
FRAMES_CHUNK_OFFSET = 5

AUDIO_MAX_DB_CLIPPED = -1.0
AUDIO_MEAN_DB_LOUD = -15.0
AUDIO_MEAN_DB_SILENT = -80.0
VIDEO_MAX_BYTES_PER_PIXEL = 0.20        # alone: unmistakable garbage
VIDEO_HIGH_ENTROPY = 7.7                # only together with a raised byte rate
VIDEO_HIGH_ENTROPY_BYTES_PER_PIXEL = 0.12
VIDEO_MAX_BITS_PER_PIXEL_FRAME = 0.6   # container bit rate / (w*h*fps); clean H3 <= 0.34, garbage 0.77-0.88
VIDEO_SAMPLE_WIDTH = 672
VIDEO_JPEG_QUALITY = 3


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def expected_frames(seconds: float) -> int:
    """The server's frame count for ``duration_seconds``: round(24*s) snapped UP to 17n+5."""
    n = int(round(seconds * FPS))
    while (n - FRAMES_CHUNK_OFFSET) % FRAMES_PER_CHUNK:
        n += 1
    return n


def expected_duration_s(seconds: float) -> float:
    return expected_frames(seconds) / FPS


def _run(cmd: list[str], timeout: int = 120) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


@dataclass
class Probe:
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    nb_frames: int | None = None
    video_streams: int = 0
    audio_streams: int = 0
    bit_rate: int | None = None
    sample_rate: int | None = None


def probe(path: Path) -> Probe:
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(out or "{}")
    pr = Probe()
    fmt = data.get("format", {})
    pr.duration_s = float(fmt["duration"]) if fmt.get("duration") else None
    pr.bit_rate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            pr.video_streams += 1
            pr.width, pr.height = int(s.get("width", 0)), int(s.get("height", 0))
            if s.get("nb_frames", "").isdigit():
                pr.nb_frames = int(s["nb_frames"])
        elif s.get("codec_type") == "audio":
            pr.audio_streams += 1
            pr.sample_rate = int(s.get("sample_rate", 0) or 0)
    return pr


@dataclass
class AudioStats:
    mean_db: float | None = None
    max_db: float | None = None

    @property
    def clipped(self) -> bool:
        return self.max_db is not None and self.max_db > AUDIO_MAX_DB_CLIPPED

    @property
    def loud(self) -> bool:
        return self.mean_db is not None and self.mean_db > AUDIO_MEAN_DB_LOUD

    @property
    def silent(self) -> bool:
        return self.mean_db is not None and self.mean_db < AUDIO_MEAN_DB_SILENT


def audio_stats(path: Path) -> AudioStats:
    out = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"])
    st = AudioStats()
    m = re.search(r"mean_volume:\s*(-?[0-9.]+) dB", out)
    if m:
        st.mean_db = float(m.group(1))
    m = re.search(r"max_volume:\s*(-?[0-9.]+) dB", out)
    if m:
        st.max_db = float(m.group(1))
    return st


@dataclass
class FrameStats:
    t: float
    entropy: float | None = None        # Y-plane entropy (bits) from ffmpeg's entropy filter
    bytes_per_pixel: float | None = None  # JPEG at fixed quality, per source pixel of the scaled frame

    @property
    def garbage(self) -> bool:
        bpp = self.bytes_per_pixel
        if bpp is None:
            return False
        if bpp > VIDEO_MAX_BYTES_PER_PIXEL:
            return True
        return self.entropy is not None and self.entropy > VIDEO_HIGH_ENTROPY and bpp > VIDEO_HIGH_ENTROPY_BYTES_PER_PIXEL


def frame_stats(path: Path, t: float) -> FrameStats:
    fs = FrameStats(t=t)
    vf = f"scale={VIDEO_SAMPLE_WIDTH}:-2"
    out = _run(["ffmpeg", "-nostdin", "-v", "info", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                "-vf", f"{vf},entropy=mode=normal,metadata=print", "-f", "null", "-"])
    m = re.search(r"lavfi\.entropy\.entropy\.normal\.Y=([0-9.]+)", out)
    if m:
        fs.entropy = float(m.group(1))
    with tempfile.TemporaryDirectory() as d:
        jpg = Path(d) / "f.jpg"
        _run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
              "-vf", vf, "-q:v", str(VIDEO_JPEG_QUALITY), str(jpg)])
        if jpg.exists():
            dims = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", str(jpg)]).strip().split(",")
            try:
                w, h = int(dims[0]), int(dims[1])
                fs.bytes_per_pixel = jpg.stat().st_size / float(w * h)
            except (ValueError, IndexError):
                pass
    return fs


@dataclass
class Verdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    probe: Probe | None = None
    audio: AudioStats | None = None
    frames: list[FrameStats] = field(default_factory=list)
    bits_per_pixel_frame: float | None = None

    def summary(self) -> str:
        a = self.audio
        fr = ", ".join(f"t{f.t:.0f}:H={f.entropy:.2f}/{(f.bytes_per_pixel or 0):.3f}" for f in self.frames if f.entropy is not None)
        return (f"{'OK' if self.ok else 'BAD'} dur={self.probe.duration_s if self.probe else '?'} "
                f"{self.probe.width if self.probe else '?'}x{self.probe.height if self.probe else '?'} "
                f"audio mean/max={a.mean_db if a else '?'}/{a.max_db if a else '?'} dB bppf={self.bits_per_pixel_frame} frames[{fr}]"
                + (" :: " + "; ".join(self.reasons) if self.reasons else ""))


def judge(path: Path, *, expect_seconds: float | None = None, expect_canvas: tuple[int, int] | None = None,
          sample_times: tuple[float, ...] | None = None, check_video: bool = True, check_audio: bool = True) -> Verdict:
    """Decide whether an H3 output is a valid video+audio clip that matches the request."""
    v = Verdict(ok=True)
    if not have_ffmpeg():
        v.reasons.append("ffmpeg/ffprobe missing: content checks skipped")
        return v
    v.probe = probe(path)
    if v.probe.video_streams < 1 or v.probe.audio_streams < 1:
        v.ok = False
        v.reasons.append(f"streams video={v.probe.video_streams} audio={v.probe.audio_streams} (H3 must emit both)")
        return v
    if expect_seconds is not None and v.probe.duration_s is not None:
        want = expected_duration_s(expect_seconds)
        if abs(v.probe.duration_s - want) > 0.25:
            v.ok = False
            v.reasons.append(f"duration {v.probe.duration_s:.3f} s, expected {want:.3f} s for duration_seconds={expect_seconds}")
    if expect_canvas is not None and (v.probe.width, v.probe.height) != tuple(expect_canvas):
        v.ok = False
        v.reasons.append(f"canvas {v.probe.width}x{v.probe.height}, expected {expect_canvas[0]}x{expect_canvas[1]}")
    if check_audio:
        v.audio = audio_stats(path)
        if v.audio.clipped or v.audio.loud:
            v.ok = False
            v.reasons.append(f"audio looks like noise: mean {v.audio.mean_db} dB max {v.audio.max_db} dB "
                             f"(clean H3 tracks: mean < {AUDIO_MEAN_DB_LOUD}, max < {AUDIO_MAX_DB_CLIPPED})")
        elif v.audio.silent:
            v.ok = False
            v.reasons.append(f"audio is silent (mean {v.audio.mean_db} dB)")
    if check_video:
        if v.probe.bit_rate and v.probe.width and v.probe.height:
            bppf = v.probe.bit_rate / float(v.probe.width * v.probe.height * FPS)
            v.bits_per_pixel_frame = round(bppf, 3)
            if bppf > VIDEO_MAX_BITS_PER_PIXEL_FRAME:
                v.ok = False
                v.reasons.append(f"video bit rate {bppf:.2f} bits/pixel/frame looks like noise (clean H3 clips <= 0.34)")
        dur = v.probe.duration_s or 0.0
        times = sample_times or tuple(sorted({round(min(max(0.0, dur * f), max(0.0, dur - 0.2)), 2) for f in (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)}))
        for t in times:
            fs = frame_stats(path, t)
            v.frames.append(fs)
            if fs.garbage:
                v.ok = False
                v.reasons.append(f"frame at {t:.1f}s looks like noise: entropy {fs.entropy} bits, {fs.bytes_per_pixel} B/px")
    return v
