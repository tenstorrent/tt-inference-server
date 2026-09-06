# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Media validity checks for MiniMax-H3 outputs (ffmpeg/ffprobe based, no model in the loop).

Why these exist: the H3 pipeline can complete a job with a well-formed mp4 whose CONTENT is garbage --
2026-09-05 on OM Quad1 the audio track came back as full-scale noise (max volume 0.0 dB) on one in
four t2va requests while the video was byte-identical to a clean run, and an older metal tree produced
tiled-noise video frames. ``ffprobe`` stream checks ("has video + audio") pass on all of those, so the
live tests judge the decoded samples themselves:

* audio: ``volumedetect`` mean/max plus its 0 dB histogram bin.  The corrupted tracks are not white
  noise: they are waveforms stuck at the rails (spectrally tonal, 9-100 % of the samples at full scale,
  mean -0.3..-10 dB, max 0.0 dB; 12 samples 2026-09-05/06).  Legitimate soundtracks can be LOUD -- a
  3:4 5 s drone measured mean -13..-15 dB with peaks at -0.8..-1.3 dB on both the old and the new
  metal tree -- but never sit at the rails (<= 0.03 % of samples at 0 dB; clean tracks 0 %).  So a
  track is *railed* (fails) when more than ``AUDIO_RAIL_SHARE`` (1 %) of its samples are in the 0 dB
  bin or its mean is above ``AUDIO_MEAN_DB_NOISE`` (-8 dB); *clipped* (max > ``AUDIO_MAX_DB_CLIPPED``)
  and *loud* (mean > ``AUDIO_MEAN_DB_LOUD``) are recorded as notes only.  Pure silence (mean < -80)
  is reported too.
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
  frames at 24 fps -- ``expected_frames`` / ``expected_duration_s`` reproduce the server's math);
  the video stream must be 24 fps and the audio stream within 0.35 s of the video (the audio grid is
  40 Hz and the mux uses ``-shortest``, so the two differ by at most one frame on a healthy clip).
  Codecs, pixel format, sample rate and channel count are recorded in the probe, not asserted.

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

AUDIO_MAX_DB_CLIPPED = -1.0             # note only: legit drones peak at -0.8 dB
AUDIO_MEAN_DB_LOUD = -15.0              # note only
AUDIO_MEAN_DB_NOISE = -8.0              # fails: no legitimate track measured above -13 dB; corrupted -0.3..-4 dB
AUDIO_RAIL_SHARE = 0.01                 # fails: share of samples in volumedetect's 0 dB bin; corrupted 9-100 %, legit <= 0.03 %
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


AUDIO_VIDEO_DRIFT_S = 0.35   # the audio grid is 40 Hz and the mux uses -shortest: <= 1 video frame apart


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
    fps: float | None = None
    video_codec: str | None = None
    pix_fmt: str | None = None
    audio_codec: str | None = None
    channels: int | None = None
    video_duration_s: float | None = None
    audio_duration_s: float | None = None
    color_space: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_range: str | None = None


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
            if str(s.get("nb_frames", "")).isdigit():
                pr.nb_frames = int(s["nb_frames"])
            pr.video_codec, pr.pix_fmt = s.get("codec_name"), s.get("pix_fmt")
            pr.color_space, pr.color_primaries = s.get("color_space"), s.get("color_primaries")
            pr.color_transfer, pr.color_range = s.get("color_transfer"), s.get("color_range")
            rate = s.get("r_frame_rate") or s.get("avg_frame_rate") or ""
            if "/" in rate:
                num, den = rate.split("/")
                pr.fps = float(num) / float(den) if float(den) else None
            if s.get("duration"):
                pr.video_duration_s = float(s["duration"])
        elif s.get("codec_type") == "audio":
            pr.audio_streams += 1
            pr.sample_rate = int(s.get("sample_rate", 0) or 0)
            pr.audio_codec = s.get("codec_name")
            pr.channels = int(s.get("channels", 0) or 0)
            if s.get("duration"):
                pr.audio_duration_s = float(s["duration"])
    return pr


@dataclass
class AudioStats:
    mean_db: float | None = None
    max_db: float | None = None
    n_samples: int | None = None
    at_full_scale: int | None = None    # volumedetect histogram_0db: samples within 0.5 dB of full scale

    @property
    def rail_share(self) -> float | None:
        if not self.n_samples or self.at_full_scale is None:
            return None
        return self.at_full_scale / self.n_samples

    @property
    def railed(self) -> bool:
        """The corruption signature: the waveform sits at the rails, or the mean is where no real track is."""
        share = self.rail_share
        return (share is not None and share > AUDIO_RAIL_SHARE) or (
            self.mean_db is not None and self.mean_db > AUDIO_MEAN_DB_NOISE)

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
    m = re.search(r"n_samples:\s*(\d+)", out)
    if m:
        st.n_samples = int(m.group(1))
    m = re.search(r"histogram_0db:\s*(\d+)", out)
    st.at_full_scale = int(m.group(1)) if m else (0 if st.n_samples else None)
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
    notes: list[str] = field(default_factory=list)     # observations that do not fail the verdict (loud audio, ...)
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
                + (" :: " + "; ".join(self.reasons) if self.reasons else "")
                + (" (note: " + "; ".join(self.notes) + ")" if self.notes else ""))


def frames_dropped(pr: Probe, expect_seconds: float) -> int | None:
    """How many of the 17n+5 frames the container is short of (the ``-shortest`` mux drops one at 4/6/15 s)."""
    if pr.nb_frames is None:
        return None
    return expected_frames(expect_seconds) - pr.nb_frames


def keyframe_ssim(path: Path, image_b64: str, t: float) -> float | None:
    """SSIM between the frame at ``t`` and a keyframe image (base64), both scaled to 640x360 -- the
    fl2va conditioning oracle: the first output frame must resemble the FIRST keyframe more than the last."""
    import base64
    with tempfile.TemporaryDirectory() as d:
        key = Path(d) / "key.img"
        key.write_bytes(base64.b64decode(image_b64))
        frame = Path(d) / "frame.png"
        _run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", str(frame)])
        if not frame.exists():
            return None
        out = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(frame), "-i", str(key),
                    "-lavfi", "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]ssim", "-f", "null", "-"])
    m = re.findall(r"All:([0-9.]+)", out)
    return float(m[-1]) if m else None


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
    if v.probe.fps is not None and abs(v.probe.fps - FPS) > 0.01:
        v.ok = False
        v.reasons.append(f"frame rate {v.probe.fps:.3f}, H3 serves {FPS} fps")
    if v.probe.video_duration_s and v.probe.audio_duration_s and abs(v.probe.video_duration_s - v.probe.audio_duration_s) > AUDIO_VIDEO_DRIFT_S:
        v.ok = False
        v.reasons.append(f"audio {v.probe.audio_duration_s:.2f} s vs video {v.probe.video_duration_s:.2f} s: streams out of step "
                         f"(a truncated or missing soundtrack)")
    if check_audio:
        v.audio = audio_stats(path)
        if v.audio.railed:
            v.ok = False
            share = v.audio.rail_share
            v.reasons.append(f"audio looks like noise: mean {v.audio.mean_db} dB max {v.audio.max_db} dB, "
                             f"{(share or 0) * 100:.1f}% of samples at full scale "
                             f"(corrupted tracks sit at the rails: > {AUDIO_RAIL_SHARE * 100:.0f}% or mean > {AUDIO_MEAN_DB_NOISE} dB)")
        elif v.audio.clipped or v.audio.loud:
            v.notes.append(f"audio is loud but not railed: mean {v.audio.mean_db} dB max {v.audio.max_db} dB "
                           f"({(v.audio.rail_share or 0) * 100:.2f}% at full scale)")
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
