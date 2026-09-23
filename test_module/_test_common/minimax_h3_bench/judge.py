# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The output contract for one MiniMax-H3 clip.

24 fps; the frame count snaps up to 17n+5, so the served duration is
``expected_frames(d) / 24`` within 0.25 s; the canvas is fixed per aspect ratio
(short edge 768); the container carries a video and an audio stream; the
soundtrack is neither silent nor stuck at the rails (the corruption signature:
9-100 % of samples in volumedetect's 0 dB bin on corrupted tracks, <= 0.03 % on
clean ones -- 5 % sits in the gap).
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from . import models as M

FPS = 24
FRAMES_PER_CHUNK, FRAMES_CHUNK_OFFSET = 17, 5
CANVAS = {
    "16:9": (1344, 768),
    "21:9": (1536, 672),
    "4:3": (1024, 768),
    "1:1": (768, 768),
    "3:4": (768, 1024),
    "9:16": (768, 1344),
}
DURATION_TOL_S = 0.25
AUDIO_RAIL_SHARE = 0.05
AUDIO_MEAN_DB_SILENT = -80.0
MIN_CLIP_BYTES = 10_000


def expected_frames(seconds: float) -> int:
    """Frames served: round(seconds*24), then snap UP to the next 17n+5."""
    n = int(round(seconds * FPS))
    while (n - FRAMES_CHUNK_OFFSET) % FRAMES_PER_CHUNK:
        n += 1
    return n


def expected_seconds(seconds: float) -> float:
    return expected_frames(seconds) / FPS


def ffprobe_json(path: str) -> dict | None:
    """Stream/format metadata as ffprobe reports it; ``ffmpeg -i`` parsed to the same
    shape when only ffmpeg is installed (imageio-ffmpeg ships no ffprobe); None when
    neither is available."""
    probe = M.ffprobe_binary()
    if probe:
        out = subprocess.run(
            [probe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        try:
            return json.loads(out.stdout or "{}")
        except ValueError:
            return {}
    ffmpeg = M.ffmpeg_binary()
    if not ffmpeg:
        return None
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = out.stderr
    streams = []
    for m in re.finditer(
        r"Stream #\d+:\d+(?:\[[^\]]*\])?(?:\([^)]*\))?: (Video|Audio): ([^\n]*)", text
    ):
        kind, rest = m.group(1).lower(), m.group(2)
        stream = {"codec_type": kind, "codec_name": rest.split(",")[0].split()[0]}
        if kind == "video":
            size = re.search(r"\b(\d{2,5})x(\d{2,5})\b", rest)
            fps = re.search(r"([\d.]+) fps", rest)
            if size:
                stream["width"], stream["height"] = (
                    int(size.group(1)),
                    int(size.group(2)),
                )
            if fps:
                stream["r_frame_rate"] = f"{fps.group(1)}/1"
        streams.append(stream)
    dur = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    fmt = {}
    if dur:
        fmt["duration"] = str(
            int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))
        )
    return {"streams": streams, "format": fmt}


def audio_stats(path: str):
    """(mean_dB, max_dB, rail_share) from ffmpeg's volumedetect, or (None,)*3."""
    ffmpeg = M.ffmpeg_binary()
    if not ffmpeg:
        return None, None, None
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path, "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    text = out.stderr
    mean = re.search(r"mean_volume: (-?[\d.]+) dB", text)
    peak = re.search(r"max_volume: (-?[\d.]+) dB", text)
    total = sum(int(n) for n in re.findall(r"histogram_-?\d+db: (\d+)", text))
    at_0db = re.search(r"histogram_0db: (\d+)", text)
    if not mean or not peak:
        return None, None, None
    share = (int(at_0db.group(1)) / total) if (at_0db and total) else 0.0
    return float(mean.group(1)), float(peak.group(1)), share


def judge(path: str, seconds: float, aspect: str = "16:9"):
    """(problems, notes) for one clip. Any problem fails the clip."""
    problems, notes = [], []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError as exc:
        return [f"clip unreadable: {exc}"], notes
    if b"ftyp" not in head:
        return ["not an mp4 container"], notes
    if size < MIN_CLIP_BYTES:
        problems.append(f"clip is only {size} bytes")
    want = expected_seconds(seconds)
    meta = ffprobe_json(path)
    if meta is None:
        served = M.mvhd_duration(path)
        if served is None or abs(served - want) > DURATION_TOL_S:
            problems.append(f"duration {served}s != {want:.3f}s (requested {seconds}s)")
        notes.append("ffmpeg/ffprobe missing: streams and audio not checked")
        return problems, notes
    streams = meta.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not video:
        problems.append("no video stream")
    if not audio:
        problems.append("no audio stream (H3 emits video + audio)")
    if video:
        width, height = video[0].get("width"), video[0].get("height")
        if (width, height) != CANVAS.get(aspect):
            problems.append(
                f"canvas {width}x{height} != {CANVAS.get(aspect)} for {aspect}"
            )
        num, _, den = (video[0].get("r_frame_rate") or "0/1").partition("/")
        try:
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        if abs(fps - FPS) > 0.01:
            problems.append(f"fps {fps:g} != {FPS}")
        frames = int(video[0].get("nb_frames") or 0)
        if frames and expected_frames(seconds) - frames == 1:
            notes.append(f"short by one frame ({frames} of {expected_frames(seconds)})")
        if M.ffprobe_binary() and not video[0].get("color_primaries"):
            notes.append("colour untagged")
    try:
        served = float(meta.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        served = 0.0
    if not served:
        served = M.mvhd_duration(path) or 0.0
    if abs(served - want) > DURATION_TOL_S:
        problems.append(f"duration {served:.3f}s != {want:.3f}s (requested {seconds}s)")
    if audio:
        mean_db, max_db, rails = audio_stats(path)
        if mean_db is None:
            notes.append("audio not measured (volumedetect output unparsed)")
        elif rails > AUDIO_RAIL_SHARE:
            problems.append(
                f"audio at the rails: {rails:.1%} of samples at 0 dB, mean {mean_db} dB -- the corruption signature"
            )
        elif mean_db < AUDIO_MEAN_DB_SILENT:
            problems.append(f"audio is silent (mean {mean_db} dB)")
        elif rails > 0.005 or max_db > -1.0:
            notes.append(
                f"loud, {rails:.2%} at the rails (mean {mean_db} dB, max {max_db} dB)"
            )
    return problems, notes


def run_problems(rec: dict, case: dict):
    """Why an ``ok`` row should not be trusted: the clip itself, or a generation time
    the poll interval could not resolve (0.0 s is not a measurement)."""
    if rec.get("out_file") and os.path.exists(rec["out_file"]):
        problems, notes = judge(
            rec["out_file"], case["duration_s"], case.get("aspect_ratio", "16:9")
        )
    else:
        problems, notes = [], ["clip no longer on disk: judged from the row"]
        want = expected_seconds(case["duration_s"])
        served = rec.get("mvhd_duration_s")
        if served is None or abs(served - want) > DURATION_TOL_S:
            problems.append(f"duration {served}s != {want:.3f}s")
        if rec.get("has_audio") is False:
            problems.append("no audio stream")
    if rec.get("api_exposes_progress") and rec.get("gen_s") == 0:
        problems.append(
            "gen_s is 0.0: queued -> completed inside one poll interval; not a measurement"
        )
    return problems, notes
