# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Reusable structural and CLIP metrics for short generated videos."""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_COUNT = 8
FRAME_METRIC_SIZE = (224, 224)
BLACK_BRIGHTNESS_THRESHOLD = 2.0
FLAT_VARIANCE_THRESHOLD = 4.0
FROZEN_DELTA_THRESHOLD = 0.5
INVALID_FRAME_FRACTION = 0.8


class MissingVideoQualityDependency(RuntimeError):
    """Raised when optional quality-evaluation dependencies are unavailable."""


@dataclass(frozen=True)
class SampledVideoFrames:
    """Uniformly sampled RGB frames and source decode counts."""

    frames: tuple[Any, ...]
    total_decoded_frames: int
    sampled_indices: tuple[int, ...]


def probe_video(
    video_path: Path,
    *,
    expected_duration: float,
    expected_ratio: str,
    duration_tolerance: float = 0.5,
    ratio_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Verify downloaded video metadata with ffprobe or imageio-ffmpeg."""

    ffprobe = shutil.which("ffprobe")
    try:
        if ffprobe is not None:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = json.loads(result.stdout)
            video_stream = next(
                stream
                for stream in metadata["streams"]
                if stream.get("codec_type") == "video"
            )
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            duration = float(metadata["format"]["duration"])
        else:
            try:
                import imageio_ffmpeg  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise MissingVideoQualityDependency(
                    "ffprobe or imageio-ffmpeg is required for video metadata"
                ) from exc

            reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
            try:
                metadata = next(reader)
            finally:
                reader.close()
            width, height = (int(value) for value in metadata["size"])
            duration = float(metadata["duration"])
            fps = metadata.get("fps")
            video_stream = {
                "codec_name": metadata.get("codec"),
                "r_frame_rate": str(fps) if fps is not None else None,
                "nb_frames": (
                    round(duration * float(fps)) if fps is not None else None
                ),
            }
    except (
        KeyError,
        TypeError,
        ValueError,
        StopIteration,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise ValueError(f"video metadata could not be decoded: {exc}") from exc

    ratio_parts = expected_ratio.split(":", 1)
    if len(ratio_parts) != 2:
        raise ValueError(f"invalid expected ratio {expected_ratio!r}")
    expected_ratio_value = float(ratio_parts[0]) / float(ratio_parts[1])
    actual_ratio = width / height if height else 0.0
    ratio_error = abs(actual_ratio - expected_ratio_value) / expected_ratio_value

    return {
        "valid": (
            width > 0
            and height > 0
            and abs(duration - expected_duration) <= duration_tolerance
            and ratio_error <= ratio_tolerance
        ),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "duration_error_seconds": abs(duration - expected_duration),
        "aspect_ratio": actual_ratio,
        "aspect_ratio_error": ratio_error,
        "codec": video_stream.get("codec_name"),
        "frame_rate": video_stream.get("r_frame_rate"),
        "frame_count": video_stream.get("nb_frames"),
    }


# One stream line of ``ffmpeg -i``, e.g. "Stream #0:0[0x1](und): Video: h264
# (High) (avc1 / 0x31637661), yuv420p(progressive), 832x480, ...": its kind,
# codec, profile (the first parenthesis) and, for video, pixel format.
_FFMPEG_STREAM = re.compile(
    r"Stream #\S+: (?P<kind>Video|Audio): (?P<codec>\w+)"
    r"(?: \((?P<profile>[^)]*)\))?(?: \([^)]*\))*(?:, (?P<pix_fmt>\w+))?"
)


def parse_ffmpeg_streams(stderr: str) -> list[dict[str, Any]]:
    """The streams ``ffmpeg -i`` describes, as ffprobe would name their fields."""
    return [
        {
            "codec_type": match["kind"].lower(),
            "codec_name": match["codec"],
            "profile": match["profile"],
            "pix_fmt": match["pix_fmt"] if match["kind"] == "Video" else None,
        }
        for match in _FFMPEG_STREAM.finditer(stderr)
    ]


def mp4_top_level_boxes(video_path: Path) -> list[str]:
    """The MP4's top-level box types in file order (``ftyp``, ``moov``, ``mdat``...)."""
    boxes = []
    with open(video_path, "rb") as f:
        while len(header := f.read(8)) == 8:
            size, kind = struct.unpack(">I4s", header)
            boxes.append(kind.decode("latin-1"))
            if size == 0:  # runs to the end of the file
                break
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0] - 8
            if size < 8:
                raise ValueError(f"MP4 box {kind!r} has size {size}")
            f.seek(size - 8, 1)
    return boxes


def probe_container(video_path: Path) -> dict[str, Any]:
    """Codec, profile, pixel format and audio of a video file, and its box order.

    Uses ffprobe when it is installed, else the ffmpeg imageio-ffmpeg ships.
    ``faststart`` is whether the ``moov`` index comes before the ``mdat`` data,
    so a player can start before the whole file has downloaded.
    """
    ffprobe = shutil.which("ffprobe")
    try:
        if ffprobe is not None:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,codec_name,profile,pix_fmt",
                    "-of",
                    "json",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(result.stdout)["streams"]
        else:
            try:
                import imageio_ffmpeg  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise MissingVideoQualityDependency(
                    "ffprobe or imageio-ffmpeg is required for container metadata"
                ) from exc
            # With no output file ffmpeg exits 1 after describing the input.
            result = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-i",
                    str(video_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            streams = parse_ffmpeg_streams(result.stderr)
        boxes = mp4_top_level_boxes(video_path)
    except (
        KeyError,
        OSError,
        struct.error,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        raise ValueError(f"container metadata could not be read: {exc}") from exc

    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        raise ValueError("no video stream")
    video = videos[0]
    return {
        "codec": video.get("codec_name"),
        "profile": video.get("profile"),
        "pix_fmt": video.get("pix_fmt"),
        "audio_streams": sum(s.get("codec_type") == "audio" for s in streams),
        "boxes": boxes,
        "faststart": "moov" in boxes
        and "mdat" in boxes
        and boxes.index("moov") < boxes.index("mdat"),
    }


def sample_video_frames(
    video_path: Path,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> SampledVideoFrames:
    """Decode a short video and uniformly select resized RGB frames."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    try:
        import imageio.v3 as iio  # pyright: ignore[reportMissingImports]
        from PIL import Image  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise MissingVideoQualityDependency(
            "frame decoding requires imageio, imageio-ffmpeg, and Pillow"
        ) from exc

    decoded = []
    try:
        for frame in iio.imiter(video_path):
            decoded.append(
                Image.fromarray(frame).convert("RGB").resize(FRAME_METRIC_SIZE)
            )
    except Exception as exc:  # noqa: BLE001 - decoder backends expose many errors
        raise ValueError(f"video frame decoding failed: {exc}") from exc

    total = len(decoded)
    if total < 2:
        raise ValueError(f"video contains only {total} decodable frame(s)")

    selected_count = min(sample_count, total)
    indices = tuple(
        round(index * (total - 1) / (selected_count - 1))
        for index in range(selected_count)
    )
    return SampledVideoFrames(
        frames=tuple(decoded[index] for index in indices),
        total_decoded_frames=total,
        sampled_indices=indices,
    )


def structural_frame_metrics(sampled: SampledVideoFrames) -> dict[str, Any]:
    """Calculate brightness, flatness, frozen-frame, and motion diagnostics."""

    try:
        import numpy as np  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise MissingVideoQualityDependency(
            "structural video metrics require numpy"
        ) from exc

    arrays = [np.asarray(frame, dtype=np.float32) for frame in sampled.frames]
    brightness = [float(array.mean()) for array in arrays]
    variances = [float(array.var()) for array in arrays]
    deltas = [
        float(np.abs(right - left).mean()) for left, right in zip(arrays, arrays[1:])
    ]

    black_fraction = sum(
        value < BLACK_BRIGHTNESS_THRESHOLD for value in brightness
    ) / len(brightness)
    flat_fraction = sum(value < FLAT_VARIANCE_THRESHOLD for value in variances) / len(
        variances
    )
    frozen_fraction = (
        sum(value < FROZEN_DELTA_THRESHOLD for value in deltas) / len(deltas)
        if deltas
        else 1.0
    )
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    median_delta = float(np.median(deltas)) if deltas else 0.0
    p90_delta = float(np.percentile(deltas, 90)) if deltas else 0.0
    delta_stddev = float(np.std(deltas)) if deltas else 0.0
    motion_coefficient_of_variation = (
        delta_stddev / mean_delta if mean_delta > 0 else 0.0
    )
    sudden_change_threshold = max(10.0, median_delta * 4)
    sudden_change_count = sum(value > sudden_change_threshold for value in deltas)

    return {
        "sampled_frame_count": len(arrays),
        "total_decoded_frames": sampled.total_decoded_frames,
        "sampled_indices": list(sampled.sampled_indices),
        "average_brightness": float(np.mean(brightness)),
        "minimum_brightness": min(brightness),
        "average_pixel_variance": float(np.mean(variances)),
        "black_frame_fraction": black_fraction,
        "flat_frame_fraction": flat_fraction,
        "mean_frame_delta": mean_delta,
        "median_frame_delta": median_delta,
        "p90_frame_delta": p90_delta,
        "frame_delta_stddev": delta_stddev,
        "motion_coefficient_of_variation": motion_coefficient_of_variation,
        "sudden_change_count": sudden_change_count,
        "frozen_pair_fraction": frozen_fraction,
        "is_black": black_fraction >= INVALID_FRAME_FRACTION,
        "is_flat": flat_fraction >= INVALID_FRAME_FRACTION,
        "is_frozen": frozen_fraction >= INVALID_FRAME_FRACTION,
    }


__all__ = [
    "MissingVideoQualityDependency",
    "SampledVideoFrames",
    "mp4_top_level_boxes",
    "parse_ffmpeg_streams",
    "probe_container",
    "probe_video",
    "sample_video_frames",
    "structural_frame_metrics",
]
