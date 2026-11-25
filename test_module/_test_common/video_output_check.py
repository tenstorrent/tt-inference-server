# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Is a downloaded video clip usable? Shared by the video benchmark and spec tests."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .video_quality_metrics import (
    FRAME_METRIC_SIZE,
    MissingVideoQualityDependency,
    probe_container,
    probe_video,
    sample_video_frames,
    structural_frame_metrics,
)

logger = logging.getLogger(__name__)

# Wan2.2 serves one shape per mesh (tt-media-server config/constants.py
# WAN22_NUM_FRAMES and wan22_target_resolution; VideoManager encodes at 16 fps):
# 1280x720 on Galaxy-class meshes (>= 32 chips), 832x480 on the rest.
WAN22_NUM_FRAMES = 81
WAN22_FPS = 16
WAN22_LARGE_MESH_DEVICES = frozenset({"GALAXY", "BLACKHOLE_GALAXY"})
WAN22_LARGE_MESH_SIZE = (1280, 720)
WAN22_SMALL_MESH_SIZE = (832, 480)
# A deployment that sets WAN22_RESOLUTION serves that size whatever its mesh.
WAN22_SIZES = {"480p": WAN22_SMALL_MESH_SIZE, "720p": WAN22_LARGE_MESH_SIZE}


def expected_video_shape(
    hf_model_repo: str, device_name: str, resolution: str | None = None
) -> tuple[int, int, int] | None:
    """``(num_frames, width, height)`` the server must return, or None if not pinned.

    ``resolution`` is the deployment's WAN22_RESOLUTION, if it sets one.
    """
    if not hf_model_repo.startswith("Wan-AI/Wan2.2-"):
        return None
    if resolution:
        return (WAN22_NUM_FRAMES, *WAN22_SIZES[resolution.strip().lower()])
    if device_name.upper() in WAN22_LARGE_MESH_DEVICES:
        return (WAN22_NUM_FRAMES, *WAN22_LARGE_MESH_SIZE)
    return (WAN22_NUM_FRAMES, *WAN22_SMALL_MESH_SIZE)


def deployment_resolution(model_spec) -> str | None:
    """The WAN22_RESOLUTION a deployment's model spec sets, if any."""
    return (getattr(model_spec, "env_vars", None) or {}).get("WAN22_RESOLUTION")


def check_video_output(
    video_path: str | Path,
    *,
    hf_model_repo: str,
    device_name: str,
    resolution: str | None = None,
    shape: tuple[int, int] | None = None,
) -> str | None:
    """Return why a downloaded clip is unusable, or None when it is fine.

    A completed job whose mp4 is black, flat, frozen or the wrong shape is a failed
    generation: without this, any file that downloads counts as a success.
    ``shape`` is the ``(width, height)`` the request asked for; without it the
    clip must have the deployment's shape.
    """
    path = Path(video_path)
    try:
        structural = structural_frame_metrics(sample_video_frames(path))
    except (ValueError, MissingVideoQualityDependency) as e:
        return f"clip could not be checked: {e}"

    problems = [
        f"{kind} ({structural[f'{kind}_frame_fraction']:.0%} of sampled frames)"
        for kind in ("black", "flat")
        if structural[f"is_{kind}"]
    ]
    if structural["is_frozen"]:
        problems.append(
            f"frozen ({structural['frozen_pair_fraction']:.0%} of sampled frame "
            "pairs unchanged)"
        )

    expected = (
        (WAN22_NUM_FRAMES, *shape)
        if shape is not None
        else expected_video_shape(hf_model_repo, device_name, resolution)
    )
    if expected is not None:
        num_frames, width, height = expected
        decoded = structural["total_decoded_frames"]
        if decoded != num_frames:
            problems.append(f"{decoded} frames, expected {num_frames}")
        try:
            probe = probe_video(
                path,
                expected_duration=num_frames / WAN22_FPS,
                expected_ratio=f"{width}:{height}",
            )
        except (ValueError, MissingVideoQualityDependency) as e:
            return f"clip could not be probed: {e}"
        if (probe["width"], probe["height"]) != (width, height):
            problems.append(
                f"{probe['width']}x{probe['height']}, expected {width}x{height}"
            )
    return "; ".join(problems) or None


def check_mp4_container(video_path: str | Path, *, audio: bool = False) -> str | None:
    """Return how an MP4 breaks the generation API's video format, or None.

    The API serves H.264 High profile, yuv420p, faststart (the ``moov`` index
    before the data, so playback can start while downloading), with an audio
    track only when the model makes sound.
    """
    try:
        container = probe_container(Path(video_path))
    except (ValueError, MissingVideoQualityDependency) as e:
        return f"container could not be probed: {e}"
    expected = {"codec": "h264", "profile": "High", "pix_fmt": "yuv420p"}
    problems = [
        f"{key} {container[key]!r}, expected {value!r}"
        for key, value in expected.items()
        if container[key] != value
    ]
    if not container["faststart"]:
        problems.append(f"not faststart (top-level boxes {container['boxes']})")
    if audio != bool(container["audio_streams"]):
        problems.append(
            f"{container['audio_streams']} audio stream(s), expected "
            f"{'one' if audio else 'none'}"
        )
    return "; ".join(problems) or None


def mean_frame_difference(path_a: str | Path, path_b: str | Path) -> float:
    """Mean absolute pixel difference (0-255) between two clips' sampled frames.

    Both clips are sampled at the same uniform positions and resized to
    FRAME_METRIC_SIZE, so identical clips score 0.
    """
    import numpy as np

    frames_a = sample_video_frames(Path(path_a)).frames
    frames_b = sample_video_frames(Path(path_b)).frames
    return float(
        np.mean(
            [
                np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean()
                for a, b in zip(frames_a, frames_b)
            ]
        )
    )


def frame_distances_to_image(
    video_path: str | Path, image_bytes: bytes, frame_indices: list[int]
) -> dict[int, float]:
    """Mean absolute pixel difference (0-255) from each chosen frame to an image.

    Frames and image are both resized to FRAME_METRIC_SIZE. Lower means the frame
    looks more like the image: how an I2V test sees where its conditioning landed.
    """
    import io

    import imageio.v3 as iio
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    target = np.asarray(image.resize(FRAME_METRIC_SIZE), np.float32)
    wanted, distances = set(frame_indices), {}
    for index, frame in enumerate(iio.imiter(Path(video_path))):
        if index in wanted:
            resized = Image.fromarray(frame).convert("RGB").resize(FRAME_METRIC_SIZE)
            distances[index] = float(
                np.abs(np.asarray(resized, np.float32) - target).mean()
            )
        if index >= max(wanted):
            break
    return distances


def save_clip(
    video_path: str | Path | None,
    output_path: str | Path | None,
    folder: str,
    name: str | None = None,
) -> Path | None:
    """Copy a clip to ``<output_path>/videos/<folder>/`` so the run keeps it.

    Shield uploads workflow_logs/, which holds the run's output path; clips
    downloaded to /tmp are gone when the job ends. Never raises: losing a copy
    must not fail the test that made the clip.
    """
    if not video_path or not output_path:
        return None
    destination = (
        Path(output_path) / "videos" / folder / (name or Path(video_path).name)
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, destination)
    except OSError as e:
        logger.warning(f"Could not save clip {video_path} to {destination}: {e}")
        return None
    logger.info(f"Saved clip for upload: {destination}")
    return destination


__all__ = [
    "check_mp4_container",
    "check_video_output",
    "save_clip",
    "deployment_resolution",
    "expected_video_shape",
    "frame_distances_to_image",
    "mean_frame_difference",
]
