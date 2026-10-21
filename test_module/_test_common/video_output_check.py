# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Is a downloaded video clip usable? Shared by the video benchmark and spec tests."""

from __future__ import annotations

from pathlib import Path

from .video_quality_metrics import (
    MissingVideoQualityDependency,
    probe_video,
    sample_video_frames,
    structural_frame_metrics,
)

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
) -> str | None:
    """Return why a downloaded clip is unusable, or None when it is fine.

    A completed job whose mp4 is black, flat, frozen or the wrong shape is a failed
    generation: without this, any file that downloads counts as a success.
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

    expected = expected_video_shape(hf_model_repo, device_name, resolution)
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


__all__ = ["check_video_output", "deployment_resolution", "expected_video_shape"]
