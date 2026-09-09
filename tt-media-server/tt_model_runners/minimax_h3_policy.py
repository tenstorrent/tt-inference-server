# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""MiniMax-H3 serving policy for this deployment.

The served envelope -- aspect ratios, durations, step count -- and its validators are owned by the
model's own policy module (``models.tt_dit.pipelines.minimax_h3.policy``) and re-exported here, so
what the media-server accepts cannot drift from what the pipeline is warmed for. Only the
request-transport concerns that pull in ``av`` -- media probing and reference-clip windows -- live
here.

Ref2VA reference *counts* match ``packing_ref2va`` (9 / 3 / 3). Per-clip and combined duration
windows are the product contract and are enforced here; the pipeline truncates instead of refusing.
"""

import io

# The serving envelope and its validators are owned by the model's policy module and re-exported
# lazily: pulling them at import time would drag ``ttnn`` into every consumer, and the server's unit
# tests mock the whole ``models.tt_dit`` tree. ``__getattr__`` defers the metal import to first
# access, so this module stays importable (and testable) without metal present.
_METAL_REEXPORTS = frozenset(
    {
        "MINIMAX_H3_ASPECT_RATIOS",
        "MINIMAX_H3_DEFAULT_ASPECT_RATIO",
        "MINIMAX_H3_DEFAULT_DURATION_S",
        "MINIMAX_H3_DURATIONS_S",
        "MINIMAX_H3_NUM_INFERENCE_STEPS",
        "minimax_h3_parse_aspect_ratio",
        "minimax_h3_frames_are_aligned",
    }
)


def __getattr__(name: str):
    if name in _METAL_REEXPORTS:
        from models.tt_dit.pipelines.minimax_h3 import policy as _metal_policy

        return getattr(_metal_policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MINIMAX_H3_ASPECT_RATIOS",
    "MINIMAX_H3_DEFAULT_ASPECT_RATIO",
    "MINIMAX_H3_DEFAULT_DURATION_S",
    "MINIMAX_H3_DURATIONS_S",
    "MINIMAX_H3_NUM_INFERENCE_STEPS",
    "MINIMAX_H3_MAX_REFERENCE_IMAGES",
    "MINIMAX_H3_MAX_REFERENCE_VIDEOS",
    "MINIMAX_H3_MAX_REFERENCE_AUDIOS",
    "MINIMAX_H3_REF_CLIP_MIN_S",
    "MINIMAX_H3_REF_CLIP_MAX_S",
    "MINIMAX_H3_REF_COMBINED_MAX_S",
    "MINIMAX_H3_FL2VA_FRAME_POS",
    "minimax_h3_parse_aspect_ratio",
    "minimax_h3_frames_are_aligned",
    "probe_media_duration_seconds",
    "check_reference_clip_durations",
]

# Ref2VA omni-reference limits. Counts match packing_ref2va; duration windows
# are the product card (the pipeline does not enforce them).
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_REF_CLIP_MIN_S = 2.0
MINIMAX_H3_REF_CLIP_MAX_S = 15.0
MINIMAX_H3_REF_COMBINED_MAX_S = 15.0

# FL2VA keyframe sentinels on ``image_prompts[].frame_pos``.
MINIMAX_H3_FL2VA_FRAME_POS = frozenset({0, -1})


def probe_media_duration_seconds(raw: bytes) -> float:
    """Container duration in seconds, from encoded bytes.

    Uses PyAV's container duration (microseconds). Raises ``ValueError`` when
    the bytes are not a readable audio/video container or have no duration.
    """
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            if container.duration is None or container.duration <= 0:
                raise ValueError("media container reports no duration")
            return container.duration / 1_000_000
    except av.FFmpegError as exc:  # av.AVError was removed in PyAV 14
        raise ValueError("media could not be probed for duration") from exc


def check_reference_clip_durations(
    *,
    video_durations: list[float],
    audio_durations: list[float],
) -> None:
    """Refuse reference clips outside the 2–15 s / combined ≤ 15 s window."""

    def _check(kind: str, durations: list[float]) -> None:
        for index, seconds in enumerate(durations):
            if not MINIMAX_H3_REF_CLIP_MIN_S <= seconds <= MINIMAX_H3_REF_CLIP_MAX_S:
                raise ValueError(
                    f"{kind}[{index}] is {seconds:g} s; each {kind} clip must be "
                    f"{MINIMAX_H3_REF_CLIP_MIN_S:g}–{MINIMAX_H3_REF_CLIP_MAX_S:g} s"
                )
        combined = sum(durations)
        if combined > MINIMAX_H3_REF_COMBINED_MAX_S:
            raise ValueError(
                f"combined {kind} duration is {combined:g} s; must be "
                f"≤ {MINIMAX_H3_REF_COMBINED_MAX_S:g} s"
            )

    _check("videos", video_durations)
    _check("audios", audio_durations)
