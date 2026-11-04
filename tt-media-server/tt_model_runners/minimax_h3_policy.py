# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""MiniMax-H3 serving policy for this deployment.

The served envelope -- aspect ratios, durations, reference counts -- and the admission check over
it are owned by the model's own policy module (``models.tt_dit.pipelines.minimax_h3.policy``) and
re-exported here, so what the media-server accepts cannot drift from what the pipeline is warmed
for. Only the request-transport concerns that pull in ``av`` -- media probing and reference-clip
windows -- live here.

Per-clip and combined duration windows are the product contract and are enforced here; the
pipeline truncates instead of refusing.
"""

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.tt_dit.pipelines.minimax_h3.policy import (
        MINIMAX_H3_ASPECT_RATIOS,
        MINIMAX_H3_DEFAULT_ASPECT_RATIO,
        MINIMAX_H3_DEFAULT_DURATION_S,
        MINIMAX_H3_DURATIONS_S,
        MINIMAX_H3_KEYFRAME_MAX_SIDE,
        MINIMAX_H3_KEYFRAME_MIN_SIDE,
        MiniMaxH3ReferenceInfo,
        admit_request,
        minimax_h3_frames_are_aligned,
        minimax_h3_parse_aspect_ratio,
    )
    from models.tt_dit.pipelines.minimax_h3.packing_ref2va import check_reference_kinds

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
        "MINIMAX_H3_KEYFRAME_MAX_SIDE",
        "MINIMAX_H3_KEYFRAME_MIN_SIDE",
        "MiniMaxH3ReferenceInfo",
        "admit_request",
        "minimax_h3_parse_aspect_ratio",
        "minimax_h3_frames_are_aligned",
    }
)


def __getattr__(name: str):
    if name in _METAL_REEXPORTS:
        from models.tt_dit.pipelines.minimax_h3 import policy as _metal_policy

        return getattr(_metal_policy, name)
    if name == "check_reference_kinds":
        from models.tt_dit.pipelines.minimax_h3.packing_ref2va import (
            check_reference_kinds,
        )

        return check_reference_kinds
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MINIMAX_H3_ASPECT_RATIOS",
    "MINIMAX_H3_DEFAULT_ASPECT_RATIO",
    "MINIMAX_H3_DEFAULT_DURATION_S",
    "MINIMAX_H3_DURATIONS_S",
    "MINIMAX_H3_KEYFRAME_MAX_SIDE",
    "MINIMAX_H3_KEYFRAME_MIN_SIDE",
    "MINIMAX_H3_NUM_INFERENCE_STEPS",
    "MINIMAX_H3_REF_CLIP_MIN_S",
    "MINIMAX_H3_REF_CLIP_MAX_S",
    "MINIMAX_H3_REF_COMBINED_MAX_S",
    "MINIMAX_H3_FL2VA_FRAME_POS",
    "MiniMaxH3ReferenceInfo",
    "ReferenceMediaProbe",
    "admit_request",
    "minimax_h3_parse_aspect_ratio",
    "minimax_h3_frames_are_aligned",
    "probe_reference_media",
    "check_reference_clip_durations",
    "check_reference_kinds",
]

# t2va / fl2va / ref2va AdaLN table is warmed at this count; not a request lever.
# Copied here so admission-time echo does not import metal/ttnn.
MINIMAX_H3_NUM_INFERENCE_STEPS = 50

# Ref2VA reference clip windows: the product card (the pipeline does not enforce them).
MINIMAX_H3_REF_CLIP_MIN_S = 2.0
MINIMAX_H3_REF_CLIP_MAX_S = 15.0
MINIMAX_H3_REF_COMBINED_MAX_S = 15.0

# FL2VA keyframe sentinels on ``image_prompts[].frame_pos``.
MINIMAX_H3_FL2VA_FRAME_POS = frozenset({0, -1})


@dataclass(frozen=True)
class ReferenceMediaProbe:
    """Container metadata of one reference clip.

    ``size`` (width, height) and ``fps`` are those of the first video stream, ``audio_channels``
    that of the first audio stream; each is None when the stream is absent.
    """

    duration_s: float
    size: tuple[int, int] | None
    fps: float | None
    audio_channels: int | None


def probe_reference_media(raw: bytes) -> ReferenceMediaProbe:
    """Probe encoded audio/video bytes without decoding them.

    Uses PyAV's container duration (microseconds) and the frame rate the pipeline's decoder reads
    (average, else guessed). Raises ``ValueError`` when the bytes are not a readable audio/video
    container or have no duration.
    """
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            if container.duration is None or container.duration <= 0:
                raise ValueError("media container reports no duration")
            size = fps = audio_channels = None
            if container.streams.video:
                stream = container.streams.video[0]
                size = (stream.codec_context.width, stream.codec_context.height)
                rate = stream.average_rate or stream.guessed_rate
                fps = float(rate) if rate else None
            if container.streams.audio:
                audio_channels = len(container.streams.audio[0].layout.channels)
            return ReferenceMediaProbe(
                duration_s=container.duration / 1_000_000,
                size=size,
                fps=fps,
                audio_channels=audio_channels,
            )
    except av.FFmpegError as exc:  # av.AVError was removed in PyAV 14
        raise ValueError("media could not be probed") from exc


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
