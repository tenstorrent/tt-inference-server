# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""MiniMax-H3 serving policy for this deployment.

Which aspect ratios, durations, and step counts the media-server will accept.
Canvas size and the video VAE's ``17n + 5`` frame rule live in the model's
packing module; copying either here is how they drift. FPS is the same: read
``MINIMAX_H3_FPS`` from packing at the point of use.

Ref2VA reference *counts* match ``packing_ref2va`` (9 / 3 / 3). Per-clip and
combined duration windows are the product contract and are enforced here;
the pipeline truncates instead of refusing.
"""

import io

# 4x8 Blackhole Galaxy only. Every published 768P working point is servable: six
# aspect ratios x twelve durations. The model accepts 1:4..4:1, but only these
# are calibrated, and each is a distinct canvas the pipeline must be warmed at.
MINIMAX_H3_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
MINIMAX_H3_DEFAULT_ASPECT_RATIO = (16, 9)

# Durations in seconds: every integer the MiniMax API accepts. The frame count
# is not free -- the video VAE encodes in 17-frame chunks, so only ``17n + 5``
# counts exist and ``align_num_frames`` rounds a request UP to the next one.
# The served clip is therefore >= the requested duration, by up to 0.67 s
# (13 s -> 13.667 s); 8 s is the only exact fit (192 frames). Rounding up
# rather than down is deliberate: a caller asking for 10 s should never be
# handed 9.4 s.
MINIMAX_H3_DURATIONS_S = tuple(range(4, 16))
MINIMAX_H3_DEFAULT_DURATION_S = 5

# Not a request lever: the AdaLN modulation table is precomputed per step count
# and every shape is warmed at 50.
MINIMAX_H3_NUM_INFERENCE_STEPS = 50

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


def minimax_h3_parse_aspect_ratio(value: str) -> tuple[int, int]:
    """``"16:9"`` -> ``(16, 9)``, restricted to the published set.

    Rejects rather than rounds: a caller asking for 2:1 wants 2:1, and quietly
    serving 16:9 would be a wrong answer dressed as a right one.
    """
    text = str(value).strip().replace("x", ":").replace("/", ":")
    parts = text.split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(
            f"aspect_ratio must look like 'W:H' (got {value!r}); supported: "
            + ", ".join(f"{w}:{h}" for w, h in MINIMAX_H3_ASPECT_RATIOS)
        )
    pair = (int(parts[0]), int(parts[1]))
    if pair not in MINIMAX_H3_ASPECT_RATIOS:
        raise ValueError(
            f"aspect_ratio {pair[0]}:{pair[1]} is not served; supported: "
            + ", ".join(f"{w}:{h}" for w, h in MINIMAX_H3_ASPECT_RATIOS)
        )
    return pair


def minimax_h3_frames_are_aligned(num_frames: int) -> bool:
    """``num_frames`` must be ``17n + 5``: 124, 243, 362, ...

    The modulus lives in packing (``FRAMES_PER_CHUNK`` / ``LATENTS_PER_CHUNK``).
    Imported here so this check cannot drift from the VAE's chunking.
    """
    from models.tt_dit.pipelines.minimax_h3.packing import (
        MINIMAX_H3_FRAMES_PER_CHUNK,
        MINIMAX_H3_LATENTS_PER_CHUNK,
    )

    return (
        num_frames >= MINIMAX_H3_LATENTS_PER_CHUNK
        and num_frames % MINIMAX_H3_FRAMES_PER_CHUNK == MINIMAX_H3_LATENTS_PER_CHUNK
    )


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
