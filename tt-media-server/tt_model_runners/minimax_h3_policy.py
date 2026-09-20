# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""MiniMax-H3 t2va serving policy for this deployment.

Which aspect ratios, durations, and step counts the media-server will accept.
Canvas size and the video VAE's ``17n + 5`` frame rule live in the model's
packing module; copying either here is how they drift. FPS is the same: read
``MINIMAX_H3_FPS`` from packing at the point of use.
"""

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
