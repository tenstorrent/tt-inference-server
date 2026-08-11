# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Output-media specifications for request-matched MiniMax fixture videos."""

from __future__ import annotations

from minimax_mock.schemas import (
    AspectRatio,
    ContentRole,
    ContentType,
    Resolution,
    VideoGenerationRequest,
)

FIXTURE_VIDEO_FPS = 24
DEFAULT_ADAPTIVE_RATIO = AspectRatio.RATIO_16_9
OUTPUT_RATIOS = (
    AspectRatio.RATIO_21_9,
    AspectRatio.RATIO_16_9,
    AspectRatio.RATIO_4_3,
    AspectRatio.RATIO_1_1,
    AspectRatio.RATIO_3_4,
    AspectRatio.RATIO_9_16,
)
RATIO_ASSET_KEYS = {
    AspectRatio.RATIO_21_9: "21x9",
    AspectRatio.RATIO_16_9: "16x9",
    AspectRatio.RATIO_4_3: "4x3",
    AspectRatio.RATIO_1_1: "1x1",
    AspectRatio.RATIO_3_4: "3x4",
    AspectRatio.RATIO_9_16: "9x16",
}

# MiniMax documents 768P output as 24 fps, dimensions divisible by 32, and an
# area no larger than 768 × 1344. The documented 2K example is 2528 × 1440.
# The remaining dimensions preserve the requested ratio while staying aligned
# to 32 pixels and near the same per-resolution pixel budget.
OUTPUT_DIMENSIONS = {
    Resolution.P768: {
        AspectRatio.RATIO_21_9: (1504, 640),
        AspectRatio.RATIO_16_9: (1344, 768),
        AspectRatio.RATIO_4_3: (1152, 864),
        AspectRatio.RATIO_1_1: (992, 992),
        AspectRatio.RATIO_3_4: (864, 1152),
        AspectRatio.RATIO_9_16: (768, 1344),
    },
    Resolution.P2K: {
        AspectRatio.RATIO_21_9: (2848, 1216),
        AspectRatio.RATIO_16_9: (2528, 1440),
        AspectRatio.RATIO_4_3: (2176, 1632),
        AspectRatio.RATIO_1_1: (1856, 1856),
        AspectRatio.RATIO_3_4: (1632, 2176),
        AspectRatio.RATIO_9_16: (1440, 2528),
    },
}


def effective_output_ratio(request: VideoGenerationRequest) -> AspectRatio:
    """Return the concrete ratio represented by a mock output video."""

    has_frame_conditioning = any(
        item.type is ContentType.IMAGE_URL
        and item.role
        in {
            None,
            ContentRole.FIRST_FRAME,
            ContentRole.LAST_FRAME,
        }
        for item in request.content
    )
    if (
        has_frame_conditioning
        or request.ratio is None
        or request.ratio is AspectRatio.ADAPTIVE
    ):
        return DEFAULT_ADAPTIVE_RATIO
    return request.ratio


def asset_template_values(
    resolution: Resolution,
    ratio: AspectRatio,
    duration: int,
) -> dict[str, str | int]:
    return {
        "resolution": resolution.value.lower(),
        "ratio": RATIO_ASSET_KEYS[ratio],
        "duration": duration,
    }
