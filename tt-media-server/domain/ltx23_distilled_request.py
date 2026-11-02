# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Literal, Optional

from config.constants import LTX_HEIGHT, LTX_WIDTH, snap_num_frames
from domain.generation_request import GenerationModel, GenerationRequest, Seed
from pydantic import Field


class Ltx23DistilledRequest(GenerationRequest):
    """One output shape for now: 1920×1080, 6 s at 25 fps. The shape fields are
    kept so more can be allowed later without a schema change."""

    OUTPUT_TYPE = "video"

    prompt: str = Field(min_length=1)
    aspect_ratio: Literal["16:9"] = Field(
        default="16:9", description="Output shape. Only `16:9` for now."
    )
    size: Literal["1080p"] = Field(
        default="1080p", description="Size tier: `1080p` is 1920×1080 at `16:9`."
    )
    duration: Literal[6] = Field(
        default=6,
        description=(
            "Seconds. 6 s at 25 fps is 153 frames (6.12 s): the frame count must "
            "be 8k + 1. Each output reports its exact `duration` and `num_frames`."
        ),
    )
    fps: Literal[25] = Field(
        default=25,
        description=(
            "Frames per second. Only 25 for now: it sets the audio length and the "
            "audio-video alignment, both fixed when the pipeline starts."
        ),
    )
    seed: Optional[Seed] = None

    @property
    def frames(self) -> int:
        return snap_num_frames(self.duration * self.fps)

    def output_size(self) -> tuple[int, int]:
        return LTX_WIDTH, LTX_HEIGHT

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [(LTX_WIDTH, LTX_HEIGHT)]


LTX_2_3_22B_DISTILLED = GenerationModel(
    id="lightricks/ltx-2.3-22b-distilled",
    request=Ltx23DistilledRequest,
    title="LTX-2.3 22B distilled",
    summary="Submit an LTX-2.3 22B distilled job",
    description=(
        "Video with sound from text. Runs the distilled checkpoint of "
        "`Lightricks/LTX-2.3` (`ltx-2.3-22b-distilled-1.1`)."
    ),
    example={
        "prompt": (
            "A fishing boat rocks in a harbour at dawn as gulls call overhead and "
            "waves slap against the hull."
        ),
        "seed": 42,
    },
)
