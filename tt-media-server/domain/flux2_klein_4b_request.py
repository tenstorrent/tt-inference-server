# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Literal, Optional

from domain.generation_request import (
    Count,
    GenerationModel,
    GenerationRequest,
    ImageFormat,
    Media,
    Seed,
)
from pydantic import Field

KleinRatio = Literal["1:1"]
KleinSize = Literal["1K", "2K"]

# (width, height) per size tier and ratio: about 1 MP at 1K, twice the sides at
# 2K. The initial release serves 1:1 only.
KLEIN_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "1K": {"1:1": (1024, 1024)},
    "2K": {"1:1": (2048, 2048)},
}
KLEIN_STEPS = 4


class Flux2klein4bRequest(GenerationRequest):
    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(min_length=1)
    images: list[Media] = Field(
        default_factory=list,
        max_length=1,
        description=(
            "Image to edit or draw from: 0 or 1 for now; a list so more can be "
            "allowed later without a schema change. Omitted = text to image. "
            "Center-cropped to the output's aspect ratio and resized to about 1 MP."
        ),
    )
    count: Count = 1
    aspect_ratio: KleinRatio = Field(
        default="1:1",
        description=(
            "Output shape. Only `1:1` for now; more can be allowed later without "
            "a schema change. Each output reports its exact size."
        ),
    )
    size: KleinSize = Field(
        default="1K",
        description=(
            "Size tier. With `aspect_ratio`, it fixes the output size:\n\n"
            "| `aspect_ratio` | `1K`      | `2K`      |\n"
            "| -------------- | --------- | --------- |\n"
            "| `1:1`          | 1024×1024 | 2048×2048 |"
        ),
    )
    format: ImageFormat = "png"
    seed: Optional[Seed] = None

    def output_size(self) -> tuple[int, int]:
        return KLEIN_SIZES[self.size][self.aspect_ratio]

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [size for sizes in KLEIN_SIZES.values() for size in sizes.values()]


FLUX2_KLEIN_4B = GenerationModel(
    id="black-forest-labs/flux.2-klein-4b",
    request=Flux2klein4bRequest,
    title="FLUX.2 klein 4B",
    summary="Submit a FLUX.2 klein 4B job",
    description="Text to image, or editing when `images` is given.",
    example={
        "prompt": "a lighthouse on a cliff, golden hour",
        "aspect_ratio": "1:1",
        "size": "2K",
        "format": "webp",
    },
)
