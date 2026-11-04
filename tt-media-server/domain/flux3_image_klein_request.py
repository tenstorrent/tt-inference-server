# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Annotated, Literal, Optional

from domain.generation_request import (
    Count,
    GenerationModel,
    GenerationRequest,
    ImageFormat,
    Media,
    Seed,
)
from pydantic import Field, WithJsonSchema

# An enum, not a const, so more ratios can be added without a schema change.
Flux3KleinRatio = Annotated[
    Literal["1:1"], WithJsonSchema({"type": "string", "enum": ["1:1"]})
]
Flux3KleinSize = Literal["1K", "2K", "4K"]

FLUX3_KLEIN_SIZES: dict[str, tuple[int, int]] = {
    "1K": (1024, 1024),
    "2K": (2048, 2048),
    "4K": (4096, 4096),
}
FLUX3_KLEIN_STEPS = 16


class Flux3ImageKleinRequest(GenerationRequest):
    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)

    prompt: str = Field(
        min_length=1,
        description=(
            "What to generate or how to edit. The model's prompt upsampler "
            "rewrites it into a detailed caption, which each output reports as "
            "its `prompt`."
        ),
    )
    images: list[Media] = Field(
        default_factory=list,
        max_length=1,
        description=(
            "The image to edit, center-cropped to a square and resized to the "
            "output size. Omitted = text to image."
        ),
    )
    count: Count = 1
    aspect_ratio: Optional[Flux3KleinRatio] = Field(
        default=None, description="Output shape. Only `1:1` for now."
    )
    size: Flux3KleinSize = Field(
        default="2K",
        description="Size tier: `1K` 1024×1024, `2K` 2048×2048, `4K` 4096×4096.",
    )
    format: ImageFormat = "png"
    seed: Optional[Seed] = None

    def output_size(self) -> tuple[int, int]:
        return FLUX3_KLEIN_SIZES[self.size]

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return list(FLUX3_KLEIN_SIZES.values())


FLUX3_IMAGE_KLEIN = GenerationModel(
    id="black-forest-labs/flux-3-image-klein",
    request=Flux3ImageKleinRequest,
    title="FLUX.3 Image klein",
    summary="Submit a FLUX.3 Image klein job",
    description=(
        "Text to image, or editing when `images` is given. Every prompt is "
        "rewritten by the model's prompt upsampler first."
    ),
    example={
        "prompt": "a lighthouse on a cliff, golden hour",
        "size": "2K",
        "format": "webp",
    },
)
