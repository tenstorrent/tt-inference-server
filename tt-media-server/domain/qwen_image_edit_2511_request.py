# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Literal, Optional

from domain.generation_request import (
    EXAMPLE_IMAGE_URL,
    EXAMPLE_PNG,
    Count,
    GenerationRequest,
    GenerationModel,
    ImageFormat,
    Media,
    Seed,
    closest_ratio,
)
from pydantic import Field

QwenRatio = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]

# diffusers' edit sizes, about 1 MP each, as tt_dit's Qwen-Image pipeline has them.
QWEN_EDIT_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
}


class QwenImageEdit2511Request(GenerationRequest):
    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)

    prompt: str = Field(min_length=1, description="The edit instruction.")
    negative_prompt: str = Field(
        default=" ",
        description=(
            'Omitted = `" "` (a single space), the model card\'s default, so '
            "classifier-free guidance always applies."
        ),
    )
    images: list[Media] = Field(
        min_length=1,
        max_length=3,
        description=(
            "The image(s) to edit or combine. Each is resized to the supported "
            "size closest to its aspect ratio. At most 3, as in tt_dit's "
            "Qwen-Image-Edit-2511 pipeline."
        ),
    )
    count: Count = 1
    aspect_ratio: Optional[QwenRatio] = Field(
        default=None,
        description=(
            "Output shape, at about 1 MP: `1:1` 1024×1024, `16:9` 1376×768, "
            "`9:16` 768×1376, `4:3` 1184×896, `3:4` 896×1184, `3:2` 1248×832, "
            "`2:3` 832×1248 (diffusers' edit sizes, as in tt_dit's pipeline). "
            "Omitted = the ratio closest to the last image's, as in diffusers' "
            "reference pipeline. Each output reports its exact size."
        ),
    )
    format: ImageFormat = "png"
    seed: Optional[Seed] = None

    def output_size(self) -> tuple[int, int]:
        ratio = self.aspect_ratio or closest_ratio(
            *self._images["images"][-1].size, QWEN_EDIT_SIZES
        )
        return QWEN_EDIT_SIZES[ratio]

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return list(QWEN_EDIT_SIZES.values())


QWEN_IMAGE_EDIT_2511 = GenerationModel(
    id="qwen/qwen-image-edit-2511",
    request=QwenImageEdit2511Request,
    title="Qwen-Image-Edit-2511",
    summary="Submit a Qwen-Image-Edit-2511 job",
    example={
        "prompt": "put the flower from the first image on the water in the second",
        "negative_prompt": "blurry",
        "images": [EXAMPLE_IMAGE_URL, EXAMPLE_PNG],
        "format": "png",
    },
)
