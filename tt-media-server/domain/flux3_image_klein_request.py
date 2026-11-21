# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
from typing import Annotated, Literal, Optional

from domain.generation_request import (
    Count,
    GenerationModel,
    GenerationRequest,
    ImageFormat,
    Media,
    Seed,
)
from pydantic import Field, WithJsonSchema, model_validator

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

# The prompt upsampler runs in the deployment (default: on the first chip) or
# ahead of it (FLUX3_PROMPT_UPSAMPLER_FROZEN=1: prompts reach the model as
# sent, the pipeline is built without it). Resident, its reserved DRAM is held
# on every chip of the mesh and the DiT's 4096 activations no longer fit (jobs
# 192950, 194825, 194959), so 4K is served only with the upsampler ahead.
FLUX3_PROMPT_UPSAMPLER_FROZEN = "FLUX3_PROMPT_UPSAMPLER_FROZEN"
FLUX3_RESIDENT_UNSERVED_SIZES = ("4K",)


def upsampler_frozen() -> bool:
    raw = os.environ.get(FLUX3_PROMPT_UPSAMPLER_FROZEN, "0")
    key = raw.strip().lower()
    if key in ("1", "true"):
        return True
    if key in ("0", "false"):
        return False
    raise ValueError(
        f"{FLUX3_PROMPT_UPSAMPLER_FROZEN}={raw!r} must be 1/true or 0/false"
    )


class Flux3ImageKleinRequest(GenerationRequest):
    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(
        min_length=1,
        max_length=20_000,
        description=(
            "What to generate or how to edit, in plain words. The model's prompt "
            "upsampler rewrites it into the detailed caption the model was trained "
            "on; it runs in the deployment or ahead of it "
            "(`FLUX3_PROMPT_UPSAMPLER_FROZEN`), and with it ahead the prompt reaches "
            "the model as sent. Each output reports the caption it was rendered "
            "from as its `prompt`. At most 20,000 characters; the upsampler's input "
            "(prompt, reference and template) is at most 8,191 tokens, a longer one "
            "fails the job, and the caption is encoded, truncated past 4096 "
            "tokens.\n\n"
            "Refer to `images` by tag: `<ref_image_0>` is the first.\n\n"
            "Bounding boxes go at the end of the prompt: tag each element in the "
            "text (`<balloon_1>`), then add one space and a JSON list with a row "
            "per tag. Coordinates are `[top, left, bottom, right]` on a 0–1000 grid "
            "with (0, 0) at the top left, whatever the output shape.\n\n"
            '- New element: `{"id": "balloon_1", "bbox": [80, 60, 480, 380], '
            '"desc": "…"}`.\n'
            '- Element of a reference (edits): `{"id": "knight_1", "from": '
            '"ref_image_0", "src_bbox": […], "tgt_bbox": […], "desc": "…"}`. The '
            "same box in both keeps it; different boxes move or resize it; "
            '`"tgt_bbox": null` removes it; `"from": null, "src_bbox": null` adds '
            "a new element."
        ),
    )
    images: list[Media] = Field(
        default_factory=list,
        max_length=1,
        description=(
            "Image to edit: 0 or 1 for now; a list so more can be allowed later "
            "without a schema change. Omitted = text to image. Turned upright (EXIF "
            "orientation) and kept at its aspect ratio: scaled down, never up, to a "
            "canvas of at most 4 MP with sides of 64–4096 px (rounded up to 16 px, "
            "the border padded with the image's mirrored edge), never cropped. Any "
            "size up to 89,478,485 pixels (PIL's decompression-bomb limit) and 30 MB "
            "is accepted."
        ),
    )
    count: Count = 1
    aspect_ratio: Flux3KleinRatio = Field(
        default="1:1",
        description=(
            "Output shape. Only `1:1` for now; more can be allowed later without "
            "a schema change. Each output reports its exact size. Boxes are placed "
            "on the output's grid."
        ),
    )
    size: Flux3KleinSize = Field(
        default="2K",
        description=(
            "Size tier. With `aspect_ratio`, it fixes the output size:\n\n"
            "| `aspect_ratio` | `1K`      | `2K`      | `4K`      |\n"
            "| -------------- | --------- | --------- | --------- |\n"
            "| `1:1`          | 1024×1024 | 2048×2048 | 4096×4096 |\n\n"
            "`4K` is served with the prompt upsampler ahead of the deployment "
            "(`FLUX3_PROMPT_UPSAMPLER_FROZEN=1`); with it in the deployment, a `4K` "
            "request is refused with 422."
        ),
    )
    format: ImageFormat = "png"
    seed: Optional[Seed] = None

    @model_validator(mode="after")
    def _size_is_served(self):
        if self.size not in self.served_sizes():
            raise ValueError(
                f"{self.size} is served with the prompt upsampler ahead of the "
                f"deployment ({FLUX3_PROMPT_UPSAMPLER_FROZEN}=1)"
            )
        return self

    def output_size(self) -> tuple[int, int]:
        return FLUX3_KLEIN_SIZES[self.size]

    @classmethod
    def served_sizes(cls) -> dict[str, tuple[int, int]]:
        """The size tiers this deployment serves."""
        if upsampler_frozen():
            return FLUX3_KLEIN_SIZES
        return {
            size: shape
            for size, shape in FLUX3_KLEIN_SIZES.items()
            if size not in FLUX3_RESIDENT_UNSERVED_SIZES
        }

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return list(cls.served_sizes().values())


FLUX3_IMAGE_KLEIN = GenerationModel(
    id="black-forest-labs/flux-3-image-klein",
    request=Flux3ImageKleinRequest,
    title="FLUX.3 Image klein",
    summary="Submit a FLUX.3 Image klein job",
    description=(
        "Text to image, or editing a reference when `images` is given. Every "
        "prompt is rewritten by the model's prompt upsampler, which runs in the "
        "deployment or ahead of it; with it ahead, prompts reach the model as sent."
    ),
    example={
        "prompt": "a lighthouse on a cliff, golden hour",
        "aspect_ratio": "1:1",
        "size": "2K",
        "format": "webp",
    },
)
