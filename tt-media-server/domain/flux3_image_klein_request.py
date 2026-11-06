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
from PIL import Image, ImageOps
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
def flux3_upsampler() -> Optional[str]:
    """Where this deployment upsamples prompts: ``"remote"`` (the console's
    upsampler), ``"local"`` (the pipeline's own, on the device), or None when
    prompts are frozen and used exactly as sent."""
    from config.settings import get_settings

    settings = get_settings()
    return None if settings.prompt_upsampler_frozen else settings.prompt_upsampler


# The side of the square a reference is cropped and resized to.
FLUX3_KLEIN_REFERENCE_SIZE = 1024


def flux3_reference(image: Image.Image, size: int = FLUX3_KLEIN_REFERENCE_SIZE):
    """The reference as the image model sees it: tt-metal's
    pipeline_flux3_klein.prepare_reference (upright, RGB, the centered square,
    LANCZOS to ``size``). The prompt upsampler is shown the same image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return (
        image if side == size else image.resize((size, size), Image.Resampling.LANCZOS)
    )


class Flux3ImageKleinRequest(GenerationRequest):
    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)

    prompt: str = Field(
        min_length=1,
        description=(
            "What to generate or how to edit. The prompt upsampler expands it into "
            "the detailed caption the image is generated from, which each output "
            "reports as its `prompt`."
        ),
    )
    images: list[Media] = Field(
        default_factory=list,
        max_length=1,
        description=(
            "The image to edit, center-cropped to a square and resized to "
            "1024×1024. Omitted = text to image."
        ),
    )
    count: Count = 1
    aspect_ratio: Flux3KleinRatio = Field(
        default="1:1", description="Output shape. Only `1:1` for now."
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

    async def prepare_prompt(self) -> str:
        # With the console's upsampler, every prompt goes through it here; the
        # local one rewrites it on the device (TTFlux3ImageKleinRunner).
        if flux3_upsampler() != "remote":
            return self.model_prompt()
        from utils.prompt_upsampler import upsample

        references = [
            flux3_reference(image) for image in self._images.get("images", [])
        ]
        return (await upsample(self.model_prompt(), references)).caption


FLUX3_IMAGE_KLEIN = GenerationModel(
    id="black-forest-labs/flux-3-image-klein",
    request=Flux3ImageKleinRequest,
    title="FLUX.3 Image klein",
    summary="Submit a FLUX.3 Image klein job",
    description="Text to image, or editing when `images` is given.",
    example={
        "prompt": "a lighthouse on a cliff, golden hour",
        "size": "2K",
        "format": "webp",
    },
)
