# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from typing import Annotated, Literal, Optional, Union

from config.settings import get_settings
from domain.generation_request import (
    Count,
    GenerationModel,
    GenerationRequest,
    ImageFormat,
    SchemaComponent,
    Seed,
)
from models.tt_dit.pipelines.ideogram4.policy import (
    IDEOGRAM4_MAX_TEXT_TOKENS,
    IDEOGRAM4_SERVING_SIZES,
    validate_size,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

IdeogramRatio = Literal["1:1", "16:9", "9:16", "2:3"]
IdeogramSize = Literal["2K"]

# (width, height) per size tier and ratio: a long side of 2048, the open
# model's maximum. The initial release serves these four.
IDEOGRAM_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "2K": {
        "1:1": IDEOGRAM4_SERVING_SIZES[0],
        "16:9": IDEOGRAM4_SERVING_SIZES[1],
        "9:16": IDEOGRAM4_SERVING_SIZES[2],
        "2:3": IDEOGRAM4_SERVING_SIZES[3],
    },
}

# The open release's sampler presets: 12, 20 and 48 steps, each with its own
# guidance schedule.
IDEOGRAM_PRESETS = {
    "turbo": "V4_TURBO_12",
    "default": "V4_DEFAULT_20",
    "quality": "V4_QUALITY_48",
}

# The caption models declare their fields in the model's key order, which
# model_dump keeps whatever order the client sent.


Color = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$"), AfterValidator(str.upper)]
Bbox = Annotated[
    list[Annotated[int, Field(ge=0, le=1000)]],
    Field(min_length=4, max_length=4),
    SchemaComponent(
        "Ideogram4Bbox",
        "`[ymin, xmin, ymax, xmax]` on a 0–1000 grid, origin top left, with "
        "`ymin <= ymax` and `xmin <= xmax`.",
    ),
]
Palette = Annotated[
    list[Color],
    Field(max_length=16),
    SchemaComponent(
        "Ideogram4Palette",
        "Dominant colors as `#RRGGBB`, either case; sent to the model in uppercase.",
    ),
]
ElementPalette = Annotated[
    list[Color],
    Field(max_length=5),
    SchemaComponent(
        "Ideogram4ElementPalette",
        "The element's colors as `#RRGGBB`, either case; sent to the model in "
        "uppercase.",
    ),
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ideogram4StyleDescription(_Strict):
    """Visual style. Exactly one of `photo` (photographs) or `art_style`
    (everything else)."""

    aesthetics: str = Field(examples=["moody, cinematic, desaturated"])
    lighting: str = Field(examples=["golden hour, rim light, dramatic shadows"])
    photo: Optional[str] = Field(
        default=None,
        description="Camera and lens, for photographs.",
        examples=["35mm, f/1.4, bokeh"],
    )
    medium: str = Field(
        examples=[
            "photograph",
            "illustration",
            "3d_render",
            "painting",
            "graphic_design",
        ]
    )
    art_style: Optional[str] = Field(
        default=None,
        description="Art style, for anything but photographs.",
        examples=["flat vector illustration, bold outlines"],
    )
    color_palette: Optional[Palette] = None

    @model_validator(mode="after")
    def _photo_or_art_style(self):
        if (self.photo is None) == (self.art_style is None):
            raise ValueError("exactly one of photo and art_style is required")
        return self


class _Element(_Strict):
    # bbox is declared by each element type, after `type`.
    @field_validator("bbox", check_fields=False)
    @classmethod
    def _ordered_corners(cls, bbox):
        if bbox is not None and (bbox[0] > bbox[2] or bbox[1] > bbox[3]):
            raise ValueError("bbox needs ymin <= ymax and xmin <= xmax")
        return bbox


class Ideogram4ObjectElement(_Element):
    type: Literal["obj"]
    bbox: Optional[Bbox] = None
    desc: str
    color_palette: Optional[ElementPalette] = None


class Ideogram4TextElement(_Element):
    type: Literal["text"]
    bbox: Optional[Bbox] = None
    text: str = Field(description="The literal text to render.")
    desc: Optional[str] = None
    color_palette: Optional[ElementPalette] = None


class Ideogram4CompositionalDeconstruction(_Strict):
    background: str
    elements: list[
        Annotated[
            Union[Ideogram4ObjectElement, Ideogram4TextElement],
            Field(discriminator="type"),
        ]
    ]


class Ideogram4Caption(_Strict):
    """A structured caption in the format Ideogram 4 was trained on, sent to
    the model as is. Key order does not matter: the server writes it out in
    the model's order. Each output reports it as its `prompt`.

    There is no plain-text prompt for now: Ideogram 4 needs a caption like
    this, and turning plain text into one takes an LLM ("magic prompt"),
    which this API doesn't run yet.

    Format: `docs/prompting.md` in `ideogram-oss/ideogram4`."""

    high_level_description: Optional[str] = Field(
        default=None,
        description="One or two sentences summing up the image. Recommended.",
    )
    style_description: Optional[Ideogram4StyleDescription] = None
    compositional_deconstruction: Ideogram4CompositionalDeconstruction


class Ideogram4Request(GenerationRequest):
    OUTPUT_TYPE = "image"

    structured_prompt: Ideogram4Caption = Field(
        description=(
            f"At most {IDEOGRAM4_MAX_TEXT_TOKENS} tokens as the model reads it, "
            "chat template included; a longer caption fails the job."
        )
    )
    count: Count = 1
    aspect_ratio: IdeogramRatio = Field(
        default="1:1",
        description=(
            "Output shape. Only these for now; more can be allowed later without "
            "a schema change. Each output reports its exact size."
        ),
    )
    size: IdeogramSize = Field(
        default="2K",
        description=(
            "Size tier: a long side of 2048, the open model's maximum. Only `2K` "
            "for now; `1K` can be allowed later without a schema change.\n\n"
            "| `aspect_ratio` | `2K`      |\n"
            "| -------------- | --------- |\n"
            "| `1:1`          | 2048×2048 |\n"
            "| `16:9`         | 2048×1152 |\n"
            "| `9:16`         | 1152×2048 |\n"
            "| `2:3`          | 1360×2048 |"
        ),
    )
    format: ImageFormat = "png"
    seed: Optional[Seed] = None
    rendering_speed: Literal["turbo", "default", "quality"] = Field(
        default="default",
        description=(
            "Speed and quality preset: `turbo` is fastest, `quality` gives the "
            "best images, `default` is in between."
        ),
    )

    def output_size(self) -> tuple[int, int]:
        return IDEOGRAM_SIZES[self.size][self.aspect_ratio]

    def media_errors(self) -> list[tuple[str, int | None, str]]:
        width, height = self.output_size()
        try:
            validate_size(
                mesh_shape=get_settings().device_mesh_shape, width=width, height=height
            )
        except ValueError as e:
            return [("size", None, str(e))]
        return []

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [size for sizes in IDEOGRAM_SIZES.values() for size in sizes.values()]

    def model_prompt(self) -> str:
        # Compact, in the model's key order, as the reference serializes it.
        return json.dumps(
            self.structured_prompt.model_dump(exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )


IDEOGRAM_4 = GenerationModel(
    id="ideogram-ai/ideogram-4",
    request=Ideogram4Request,
    title="Ideogram 4.0",
    summary="Submit an Ideogram 4.0 text-to-image job",
    example={
        "structured_prompt": {
            "high_level_description": (
                'A neon sign reading "OPEN LATE" on a brick wall at night.'
            ),
            "style_description": {
                "aesthetics": "moody, urban, vibrant",
                "lighting": "neon glow, deep shadows",
                "medium": "graphic_design",
                "art_style": "retro neon signage",
                "color_palette": ["#FF2E88", "#1B1B2F", "#F5F5F5"],
            },
            "compositional_deconstruction": {
                "background": "A dark red brick wall, wet from rain.",
                "elements": [
                    {
                        "type": "text",
                        "bbox": [350, 150, 650, 850],
                        "text": "OPEN LATE",
                        "desc": "Glowing pink neon letters, slightly flickering.",
                    }
                ],
            },
        },
        "aspect_ratio": "2:3",
        "size": "2K",
        "rendering_speed": "quality",
    },
)
