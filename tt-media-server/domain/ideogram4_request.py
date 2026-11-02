# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from typing import Annotated, Literal, Optional, Union

from domain.generation_request import (
    Count,
    GenerationRequest,
    ImageFormat,
    SchemaComponent,
    Seed,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IdeogramRatio = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
IdeogramSize = Literal["1K", "2K"]

# (width, height) per size tier and ratio: Ideogram's sizes of about 1 MP at
# 1K, a long side of 2048 (the open model's maximum) at 2K.
IDEOGRAM_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "1K": {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "3:2": (1248, 832),
        "2:3": (832, 1248),
    },
    "2K": {
        "1:1": (2048, 2048),
        "16:9": (2048, 1152),
        "9:16": (1152, 2048),
        "4:3": (2048, 1536),
        "3:4": (1536, 2048),
        "3:2": (2048, 1360),
        "2:3": (1360, 2048),
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

Color = Annotated[str, Field(pattern=r"^#[0-9A-F]{6}$")]
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
    SchemaComponent("Ideogram4Palette", "Dominant colors as uppercase `#RRGGBB`."),
]
ElementPalette = Annotated[
    list[Color],
    Field(max_length=5),
    SchemaComponent(
        "Ideogram4ElementPalette", "The element's colors as uppercase `#RRGGBB`."
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
    MODEL_ID = "ideogram-ai/ideogram-4"
    OUTPUT_TYPE = "image"
    TITLE = "Ideogram 4.0"
    SUMMARY = "Submit an Ideogram 4.0 text-to-image job"
    EXAMPLE = {
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
    }

    structured_prompt: Ideogram4Caption
    count: Count = 1
    aspect_ratio: IdeogramRatio = Field(
        default="1:1", description="Output shape. Each output reports its exact size."
    )
    size: IdeogramSize = Field(
        default="2K",
        description=(
            "Size tier. With `aspect_ratio`, it fixes the output size. `1K` uses "
            "the model's native sizes of about 1 MP; `2K` has a long side of 2048, "
            "the open model's maximum (256–2048 per side, multiples of 16):\n\n"
            "| `aspect_ratio` | `1K`      | `2K`      |\n"
            "| -------------- | --------- | --------- |\n"
            "| `1:1`          | 1024×1024 | 2048×2048 |\n"
            "| `16:9`         | 1280×720  | 2048×1152 |\n"
            "| `9:16`         | 720×1280  | 1152×2048 |\n"
            "| `4:3`          | 1152×864  | 2048×1536 |\n"
            "| `3:4`          | 864×1152  | 1536×2048 |\n"
            "| `3:2`          | 1248×832  | 2048×1360 |\n"
            "| `2:3`          | 832×1248  | 1360×2048 |"
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
