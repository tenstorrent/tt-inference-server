# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-H3-Image: text to image, or an edit of 1 to 16 reference images.

The served envelope is the bring-up record's Stage 2 spec: a fixed list of 18
sizes (5 ratios plus portrait flips, at ~16 MP and at 2K), JPEG by default, no
prompt rewriting, the release's fixed 6-step schedule. tt-metal's
models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3_image.py checks the same
reference limits and is sized (and warmed) for exactly what they admit: the
runner passes it H3_IMAGE_REFERENCE_AREA_LIMIT.
"""

from typing import Literal, Optional

from domain.generation_request import (
    EXAMPLE_IMAGE_URL,
    EXAMPLE_PNG,
    Count,
    GenerationModel,
    GenerationRequest,
    Media,
    Seed,
    closest_ratio,
)
from pydantic import Field

H3ImageRatio = Literal[
    "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21"
]
H3ImageResolution = Literal["4k", "2k"]

# Landscape sizes per ratio (portrait = flipped), on the 32-px grid, equal area
# within a tier.
_LANDSCAPE: dict[str, dict[str, tuple[int, int]]] = {
    "4k": {
        "1:1": (4096, 4096),
        "4:3": (4736, 3552),
        "3:2": (5024, 3360),
        "16:9": (5472, 3072),
        "21:9": (6272, 2688),
    },
    "2k": {
        "1:1": (2048, 2048),
        "4:3": (2368, 1760),
        "3:2": (2496, 1664),
        "16:9": (2720, 1536),
        "21:9": (3136, 1344),
    },
}


def _with_portraits(sizes: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    out = dict(sizes)
    for ratio, (w, h) in sizes.items():
        a, b = ratio.split(":")
        if a != b:
            out[f"{b}:{a}"] = (h, w)
    return out


H3_IMAGE_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    tier: _with_portraits(sizes) for tier, sizes in _LANDSCAPE.items()
}

# References: MiniMax's API limits, plus the per-request total the Galaxy is
# sized for. Effective area = a reference's pixels after its short side is
# scaled into 1024–2048 px (the release pipeline's VAE copy): a 4032×3024 photo
# counts as 2720×2048 = 5.6 MP, a 1024×1024 one as 1.0 MP.
H3_IMAGE_MAX_REFERENCES = 16
H3_IMAGE_REFERENCE_SIDE = (128, 8192)
H3_IMAGE_REFERENCE_MAX_RATIO = 4
H3_IMAGE_REFERENCE_SHORT_EDGE = (1024, 2048)
# TBD by the Galaxy sweep (32 / 48 / 64 / 96 MP). The runner passes it to the
# pipeline, which sizes its arenas and warmup from it. The `images` description
# quotes it; the class docstring repeats it (a test checks both). After a
# change, rerun scripts/dump_generations_openapi.py.
H3_IMAGE_REFERENCE_AREA_LIMIT = 48_000_000
_AREA_LIMIT_MP = f"{H3_IMAGE_REFERENCE_AREA_LIMIT / 1e6:g} MP"
H3_IMAGE_GRID = 32


def effective_area(width: int, height: int) -> int:
    """Pixels of a reference once its short side is scaled into 1024–2048 px,
    each side rounded to 32 px, as the release pipeline sizes its VAE copy."""
    lo, hi = H3_IMAGE_REFERENCE_SHORT_EDGE
    short = min(width, height)
    if lo <= short <= hi:
        return width * height
    scale = (lo if short < lo else hi) / short
    w = max(H3_IMAGE_GRID, round(width * scale / H3_IMAGE_GRID) * H3_IMAGE_GRID)
    h = max(H3_IMAGE_GRID, round(height * scale / H3_IMAGE_GRID) * H3_IMAGE_GRID)
    return w * h


class MiniMaxH3ImageRequest(GenerationRequest):
    """Text to image when `images` is empty, else an edit of those images.

    References: at most 16; each 128–8192 px a side with aspect ratio at most
    4:1; and at most 48 MP of effective area together (each counted after its
    short side is scaled into 1024–2048 px), or 422. The prompt reaches the
    model as sent (no rewriting); past 4096 tokens it is truncated, as in the
    release pipeline."""

    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)
    # Up to 16 references of up to 8192 px a side: only their files and sizes
    # are kept until the runner decodes them.
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(
        min_length=1,
        max_length=32_000,
        description=(
            "What to generate, or the edit to make. Refer to references as "
            "`<Picture 1>` … `<Picture N>`, numbered in the order of `images`. "
            "Sent to the model as is; past 4096 tokens it is truncated."
        ),
    )
    images: list[Media] = Field(
        default_factory=list,
        max_length=H3_IMAGE_MAX_REFERENCES,
        description=(
            "Reference images for an edit; omit for text to image. 1–16 images, "
            "each 128–8192 px a side with aspect ratio at most 4:1, and at most "
            f"{_AREA_LIMIT_MP} of effective area together: each image counts at "
            "its size after its short side is scaled into 1024–2048 px (a "
            "4032×3024 photo counts 5.6 MP, a 1024×1024 image 1.0 MP). Over a "
            "limit: 422."
        ),
    )
    aspect_ratio: Optional[H3ImageRatio] = Field(
        default=None,
        description=(
            "Output shape. Omitted = `1:1` for text to image, and for an edit the "
            "ratio closest to the first image's. Sizes are in `resolution`."
        ),
    )
    resolution: H3ImageResolution = Field(
        default="4k",
        description=(
            "`4k`, about 16 MP: 1:1 4096×4096, 4:3 4736×3552, 3:2 5024×3360, "
            "16:9 5472×3072, 21:9 6272×2688. `2k`, about 4.2 MP: 1:1 2048×2048, "
            "4:3 2368×1760, 3:2 2496×1664, 16:9 2720×1536, 21:9 3136×1344. "
            "Portrait ratios flip width and height. Each output reports its "
            "exact size."
        ),
    )
    count: Count = 1
    format: Literal["jpeg", "png"] = Field(
        default="jpeg",
        description="Output format. `png` is lossless but adds ~4.5 s at 4k.",
    )
    seed: Optional[Seed] = None

    def media_errors(self) -> list[tuple[str, int | None, str]]:
        errors = []
        sizes = self._image_sizes.get("images", [])
        lo, hi = H3_IMAGE_REFERENCE_SIDE
        for index, (w, h) in enumerate(sizes):
            if not (lo <= min(w, h) and max(w, h) <= hi):
                errors.append(
                    (
                        "images",
                        index,
                        f"is {w}×{h}; each side must be from {lo} to {hi} px",
                    )
                )
            elif max(w, h) > H3_IMAGE_REFERENCE_MAX_RATIO * min(w, h):
                errors.append(
                    ("images", index, f"is {w}×{h}; the ratio must be from 1:4 to 4:1")
                )
        if sizes and not errors:
            total = sum(effective_area(w, h) for w, h in sizes)
            if total > H3_IMAGE_REFERENCE_AREA_LIMIT:
                # Against the field as a whole: no one image is over.
                errors.append(
                    (
                        "images",
                        None,
                        f"together {total / 1e6:.1f} MP of effective area (each "
                        "counted after its short side is scaled into 1024–2048 px); "
                        f"at most {H3_IMAGE_REFERENCE_AREA_LIMIT / 1e6:g} MP",
                    )
                )
        return errors

    def output_size(self) -> tuple[int, int]:
        sizes = H3_IMAGE_SIZES[self.resolution]
        ratio = self.aspect_ratio
        if ratio is None:
            images = self._image_sizes.get("images")
            ratio = closest_ratio(*images[0], sizes) if images else "1:1"
        return sizes[ratio]

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [size for sizes in H3_IMAGE_SIZES.values() for size in sizes.values()]


MINIMAX_H3_IMAGE = GenerationModel(
    id="minimaxai/minimax-h3-image",
    request=MiniMaxH3ImageRequest,
    title="MiniMax-H3-Image",
    summary="Submit a MiniMax-H3-Image job",
    description=(
        "Images up to about 16 MP from text, or an edit of 1 to 16 reference "
        "images. Runs `MiniMaxAI/MiniMax-H3-Image` (6 steps) on one Galaxy."
    ),
    example={
        "prompt": (
            "Place the flower from <Picture 1> in a glass vase on the sunlit "
            "windowsill of <Picture 2>."
        ),
        "images": [EXAMPLE_IMAGE_URL, EXAMPLE_PNG],
        "aspect_ratio": "3:2",
        "resolution": "4k",
        "format": "jpeg",
    },
)
