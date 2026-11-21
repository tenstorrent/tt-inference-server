# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Literal, Optional

from config.settings import get_settings
from domain.generation_request import (
    EXAMPLE_IMAGE_URL,
    EXAMPLE_PNG,
    GenerationModel,
    GenerationRequest,
    Media,
    Seed,
)
from models.tt_dit.pipelines.wan.policy import (
    WAN_RESOLUTIONS_480P,
    WAN_RESOLUTIONS_720P,
    validate_request,
)
from pydantic import Field

WanRatio = Literal["16:9", "9:16"]
WanSize = Literal["480p", "720p"]

# (width, height) per size tier and ratio: Wan2.2's reference sizes.
WAN_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "480p": {"16:9": WAN_RESOLUTIONS_480P[0], "9:16": WAN_RESOLUTIONS_480P[1]},
    "720p": {"16:9": WAN_RESOLUTIONS_720P[0], "9:16": WAN_RESOLUTIONS_720P[1]},
}


def _size_table() -> str:
    def line(ratio: str, small: str, large: str) -> str:
        return f"| {ratio:<14} | {small:<7} | {large:<8} |"

    rows = [
        line("`aspect_ratio`", "`480p`", "`720p`"),
        line("-" * 14, "-" * 7, "-" * 8),
    ]
    for ratio in WAN_SIZES["480p"]:
        (w1, h1), (w2, h2) = WAN_SIZES["480p"][ratio], WAN_SIZES["720p"][ratio]
        rows.append(line(f"`{ratio}`", f"{w1}×{h1}", f"{w2}×{h2}"))
    return "\n".join(rows)


WAN_FPS = 16
# sample_neg_prompt in Wan2.2's wan/configs/shared_config.py.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


class Wan22i2vA14bRequest(GenerationRequest):
    OUTPUT_TYPE = "video"
    MEDIA_FIELDS = ("first_frame", "last_frame")
    # A frame can decode to hundreds of MB, too much to hold while queued and
    # to send to every rank of a multi-host mesh.
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(min_length=1)
    negative_prompt: str = Field(
        default=WAN_NEGATIVE_PROMPT,
        description=(
            "Omitted = Wan2.2's reference default (`sample_neg_prompt` in "
            "`wan/configs/shared_config.py`), a list of common defects in "
            "Chinese: oversaturated, overexposed, static, blurry details, "
            "subtitles, worst quality, JPEG artifacts, deformed hands and faces, "
            "extra fingers, three legs, cluttered background, walking backwards, "
            "and similar. A request's `negative_prompt` replaces it rather than "
            "adding to it."
        ),
    )
    first_frame: Media = Field(
        description=(
            "The image the video starts from. Resized and center-cropped to the "
            "output size."
        )
    )
    last_frame: Optional[Media] = Field(
        default=None,
        description=(
            "The image the video ends on. Resized and center-cropped to the "
            "output size."
        ),
    )
    aspect_ratio: WanRatio = Field(
        description="Output shape. With `size`, it fixes the output size (see `size`)."
    )
    size: WanSize = Field(
        description=(
            "Size tier. With `aspect_ratio`, it fixes the output size:\n\n"
            f"{_size_table()}\n\n"
            "These are Wan2.2's reference sizes."
        )
    )
    frames: int = Field(
        default=81,
        ge=81,
        le=81,
        description=(
            "Number of frames. Only 81 for now, the default of Wan2.2's reference "
            f"config. Played at a fixed {WAN_FPS} fps (the reference config's), so "
            "about 5 s; the output reports `fps` and `duration`."
        ),
    )
    seed: Optional[Seed] = None

    @property
    def fps(self) -> int:
        return WAN_FPS

    def output_size(self) -> tuple[int, int]:
        return WAN_SIZES[self.size][self.aspect_ratio]

    def media_errors(self) -> list[tuple[str, int | None, str]]:
        width, height = self.output_size()
        try:
            validate_request(
                mesh_shape=get_settings().device_mesh_shape, width=width, height=height
            )
        except ValueError as e:
            return [("size", None, str(e))]
        return []

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [size for sizes in WAN_SIZES.values() for size in sizes.values()]


WAN22_I2V_A14B = GenerationModel(
    id="wan-ai/wan2.2-i2v-a14b",
    request=Wan22i2vA14bRequest,
    title="Wan2.2 I2V A14B",
    summary="Submit a Wan2.2 image-to-video job",
    example={
        "prompt": (
            "A close-up of pink plumeria blossoms; the view slowly pulls back and "
            "turns into a sailboat drifting across the sea at dawn."
        ),
        "first_frame": EXAMPLE_IMAGE_URL,
        "last_frame": EXAMPLE_PNG,
        "aspect_ratio": "16:9",
        "size": "480p",
        "frames": 81,
        "seed": 1234,
    },
)
