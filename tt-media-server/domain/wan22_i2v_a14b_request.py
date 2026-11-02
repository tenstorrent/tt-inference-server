# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Literal, Optional

from domain.generation_request import GenerationRequest, Media, Seed
from pydantic import Field

WanRatio = Literal["16:9", "9:16"]
WanSize = Literal["480p", "720p"]

# (width, height) per size tier and ratio: Wan2.2's reference sizes.
WAN_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "480p": {"16:9": (832, 480), "9:16": (480, 832)},
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280)},
}
WAN_FPS = 16
# sample_neg_prompt in Wan2.2's wan/configs/shared_config.py.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


class Wan22i2vA14bRequest(GenerationRequest):
    MODEL_ID = "wan-ai/wan2.2-i2v-a14b"
    OUTPUT_TYPE = "video"
    TITLE = "Wan2.2 I2V A14B"
    SUMMARY = "Submit a Wan2.2 image-to-video job"
    EXAMPLE = {
        "prompt": "the boat drifts out of the harbour at dawn",
        "first_frame": "data:image/png;base64,iVBORw0KGgo…",
        "last_frame": "https://storage.internal/inputs/end.png",
        "aspect_ratio": "16:9",
        "size": "720p",
        "frames": 81,
        "seed": 1234,
    }
    MEDIA_FIELDS = ("first_frame", "last_frame")

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
            "| `aspect_ratio` | `480p`  | `720p`   |\n"
            "| -------------- | ------- | -------- |\n"
            "| `16:9`         | 832×480 | 1280×720 |\n"
            "| `9:16`         | 480×832 | 720×1280 |\n\n"
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

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [size for sizes in WAN_SIZES.values() for size in sizes.values()]
