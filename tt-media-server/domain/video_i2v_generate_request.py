# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Image-to-Video request schema for Wan2.2 I2V and MiniMax-H3 FL2VA.

Extends ``VideoGenerateRequest`` with a list of image prompts. Each entry
pairs an image — base64-encoded, or an http(s)/presigned URL (#4974) — with
a frame position so the caller can anchor the generation at one or more
frames across the output video.

Validation mirrors the upstream ``WanPipelineI2V.prepare_latents`` contract
for Wan, and MiniMax-H3 FL2VA's first/last keyframe sentinels (``0``, ``-1``).
On an H3 FL2VA deployment each keyframe is also admitted against the MiniMax
input media card (``check_h3_image``: <= 30 MB, JPG/PNG/WEBP/HEIC/HEIF,
[256, 5760] px, aspect 0.4-2.5). The pipeline-level ``num_frames`` for Wan
(used by both the runner and the validators below) is the single source of
truth in ``config.constants.WAN22_NUM_FRAMES``.
"""

import os
from typing import List

from config.constants import WAN22_NUM_FRAMES, ModelNames, ModelRunners
from config.settings import get_settings
from domain.video_generate_request import VideoGenerateRequest
from pydantic import BaseModel, Field, field_validator
from tt_model_runners.minimax_h3_policy import (
    MEDIA_B64_FIELD_HEADROOM,
    MINIMAX_H3_IMAGE_MAX_BYTES,
    base64_len_for_bytes,
    check_h3_image,
    decode_base64_media,
)
from utils.image_manager import ImageManager
from utils.media_downloader import is_media_url

# One inline image is capped at the MiniMax card's 30 MB file size, measured as
# base64 text (4/3 of the bytes, plus room for a data-URL prefix). This bounds a
# field; the 64 MB request-body cap (open_ai_api/body_limit.py) bounds the request,
# so several 30 MB images cannot arrive inline -- they go by URL, as the card says.
MAX_BASE64_IMAGE_LEN = (
    base64_len_for_bytes(MINIMAX_H3_IMAGE_MAX_BYTES) + MEDIA_B64_FIELD_HEADROOM
)


class ImagePromptEntry(BaseModel):
    """One image + its frame position inside the generated video.

    ``frame_pos=-1`` is the last frame (Python list indexing). Wan's pipeline
    already resolves negatives; MiniMax-H3 FL2VA accepts only ``{0, -1}``.
    """

    image: str = Field(min_length=1, max_length=MAX_BASE64_IMAGE_LEN)
    frame_pos: int = Field(default=0, ge=-1, lt=WAN22_NUM_FRAMES)

    @field_validator("image")
    @classmethod
    def validate_decodable_image(cls, v: str) -> str:
        """Ensure the base64 string decodes to a valid PIL image via ImageManager."""

        if is_media_url(v):
            # Remote asset (e.g. presigned S3 URL): downloaded, decoded, and
            # policy-checked at the API layer before enqueue
            # (open_ai_api/video.py), where failures map to real HTTP
            # statuses instead of a blanket 422 here.
            return v

        if _is_minimax_h3_fl2va():
            # The MiniMax input media card, header-only (Image.open is lazy), so a
            # 30 MB keyframe costs milliseconds here and is decoded once, on the worker.
            check_h3_image(decode_base64_media(v), label="image")
            return v

        try:
            img = ImageManager().base64_to_pil_image(v)
        except Exception as exc:
            raise ValueError(
                "image could not be decoded to a valid PIL image "
                "(supported formats: PNG, JPEG, WebP, etc.)"
            ) from exc
        if img.size[0] < 1 or img.size[1] < 1:
            raise ValueError(
                "image has invalid dimensions (width and height must be >= 1)"
            )
        return v


class VideoI2VGenerateRequest(VideoGenerateRequest):
    """Video generation request with image conditioning (I2V / FL2VA)."""

    image_prompts: List[ImagePromptEntry] = Field(
        min_length=1, max_length=WAN22_NUM_FRAMES
    )

    @field_validator("image_prompts")
    @classmethod
    def validate_unique_frame_positions(cls, v: List[ImagePromptEntry]):
        """Duplicate frame_pos would trigger an assert inside the pipeline."""
        seen: set[int] = set()
        for entry in v:
            if entry.frame_pos in seen:
                raise ValueError(
                    f"duplicate frame_pos={entry.frame_pos} in image_prompts; "
                    "each image must target a distinct frame"
                )
            seen.add(entry.frame_pos)
        if _is_minimax_h3_fl2va():
            from tt_model_runners.minimax_h3_policy import MINIMAX_H3_FL2VA_FRAME_POS

            if len(v) > 2:
                raise ValueError(
                    "MiniMax-H3 FL2VA accepts at most two image_prompts "
                    "(frame_pos 0 = first keyframe, -1 = last keyframe)"
                )
            illegal = [
                entry.frame_pos
                for entry in v
                if entry.frame_pos not in MINIMAX_H3_FL2VA_FRAME_POS
            ]
            if illegal:
                raise ValueError(
                    "MiniMax-H3 FL2VA image_prompts[].frame_pos must be 0 "
                    f"(first) or -1 (last); got {illegal}"
                )
        return v


def _is_minimax_h3_fl2va() -> bool:
    try:
        runner = get_settings().model_runner
    except Exception:  # noqa: BLE001
        return False
    if runner == ModelRunners.TT_MINIMAX_H3_FL2VA.value:
        return True
    if runner != ModelRunners.SP_RUNNER.value:
        return False
    model_env = os.getenv("MODEL")
    if not model_env:
        return False
    try:
        return ModelNames(model_env) is ModelNames.MINIMAX_H3_FL2VA
    except ValueError:
        return False
