# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Image-to-Video request schema for Wan2.2 I2V.

Extends ``VideoGenerateRequest`` with a list of image prompts. Each entry
pairs an image — base64-encoded, or an http(s)/presigned URL (#4974) — with
a frame position so the caller can anchor the generation at one or more
frames across the output video.

Validation mirrors the upstream ``WanPipelineI2V.prepare_latents`` contract.
The pipeline-level ``num_frames`` (used by both the runner and the validators
below) is the single source of truth in ``config.constants.WAN22_NUM_FRAMES``.
"""

from typing import List

from config.constants import WAN22_NUM_FRAMES
from domain.video_generate_request import VideoGenerateRequest
from pydantic import BaseModel, Field, field_validator
from utils.image_manager import ImageManager
from utils.media_downloader import is_media_url

# The cap exists to bound HTTP body size, not to match
# any pipeline constraint.
MAX_BASE64_IMAGE_LEN = 10_000_000


class ImagePromptEntry(BaseModel):
    """One image + its frame position inside the generated video."""

    image: str = Field(min_length=1, max_length=MAX_BASE64_IMAGE_LEN)
    frame_pos: int = Field(default=0, ge=0, lt=WAN22_NUM_FRAMES)

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
    """Video generation request with image conditioning (I2V)."""

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
        return v
