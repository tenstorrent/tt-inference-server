# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Image-to-Video request schema for Wan2.2 I2V.

Extends ``VideoGenerateRequest`` with a list of image prompts. Each entry
pairs a base64-encoded image with a frame position so the caller can anchor
the generation at one or more frames across the output video.

Validation mirrors the upstream ``WanPipelineI2V.prepare_latents`` contract.
The pipeline-level ``num_frames`` (used by both the runner and the validators
below) is the single source of truth in ``config.constants.WAN22_NUM_FRAMES``.
"""

from typing import List

from config.constants import WAN22_NUM_FRAMES
from domain.errors import ClientRequestError
from domain.video_generate_request import VideoGenerateRequest
from pydantic import BaseModel, Field, field_validator
from utils.base64_image import validate_base64_image

# The cap exists to bound HTTP body size, not to match
# any pipeline constraint.
MAX_BASE64_IMAGE_LEN = 10_000_000


class ImagePromptEntry(BaseModel):
    """One image + its frame position inside the generated video."""

    image: str = Field(min_length=1, max_length=MAX_BASE64_IMAGE_LEN)
    frame_pos: int = Field(default=0, ge=0, lt=WAN22_NUM_FRAMES)

    @field_validator("image")
    @classmethod
    def validate_image_is_decodable(cls, v: str) -> str:
        """Reject a payload that is not a base64-encoded image.

        Length alone used to be the only check here, so ``"!!!not-base64!!!"`` or
        an arbitrary blob wrapped in ``data:image/png;base64,`` was accepted, sent
        to the device, and only failed there — as a 500 to the client and an
        increment on the worker's error count, six of which restart the worker
        (#4811). ``validate_base64_image`` sniffs the header, so this stays cheap
        enough to run on every request; anything that gets past it and fails the
        real decode is caught in ``ImageManager`` and classified there.

        Re-raised as ``ValueError`` so pydantic reports it like every other field
        error on this schema and FastAPI answers 422 with the usual body. The
        underlying ``ClientRequestError`` is the right type deeper in the stack
        (it carries an HTTP status across the worker boundary) but here it would
        bypass pydantic's error collection entirely.
        """
        try:
            return validate_base64_image(v, what="image_prompts[].image")
        except ClientRequestError as e:
            raise ValueError(e.detail) from e


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
