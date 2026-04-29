# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Image-to-Video request schema for Wan2.2 I2V.

Extends ``VideoGenerateRequest`` with a list of image prompts. Each entry
pairs a base64-encoded image with a frame position so the caller can anchor
the generation at one or more frames across the output video.

Validation mirrors the upstream ``WanPipelineI2V.prepare_latents`` contract:

``WAN22_DEFAULT_NUM_FRAMES`` must stay in sync with the ``num_frames`` value
used in ``TTWan22Runner.run`` / ``TTWan22I2VRunner.run``. If the runner
becomes parameterizable, this constant should move with it (or become a
per-request field).
"""

from typing import List

from domain.video_generate_request import VideoGenerateRequest
from pydantic import BaseModel, Field, field_validator

# Hardcoded in TTWan22Runner.run;
WAN22_DEFAULT_NUM_FRAMES = 81

# The cap exists to bound HTTP body size, not to match
# any pipeline constraint.
MAX_BASE64_IMAGE_LEN = 10_000_000


class ImagePromptEntry(BaseModel):
    """One image + its frame position inside the generated video."""

    image: str = Field(min_length=1, max_length=MAX_BASE64_IMAGE_LEN)
    frame_pos: int = Field(default=0, ge=0, lt=WAN22_DEFAULT_NUM_FRAMES)


class VideoI2VGenerateRequest(VideoGenerateRequest):
    """Video generation request with image conditioning (I2V)."""

    image_prompts: List[ImagePromptEntry] = Field(
        min_length=1, max_length=WAN22_DEFAULT_NUM_FRAMES
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
