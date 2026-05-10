# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import List, Literal, Optional

from domain.base_request import BaseRequest
from pydantic import BaseModel, Field


class ImageFrame(BaseModel):
    image: str = Field(description="Base64-encoded image or HTTP URL")
    frame_pos: int = Field(description="Frame position in the video (0-indexed)")


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = None
    image: Optional[str] = Field(default=None, description="Base64-encoded image or HTTP URL for image-to-video (placed at frame 0)")
    image_frames: Optional[List[ImageFrame]] = Field(default=None, description="List of conditioning frames with explicit positions for I2V. Takes precedence over 'image'.")
    extract_frames: Optional[List[int]] = Field(
        default=None,
        description="Frame indices (0..N-1) to return alongside the MP4. Triggers multipart/mixed response. Example: [80] returns the last frame for video chaining.",
    )
    frame_format: Literal["webp", "png", "jpeg"] = Field(
        default="webp",
        description="Encoding format for extracted frames. 'webp' (default) and 'png' are lossless; 'jpeg' is lossy and not recommended for chaining.",
    )
