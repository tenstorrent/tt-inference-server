# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from domain.image_generate_request import ImageGenerateRequest
from pydantic import Field


class ImageToImageRequest(ImageGenerateRequest):
    image: str
    strength: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
    """ TODO: Reintroduce these fields when https://github.com/tenstorrent/tt-metal/issues/31032 is resolved
    aesthetic_score: Optional[float] = 6.0
    negative_aesthetic_score: Optional[float] = 2.5
    """
