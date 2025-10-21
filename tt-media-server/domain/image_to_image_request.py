# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.base_image_request import BaseImageRequest
from typing import Optional

class ImageToImageRequest(BaseImageRequest):
    image: str  
    strength: Optional[float] = 0.3
    aesthetic_score: Optional[float] = 6.0
    negative_aesthetic_score: Optional[float] = 2.5
