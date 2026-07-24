# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from domain.image_to_image_request import ImageToImageRequest


class ImageEditRequest(ImageToImageRequest):
    # Optional so the shared /edits endpoint serves both mask-based edits (SDXL)
    # and instruction-only edits with no mask (FLUX.1-Kontext).
    mask: Optional[str] = None
