# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.image_to_image_request import ImageToImageRequest


class ImageEditRequest(ImageToImageRequest):
    mask: str
