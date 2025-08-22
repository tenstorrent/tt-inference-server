# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from utils.image_manager import ImageManager

class ImageService(BaseService):

    def post_process(self, result):
        return ImageManager("img").convert_image_to_bytes(result)

