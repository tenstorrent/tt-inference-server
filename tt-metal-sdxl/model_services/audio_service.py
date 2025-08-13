# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from utils.image_manager import ImageManager

class AudioService(BaseService):

    def post_processing(self, result):
        return result # to implement