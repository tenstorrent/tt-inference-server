# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService

class CNNService(BaseService):

    def pre_process(self, request):
        return request.prompt