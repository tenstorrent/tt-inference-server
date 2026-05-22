# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from model_services.base_service import BaseService


class CNNService(BaseService):
    def __init__(self):
        super().__init__()
