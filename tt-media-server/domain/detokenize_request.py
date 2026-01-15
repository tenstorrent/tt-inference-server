# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from config.settings import settings
from domain.base_request import BaseRequest


class DetokenizeRequest(BaseRequest):
    model: str = settings.vllm.model
    tokens: list[int]
