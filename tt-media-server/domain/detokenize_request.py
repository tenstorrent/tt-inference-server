# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from domain.base_request import BaseRequest
from config.vllm_settings import VLLMSettings


class DetokenizeRequest(BaseRequest):
    model: str = VLLMSettings.model.value
    tokens: list[int]
