# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from domain.base_request import BaseRequest
from pydantic import PrivateAttr

class LoraInferenceRequest(BaseRequest):
    prompt: str
    max_new_tokens: int = 64
    use_base_model: bool = False

    _adapter_path: str = PrivateAttr(default=None)