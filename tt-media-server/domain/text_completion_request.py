# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.base_request import BaseRequest
from pydantic import PrivateAttr


class TextCompletionRequest(BaseRequest):
    text: str

    _stream: bool = PrivateAttr(default=True)
