# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.base_request import BaseRequest
from typing import List, Union

class TextEmbeddingRequest(BaseRequest):
    input: Union[str, List[str]]
