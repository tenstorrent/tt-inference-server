# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Union, Optional

from domain.base_request import BaseRequest
from pydantic import Field

class TextEmbeddingRequest(BaseRequest):
    input: Union[list[str], str]
    dimensions: Optional[int] = Field(default=None, ge=0)
