# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.base_request import BaseRequest
from pydantic import Field
from typing import Union, Optional

class TextEmbeddingRequest(BaseRequest):
    input: Union[list[str], str]
    dimensions: Optional[int] = Field(default=None, ge=0)