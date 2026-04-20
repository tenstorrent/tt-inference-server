# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from domain.base_request import BaseRequest
from pydantic import Field


class TextEmbeddingRequest(BaseRequest):
    input: str
    model: str
    dimensions: Optional[int] = Field(default=None, ge=0)
