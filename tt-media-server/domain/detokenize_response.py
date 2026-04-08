# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pydantic import BaseModel


class DetokenizeResponse(BaseModel):
    prompt: str
