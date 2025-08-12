# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pydantic import BaseModel, PrivateAttr

class BaseRequest(BaseModel):
    _task_id: str = PrivateAttr()