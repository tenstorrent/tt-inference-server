# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC
from uuid import uuid4

from pydantic import BaseModel, PrivateAttr


class BaseRequest(BaseModel, ABC):
    _task_id: str = PrivateAttr(default_factory=lambda: str(uuid4()))
