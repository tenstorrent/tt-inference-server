# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from uuid import uuid4
from pydantic import BaseModel, PrivateAttr
from abc import ABC, abstractmethod

class BaseRequest(BaseModel, ABC):
    _task_id: str = PrivateAttr(default_factory=lambda: str(uuid4()))

    @abstractmethod
    def get_model_input(self):
        pass