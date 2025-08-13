# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pydantic import BaseModel
from abc import ABC, abstractmethod

class BaseRequest(BaseModel, ABC):
    def __init__(self):
        self._task_id = uuid4()

    @abstractmethod
    def get_model_input(self):
        pass