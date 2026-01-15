# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from domain.base_request import BaseRequest
from typing import Optional
from pydantic import model_validator
from config.settings import settings

class TrainingRequest(BaseRequest):
    hyperparameters: dict
    dataset: Optional[str] = None

