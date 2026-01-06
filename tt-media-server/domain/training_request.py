# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional

from domain.base_request import BaseRequest


class TrainingRequest(BaseRequest):
    model_id: str
    dataset_id: str
    hyperparameters: dict
    job_type_specific_parameters: Optional[dict] = None
    checkpoint_config: dict
