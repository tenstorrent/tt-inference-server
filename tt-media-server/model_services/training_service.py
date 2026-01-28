# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_job_service import BaseJobService


class TrainingService(BaseJobService):
    def __init__(self):
        super().__init__()
