# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
from abc import ABC, abstractmethod

from config.settings import get_settings
from utils.logger import TTLogger
from utils.runner_utils import setup_runner_environment


class BaseDeviceRunner(ABC):
    def __init__(
        self, device_id: str, cpu_threads: str = None, num_torch_threads: int = None
    ):
        self.device_id = device_id
        self.logger = TTLogger()
        self.settings = get_settings()
        self.ttnn_device = None

        # Skip in main process when runner is only used for download_weights (device_id "-1")
        if self.device_id != "-1":
            if not cpu_threads:
                # Dynamic batcher is used for LLM workloads where VLLM performs better with higher thread counts
                cpu_threads = "16" if self.settings.use_dynamic_batcher else "2"
            if not num_torch_threads:
                num_torch_threads = 16 if self.settings.use_dynamic_batcher else 1
            setup_runner_environment(device_id, cpu_threads, num_torch_threads)

        if not os.getenv("HF_TOKEN", None) and not (
            os.getenv("HF_HOME", None) and any(os.scandir(os.getenv("HF_HOME")))
        ):
            self.logger.warning(
                "HF_TOKEN environment variable is not set and no cached models found in HF_HOME. Some models may not load properly."
            )

        # setup is tensor parallel if device mesh shape first param starts with 2
        self.is_tensor_parallel = self.settings.device_mesh_shape[0] > 1
        if self.is_tensor_parallel:
            self.logger.info(
                f"Device {self.device_id}: Tensor parallel mode enabled with mesh shape {self.settings.device_mesh_shape}"
            )

    @abstractmethod
    def warmup(self):
        pass

    def load_weights(self):
        return False

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def set_device(self):
        return {}

    def close_device(self):
        return True

    def is_request_batchable(self, request, batch=None):
        return True
