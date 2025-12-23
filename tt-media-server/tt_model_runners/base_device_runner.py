# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from abc import ABC, abstractmethod

from config.settings import get_settings
from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits


class BaseDeviceRunner(ABC):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        self.device_id = device_id
        self.logger = TTLogger()
        self.settings = get_settings()
        self.ttnn_device = None

        set_torch_thread_limits(num_torch_threads)

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
    def load_model(self):
        pass

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        pass

    def set_device(self):
        return {}

    def close_device(self):
        return True

    def is_request_batchable(self, request, batch=None):
        return True
