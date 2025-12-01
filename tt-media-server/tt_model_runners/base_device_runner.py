# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from abc import ABC, abstractmethod

import torch

from config.settings import get_settings
from utils.logger import TTLogger


class BaseDeviceRunner(ABC):
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.logger = TTLogger()
        self.settings = get_settings()
        self.ttnn_device = None

        # Limit the number of threads torch can create in order to avoid thread explosion when running multi-process scenarios (such as 32 processes on a galaxy).
        # This way, torch can create only one thread per process, instead of predefined number of them (32).
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)

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

    def get_pipeline_device_params(self):
        return None

    def set_device(self):
        return {}

    def close_device(self):
        return True

    def get_updated_device_params(self, device_params):
        return None

    def _mesh_device(self):
        return None

    def _configure_fabric(self, updated_device_params):
        return None

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        return None
