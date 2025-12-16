# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from abc import ABC, abstractmethod

from config.settings import get_settings
from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits

from tt_model_trainers.base_device_trainer import BaseDeviceTrainer


class BlacksmithTrainer(BaseDeviceTrainer):
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.logger = TTLogger()
        self.settings = get_settings()
        self.ttnn_device = None

        set_torch_thread_limits()

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
    def get_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def start_training(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop_training(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_artifacts(self, *args, **kwargs):
        pass

    def set_device(self):
        return {}

    def close_device(self):
        return True
