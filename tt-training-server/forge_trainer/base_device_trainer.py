# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from abc import ABC, abstractmethod

from config.settings import get_settings
from utils.logger import TTLogger
from utils.torch_utils import set_torch_thread_limits

from tt_model_trainers.base_device_trainer import BaseDeviceTrainer


class BaseDeviceTrainer(ABC):
    def __init__(self, device_id: str):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass
    
    @abstractmethod
    def start_trainining(self):
        pass

    @abstractmethod
    def stop_training(self):
        pass

    @abstractmethod
    def get_training_artifacts(self):
        pass

    


