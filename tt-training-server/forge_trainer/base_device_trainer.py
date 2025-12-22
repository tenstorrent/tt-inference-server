# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from abc import ABC, abstractmethod


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

    


