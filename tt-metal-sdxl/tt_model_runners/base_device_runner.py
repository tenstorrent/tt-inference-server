# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import abstractmethod

class DeviceRunner:
    device_id: str = None

    def __init__(self, device_id: str):
        self.device_id = device_id

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def runInference(self, prompt: str, num_inference_steps: int = 50, negative_prompt: str = None):
        pass

    @abstractmethod
    def close_device(self):
        pass

    @abstractmethod
    def get_device(self):
        pass

    @abstractmethod
    def get_devices(self):
        pass