# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC, abstractmethod

class BaseDeviceRunner(ABC):
    def __init__(self, device_id: str):
        self.device_id = device_id

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def close_device(self):
        pass

    @abstractmethod
    def get_device(self):
        pass