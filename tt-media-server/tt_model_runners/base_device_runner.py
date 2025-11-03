# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC, abstractmethod
import os

from utils.logger import TTLogger

class BaseDeviceRunner(ABC):
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.logger = TTLogger()

        if not os.getenv("HF_TOKEN", None) and not (os.getenv("HF_HOME", None) and any(os.scandir(os.getenv("HF_HOME")))):
            self.logger.warning("HF_TOKEN environment variable is not set and no cached models found in HF_HOME. Some models may not load properly.")

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

    def get_updated_device_params(self, device_params):
        import ttnn

        new_device_params = device_params.copy()

        dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = new_device_params.pop("dispatch_core_type", None)

        if ttnn.device.is_blackhole() and dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
            self.logger.warning("blackhole arch does not support DispatchCoreAxis.ROW, using DispatchCoreAxis.COL instead.")
            dispatch_core_axis = ttnn.DispatchCoreAxis.COL

        dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
        new_device_params["dispatch_core_config"] = dispatch_core_config

        return new_device_params