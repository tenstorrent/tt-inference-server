# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC, abstractmethod
import os
import ttnn
from config.settings import get_settings

from utils.logger import TTLogger

class BaseDeviceRunner(ABC):
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.logger = TTLogger()
        self.settings = get_settings()
        self.ttnn_device = None

        if not os.getenv("HF_TOKEN", None) and not (os.getenv("HF_HOME", None) and any(os.scandir(os.getenv("HF_HOME")))):
            self.logger.warning("HF_TOKEN environment variable is not set and no cached models found in HF_HOME. Some models may not load properly.")

        # setup is tensor parallel if device mesh shape first param starts with 2
        self.is_tensor_parallel = self.settings.device_mesh_shape[0] > 1
        if self.is_tensor_parallel:
            self.logger.info(f"Device {self.device_id}: Tensor parallel mode enabled with mesh shape {self.settings.device_mesh_shape}")

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def run_inference(self, *args, **kwargs):
        pass

    def get_pipeline_device_params(self):
        return None

    def set_device(self):
        if self.ttnn_device is None:
            # for now use all available devices
            self.ttnn_device = self._mesh_device()
        return self.ttnn_device

    def close_device(self):
        try:
            self.logger.info(f"Device {self.device_id}: Closing mesh device...")
            if self.ttnn_device is not None:
                ttnn.close_mesh_device(self.ttnn_device)
                self.logger.info(f"Device {self.device_id}: Successfully closed mesh device")
            else:
                self.logger.info(f"Device {self.device_id}: Device is None, no need to close")
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise RuntimeError(f"Device {self.device_id}: Device cleanup failed: {str(e)}") from e

    def get_updated_device_params(self, device_params):
        if device_params is None:
            device_params = {}

        new_device_params = device_params.copy()

        dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = new_device_params.pop("dispatch_core_type", None)

        if ttnn.device.is_blackhole() and dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
            self.logger.warning("blackhole arch does not support DispatchCoreAxis.ROW, using DispatchCoreAxis.COL instead.")
            dispatch_core_axis = ttnn.DispatchCoreAxis.COL

        dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
        new_device_params["dispatch_core_config"] = dispatch_core_config

        return new_device_params

    def _mesh_device(self):
        try:
            # Get available devices
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise RuntimeError("No TTNN devices available")
            self.logger.info(f"Device {self.device_id}: Found {len(device_ids)} available TTNN devices: {device_ids}")

            mesh_shape = ttnn.MeshShape(self.settings.device_mesh_shape)

            device_params = self.get_pipeline_device_params()
            updated_device_params = self.get_updated_device_params(device_params)
            fabric_config = self._configure_fabric(updated_device_params)
            mesh_device = self._initialize_mesh_device(mesh_shape, updated_device_params, fabric_config)

            self.logger.info(f"Device {self.device_id}: Successfully created multidevice with {mesh_device.get_num_devices()} devices")
            return mesh_device
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Unexpected error during device initialization: {e}")
            raise RuntimeError(f"Unexpected device initialization error: {str(e)}") from e

    def _configure_fabric(self, updated_device_params):
        try:
            fabric_config = updated_device_params.pop("fabric_config", ttnn.FabricConfig.FABRIC_1D)
            ttnn.set_fabric_config(fabric_config)
            return fabric_config
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Fabric configuration failed: {e}")
            raise RuntimeError(f"Fabric configuration failed: {str(e)}") from e

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        try:
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
        except Exception as e:
            try:
                if fabric_config:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception as reset_error:
                self.logger.warning(f"Device {self.device_id}: Failed to reset fabric after device initialization failure: {reset_error}")
            self.logger.error(f"Device {self.device_id}: Mesh device initialization failed: {e}")
            raise RuntimeError(f"Mesh device initialization failed: {str(e)}") from e
        return mesh_device
