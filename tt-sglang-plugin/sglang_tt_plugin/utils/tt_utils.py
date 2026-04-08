# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import logging
import os
from abc import ABC

import torch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class BaseMetalDeviceRunner(ABC):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        self.device_id = device_id
        self.logger = logger
        self.ttnn_device = None

        self.set_torch_thread_limits(num_torch_threads)

        if not os.getenv("HF_TOKEN", None) and not (
            os.getenv("HF_HOME", None) and any(os.scandir(os.getenv("HF_HOME")))
        ):
            self.logger.warning(
                "HF_TOKEN environment variable is not set and no cached models found in HF_HOME. Some models may not load properly."
            )

        self.is_tensor_parallel = False

    def get_pipeline_device_params(self):
        """Return device params including trace_region_size for tracing support."""
        return {
            "trace_region_size": 50000000,  # ~50MB for trace buffers
        }

    def set_device(self):
        if self.ttnn_device is None:
            # Worker environment setup (TT_VISIBLE_DEVICES, TT_METAL_CACHE, TT_CACHE_HOME)
            # is done in TTModels.__init__ BEFORE this is called, to ensure it runs
            # before any tt-metal imports that might read cache paths

            # Now open device - will see only the devices assigned to this worker
            self.ttnn_device = self._mesh_device()
        server_args = get_global_server_args()
        self.max_batch_size = server_args.max_running_requests or 32
        return self.ttnn_device

    def close_device(self):
        import ttnn

        try:
            self.logger.info(f"Device {self.device_id}: Closing mesh device...")
            if self.ttnn_device is not None:
                ttnn.close_mesh_device(self.ttnn_device)
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed mesh device"
                )
            else:
                self.logger.info(
                    f"Device {self.device_id}: Device is None, no need to close"
                )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Device cleanup failed: {str(e)}"
            ) from e

    def get_updated_device_params(self, device_params):
        import ttnn

        if device_params is None:
            device_params = {}

        new_device_params = device_params.copy()

        dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
        fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

        if ttnn.device.is_blackhole():
            # Only when both fabric_config and fabric_tensix_config are set, we can use ROW dispatch, otherwise force to use COL dispatch
            fabric_config = new_device_params.get("fabric_config", None)
            if not (fabric_config and fabric_tensix_config):
                # When not both are set, force COL dispatch
                if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                    self.logger.warning(
                        "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                    )
                    dispatch_core_axis = ttnn.DispatchCoreAxis.COL
            elif fabric_config and fabric_tensix_config:
                self.logger.warning(
                    f"Blackhole with fabric_config and fabric_tensix_config enabled, using fabric_tensix_config={fabric_tensix_config}"
                )

        dispatch_core_config = ttnn.DispatchCoreConfig(
            dispatch_core_type, dispatch_core_axis, fabric_tensix_config
        )
        new_device_params["dispatch_core_config"] = dispatch_core_config

        return new_device_params

    def _mesh_device(self):
        import ttnn

        try:
            # Get available devices
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise RuntimeError("No TTNN devices available")

            num_devices_available = len(device_ids)
            self.logger.info(
                f"Device {self.device_id}: Found {num_devices_available} available TTNN devices: {device_ids}"
            )

            # Get mesh shape from DEVICE_MESH_SHAPE env var (set by --mesh-shape CLI arg)
            mesh_shape_str = os.environ.get("DEVICE_MESH_SHAPE")
            if mesh_shape_str:
                rows, cols = map(int, mesh_shape_str.split(","))
                mesh_shape = ttnn.MeshShape(rows, cols)
                num_devices_requested = rows * cols
                self.logger.info(
                    f"Device {self.device_id}: Using mesh shape ({rows}, {cols}) from --mesh-shape CLI arg"
                )
            else:
                # Default: 1 row, N columns (width sharding for LLMs)
                mesh_shape = ttnn.MeshShape(1, num_devices_available)
                num_devices_requested = num_devices_available
                self.logger.info(
                    f"Device {self.device_id}: Using default mesh shape (1, {num_devices_available})"
                )

            # Configure fabric BEFORE opening mesh device
            # Use requested device count, not available - fabric requires all devices be active
            fabric_config = self._configure_fabric(num_devices_requested)

            device_params = self.get_pipeline_device_params()
            updated_device_params = self.get_updated_device_params(device_params)
            mesh_device = self._initialize_mesh_device(
                mesh_shape, updated_device_params, fabric_config
            )

            self.logger.info(
                f"Device {self.device_id}: Successfully created multidevice with {mesh_device.get_num_devices()} devices"
            )
            return mesh_device
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Unexpected error during device initialization: {e}"
            )
            raise RuntimeError(
                f"Unexpected device initialization error: {str(e)}"
            ) from e

    def _configure_fabric(self, num_devices: int):
        """Configure fabric before opening mesh device.

        For multi-device setups (N300 with 2 chips), we need FABRIC_1D
        for CCL operations (all_gather, reduce_scatter).

        Each worker process has its own MetalContext, and TT_VISIBLE_DEVICES
        restricts which physical devices it can see. This is the same approach
        used by tt-sglang-plugin in tt-inference-server.
        """
        import ttnn

        if num_devices == 1:
            self.logger.info(
                f"Device {self.device_id}: Single device, no fabric config needed"
            )
            return None

        # FABRIC_1D for N300 (linear topology with 2 chips connected via Ethernet)
        fabric_config = ttnn.FabricConfig.FABRIC_1D

        self.logger.info(
            f"Device {self.device_id}: Setting fabric config to {fabric_config} for {num_devices} devices"
        )

        ttnn.set_fabric_config(fabric_config)

        return fabric_config

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        import ttnn

        try:
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Mesh device initialization failed: {e}"
            )
            raise RuntimeError(f"Mesh device initialization failed: {str(e)}") from e
        return mesh_device

    def set_torch_thread_limits(self, num_threads: int = 1):
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)
        if torch.get_num_interop_threads() != num_threads:
            torch.set_num_interop_threads(num_threads)
