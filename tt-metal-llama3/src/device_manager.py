from enum import IntEnum, auto

import ttnn

from models.demos.t3000.llama2_70b.tt.llama_common import check_mesh_device
from conftest import get_dispatch_core_type


class DeviceType(IntEnum):
    n150 = auto()
    n300 = auto()
    t3k_mesh_device = auto()


class DeviceManager:
    def __init__(self, model_name):
        self.devices = []
        self.model_device_map = {
            "llama-3.1-8b-instruct": {DeviceType.n150, DeviceType.n300},
            "llama-3.1-8b": {DeviceType.n150, DeviceType.n300},
            "llama-3-8b-instruct": {DeviceType.n150, DeviceType.n300},
            "llama-3-8b": {DeviceType.n150, DeviceType.n300},
            "llama-3.1-70b-instruct": {DeviceType.t3k_mesh_device},
            "llama-3.1-70b": {DeviceType.t3k_mesh_device},
            "llama-3.1-70b-instruct": {DeviceType.t3k_mesh_device},
            "llama-3-70b": {DeviceType.t3k_mesh_device},
        }
        self.device_set = self.model_device_map[model_name]
        if DeviceType.n150 in self.device_set:
            self.get_device = self.get_n150_device
            self.init_device = self.init_t3k_mesh_device
            self.close_device = self.close_n150_device
            self.device_type = DeviceType.n150
        elif DeviceType.n300 in self.device_set:
            self.get_device = self.get_n150_device
            self.close_device = self.close_n150_device
            self.device_type = DeviceType.n300
        elif DeviceType.t3k_mesh_device in self.device_set:
            self.get_device = self.get_t3k_mesh_device
            self.close_device = self.close_t3k_mesh_device
            self.device_type = DeviceType.t3k_mesh_device

    def check_device_compatiblity(self, devices):
        # TODO: check that required devices are available
        pass

    def get_available_devices(self):
        # TODO
        pass

    def init_device(self, device, enable_async=False, enable_program_cache=False, **kwargs):
        if self.device_type == DeviceType.t3k_mesh_device:
            self.init_t3k_mesh_device(device, enable_async, enable_program_cache, **kwargs)
        elif self.device_type == DeviceType.n150:
            device.enable_async(enable_async)
            device.enable_program_cache(enable_program_cache)

    def get_n150_device(self, **kwargs):
        num_devices = ttnn.GetNumPCIeDevices()
        assert num_devices >= 1
        device_id = 0
        device = ttnn.CreateDevice(device_id=device_id, dispatch_core_type=get_dispatch_core_type(), **device_params)
        ttnn.SetDefaultDevice(device)
        return device
    
    def init_n150_device(self, device, **kwargs):
        pass

    def close_n150_device(self, device, **kwargs):
        ttnn.DumpDeviceProfiler(device)
        ttnn.synchronize_device(device)
        ttnn.close_device(device)

    def get_t3k_mesh_device(self, num_devices_requested, **kwargs):
        logger.info("get_t3k_mesh_device ...")
        assert ttnn.get_num_devices() == 8
        device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
        # device_params is empty dict in llama3 70B demo pytest execution
        device_params = {}
        mesh_device = ttnn.open_mesh_device(
            ttnn.MeshShape(1, num_devices_requested),
            device_ids[:num_devices_requested],
            dispatch_core_type=get_dispatch_core_type(),
            **device_params,
        )
        logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def init_t3k_mesh_device(self, mesh_device, **kwargs):
        for i in t3k_mesh_device.get_device_ids():
            device = t3k_mesh_device.get_device(i)
            device.enable_async(True)
            device.enable_program_cache()

    def close_t3k_mesh_device(self, mesh_device, **kwargs):
        for device in mesh_device.get_devices():
            device.disable_and_clear_program_cache()
            ttnn.DumpDeviceProfiler(device)
        ttnn.close_mesh_device(mesh_device)
        del mesh_device
    