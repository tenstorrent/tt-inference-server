from enum import IntEnum, auto

import ttnn

from models.demos.t3000.llama2_70b.tt.llama_common import check_mesh_device
from conftest import get_dispatch_core_type


class DeviceType(IntEnum):
    n150 = auto()
    n300 = auto()
    t3k_mesh_device = auto()
    mock_device = auto()


class DeviceManager:
    def __init__(self, model_name, device_type=None):
        # order by default best choice
        self.default_model_device_map = {
            "llama-3.1-8b-instruct": [DeviceType.n150, DeviceType.n300],
            "llama-3.1-8b": [DeviceType.n150, DeviceType.n300],
            "llama-3-8b-instruct": [DeviceType.n150, DeviceType.n300],
            "llama-3-8b": [DeviceType.n150, DeviceType.n300],
            "llama-3.1-70b-instruct": [DeviceType.t3k_mesh_device],
            "llama-3.1-70b": [DeviceType.t3k_mesh_device],
            "llama-3.1-70b-instruct": [DeviceType.t3k_mesh_device],
            "llama-3-70b": [DeviceType.t3k_mesh_device],
        }
        self.enable_async = False
        self.enable_program_cache = False
        self.device_type = self.get_device_type(
            model_name=model_name, device_type=device_type
        )
        if self.device_type == DeviceType.n150:
            self.open_device = self.get_n150_device
            self.close_device = self.close_n150_device
        elif self.device_type == DeviceType.n300:
            # is n300 init same as n150 for llama3?
            self.open_device = self.get_n150_device
            self.close_device = self.close_n150_device
        elif self.device_type == DeviceType.t3k_mesh_device:
            self.open_device = self.get_t3k_mesh_device
            self.close_device = self.close_t3k_mesh_device
        elif self.device_type == DeviceType.mock_device:
            self.open_device = lambda *args, **kwargs: None
            self.close_device = lambda *args, **kwargs: None

    def get_device_type(self, model_name, device_type=None):
        self.compatible_devices = self.default_model_device_map[model_name]
        if device_type:
            if not device_type == DeviceType.mock_device:
                assert device_type in self.compatible_devices
            return device_type
        # TODO: add logic to check for available devices and compatibility
        return self.compatible_devices[0]

    def check_device_compatiblity(self, devices):
        # TODO: check that required devices are available
        pass

    def get_available_devices(self):
        # TODO
        pass

    def is_single_card_n300(device):
        num_pcie = ttnn.GetNumPCIeDevices()
        num_devices = ttnn.GetNumAvailableDevices()
        # N150 has 1 chip; N300 has 2 chips (1 pcie); T3000 has 8 chips (4 pcie)
        return (
            num_pcie == 1 and num_devices == 2 and device.arch().name == "WORMHOLE_B0"
        )

    def get_n150_device(self, device_params={}, **kwargs):
        num_devices = ttnn.GetNumPCIeDevices()
        assert num_devices >= 1
        device_id = 0
        device = ttnn.CreateDevice(
            device_id=device_id,
            dispatch_core_type=get_dispatch_core_type(),
            **device_params,
        )
        ttnn.SetDefaultDevice(device)
        self.init_n150_device(device, **kwargs)
        return device

    def init_n150_device(
        self, device, enable_async=False, enable_program_cache=False, param={}, **kwargs
    ):
        if enable_async:
            self.enable_async = True
            device.enable_async(param)
        if enable_program_cache:
            self.enable_program_cache = True
            device.enable_program_cache()

    def close_n150_device(
        self, device, enable_async=False, enable_program_cache=False, **kwargs
    ):
        if self.enable_async:
            device.enable_async(False)
        if self.enable_program_cache:
            device.disable_and_clear_program_cache()
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
        logger.debug(
            f"multidevice with {mesh_device.get_num_devices()} devices is created"
        )
        check_mesh_device(mesh_device, self.model_config)
        self.init_t3k_mesh_device(mesh_device, **kwargs)
        return mesh_device

    def init_t3k_mesh_device(
        self, mesh_device, enable_async=False, enable_program_cache=False, **kwargs
    ):
        for i in t3k_mesh_device.get_device_ids():
            device = t3k_mesh_device.get_device(i)
            if enable_async:
                device.enable_async(True)
            if enable_program_cache:
                device.enable_program_cache()

    def close_t3k_mesh_device(
        self, mesh_device, enable_async=False, enable_program_cache=False, **kwargs
    ):
        for device in mesh_device.get_devices():
            device.disable_and_clear_program_cache()
            ttnn.DumpDeviceProfiler(device)
        ttnn.close_mesh_device(mesh_device)
        del mesh_device
