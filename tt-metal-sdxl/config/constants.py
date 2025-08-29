from enum import Enum

# MODEL environment variable
class SupportedModels(Enum):
    STABLE_DIFFUSION_XL_BASE = "stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_5_LARGE = "stable-diffusion-3.5-large"
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"
    MICROSOFT_RESNET_50 = "microsoft/resnet-50"

class ModelRunners(Enum):
    TT_SDXL_TRACE = "tt-sdxl-trace"
    TT_SD3_5 = "tt-sd3.5"
    TT_SD3_5_TRACE = "tt-sd3.5-trace"
    TT_WHISPER = "tt-whisper"
    TT_YOLOV4 = "tt-yolov4"
    FORGE = "forge"
    MOCK = "mock"

class ModelServices(Enum):
    IMAGE = "image"
    CNN = "cnn"
    AUDIO = "audio"
    TEXT = "text"

# DEVICE engvironment variable
class DeviceTypes(Enum):
    N150 = "n150"
    N300 = "n300"
    GALAXY = "galaxy"
    QUIETBOX = "quietbox"

# Combined model-device specific configurations
# useful when whole device is being used by a single model type
# also for CI testing
ModelConfigs = {
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "model_service": ModelServices.IMAGE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "0",
        "batch_size": 1,
    },
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "model_service": ModelServices.IMAGE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "0,1",
        "batch_size": 2,
    },
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.GALAXY): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "model_service": ModelServices.IMAGE.value,
        "device_mesh_shape": (1, 2),
        "device_ids": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "batch_size": 4,
    },
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.QUIETBOX): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "model_service": ModelServices.IMAGE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "0,1,2,3",
        "batch_size": 2,
    },    
    (SupportedModels.STABLE_DIFFUSION_3_5_LARGE, DeviceTypes.QUIETBOX): {
        "model_runner": ModelRunners.TT_SD3_5.value,
        "model_service": ModelServices.IMAGE.value,
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": "", #HACK to use all devices. device id split will retun and empty string to be passed to os.environ[TT_VISIBLE_DEVICES] in device_worker.py
        "batch_size": 1,
    },
    (SupportedModels.STABLE_DIFFUSION_3_5_LARGE, DeviceTypes.GALAXY): {
        "model_runner": ModelRunners.TT_SD3_5.value,
        "model_service": ModelServices.IMAGE.value,
        "device_mesh_shape": (4, 8),
        "device_ids": "", #HACK to use all devices. device id split will retun and empty string to be passed to os.environ[TT_VISIBLE_DEVICES] in device_worker.py
        "batch_size": 1,
    },
    (SupportedModels.DISTIL_WHISPER_LARGE_V3, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "model_service": ModelServices.AUDIO.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "0",
    },
    (SupportedModels.DISTIL_WHISPER_LARGE_V3, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "model_service": ModelServices.AUDIO.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "0,1",
    },
    (SupportedModels.MICROSOFT_RESNET_50, DeviceTypes.N150): {
        "model_runner": ModelRunners.FORGE.value,
        "model_service": ModelServices.CNN.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "0",
    },
    (SupportedModels.MICROSOFT_RESNET_50, DeviceTypes.N300): {
        "model_runner": ModelRunners.FORGE.value,
        "model_service": ModelServices.CNN.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "0,1",
    },
}