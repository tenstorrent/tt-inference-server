from enum import Enum

# MODEL environment variable
class SupportedModels(Enum):
    STABLE_DIFFUSION_XL_BASE = "stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_5_LARGE = "stable-diffusion-3.5-large"
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    MICROSOFT_RESNET_50 = "microsoft/resnet-50"
    VOVNET = "vovnet"
    MOBILENETV2 = "mobilenetv2"

class ModelRunners(Enum):
    TT_SDXL_TRACE = "tt-sdxl-trace"
    TT_SD3_5 = "tt-sd3.5"
    TT_WHISPER = "tt-whisper"
    TT_YOLOV4 = "tt-yolov4"
    TT_XLA_RESNET = "tt-xla-resnet"
    TT_XLA_VOVNET = "tt-xla-vovnet"
    TT_XLA_MOBILENETV2 = "tt-xla-mobilenetv2"
    MOCK = "mock"

class ModelServices(Enum):
    IMAGE = "image"
    CNN = "cnn"
    AUDIO = "audio"

MODEL_SERVICE_RUNNER_MAP = {
    ModelServices.IMAGE: {ModelRunners.TT_SDXL_TRACE, ModelRunners.TT_SD3_5},
    ModelServices.AUDIO: {ModelRunners.TT_WHISPER},
    ModelServices.CNN: {
        ModelRunners.TT_XLA_RESNET, 
        ModelRunners.TT_XLA_VOVNET,
        ModelRunners.TT_XLA_MOBILENETV2,
        ModelRunners.TT_YOLOV4},
}

# DEVICE engvironment variable
class DeviceTypes(Enum):
    N150 = "n150"
    N300 = "n300"
    GALAXY = "galaxy"
    T3K = "t3k"

# Combined model-device specific configurations
# useful when whole device is being used by a single model type
# also for CI testing
ModelConfigs = {
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0)",
        "max_batch_size": 1,
    },
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0),(1)",
        "max_batch_size": 2,
    },
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.GALAXY): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)",
        "max_batch_size": 1,
    },
    (SupportedModels.STABLE_DIFFUSION_XL_BASE, DeviceTypes.T3K): {
        "model_runner": ModelRunners.TT_SDXL_TRACE.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0),(1),(2),(3)",
        "max_batch_size": 2,
    },    
    (SupportedModels.STABLE_DIFFUSION_3_5_LARGE, DeviceTypes.T3K): {
        "model_runner": ModelRunners.TT_SD3_5.value,
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": "", #HACK to use all devices. device id split will retun and empty string to be passed to os.environ[TT_VISIBLE_DEVICES] in device_worker.py
        "max_batch_size": 1
    },
    (SupportedModels.STABLE_DIFFUSION_3_5_LARGE, DeviceTypes.GALAXY): {
        "model_runner": ModelRunners.TT_SD3_5.value,
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": "", #HACK to use all devices. device id split will retun and empty string to be passed to os.environ[TT_VISIBLE_DEVICES] in device_worker.py
        "max_batch_size": 1
    },
    (SupportedModels.DISTIL_WHISPER_LARGE_V3, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0)",
        "max_batch_size": 1,
    },
    (SupportedModels.DISTIL_WHISPER_LARGE_V3, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0),(1)",
        "max_batch_size": 1,
    },
    (SupportedModels.DISTIL_WHISPER_LARGE_V3, DeviceTypes.GALAXY): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)",
        "max_batch_size": 1,
    },
    (SupportedModels.DISTIL_WHISPER_LARGE_V3, DeviceTypes.T3K): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0),(1),(2),(3)",
        "max_batch_size": 1,
    },
    (SupportedModels.OPENAI_WHISPER_LARGE_V3, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0)",
        "max_batch_size": 1,
    },
    (SupportedModels.OPENAI_WHISPER_LARGE_V3, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0),(1)",
        "max_batch_size": 1,
    },
    (SupportedModels.OPENAI_WHISPER_LARGE_V3, DeviceTypes.GALAXY): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)",
        "max_batch_size": 1,
    },
    (SupportedModels.OPENAI_WHISPER_LARGE_V3, DeviceTypes.T3K): {
        "model_runner": ModelRunners.TT_WHISPER.value,
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": "(0),(1),(2),(3)",
        "max_batch_size": 1,
    },
    (SupportedModels.MICROSOFT_RESNET_50, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_XLA_RESNET.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "(0)",
    },
    (SupportedModels.MICROSOFT_RESNET_50, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_XLA_RESNET.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "(0),(1)",
    },
    (SupportedModels.VOVNET, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_XLA_VOVNET.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "(0)",
    },
    (SupportedModels.VOVNET, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_XLA_VOVNET.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "(0),(1)",
    },
    (SupportedModels.MOBILENETV2, DeviceTypes.N150): {
        "model_runner": ModelRunners.TT_XLA_MOBILENETV2.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "(0)",
    },
    (SupportedModels.MOBILENETV2, DeviceTypes.N300): {
        "model_runner": ModelRunners.TT_XLA_MOBILENETV2.value,
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": "(0),(1)",
    },
}
