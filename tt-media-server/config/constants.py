from enum import Enum

class SupportedModels(Enum):
    STABLE_DIFFUSION_XL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_IMG2IMG = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_5_LARGE = "stabilityai/stable-diffusion-3.5-large"
    FLUX_1_DEV = "black-forest-labs/FLUX.1-dev"
    FLUX_1_SCHNELL = "black-forest-labs/FLUX.1-schnell"
    MOCHI_1 = "genmo/mochi-1-preview"
    WAN_2_2 = "Wan2.2-T2V-A14B-Diffusers"
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    PYANNOTE_SPEAKER_DIARIZATION = "pyannote/speaker-diarization-3.0"

# MODEL environment variable
class ModelNames(Enum):
    STABLE_DIFFUSION_XL_BASE = "stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_IMG2IMG = "stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_3_5_LARGE = "stable-diffusion-3.5-large"
    FLUX_1_DEV = "flux.1-dev"
    FLUX_1_SCHNELL = "flux.1-schnell"
    MOCHI_1 = "mochi-1-preview"
    WAN_2_2 = "Wan2.2-T2V-A14B-Diffusers"
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "openai-whisper-large-v3"
    MICROSOFT_RESNET_50 = "microsoft/resnet-50"
    VOVNET = "vovnet"
    MOBILENETV2 = "mobilenetv2"

class ModelRunners(Enum):
    TT_SDXL_TRACE = "tt-sdxl-trace"
    TT_SDXL_IMAGE_TO_IMAGE = "tt-sdxl-image-to-image"
    TT_SD3_5 = "tt-sd3.5"
    TT_FLUX_1_DEV = "tt-flux.1-dev"
    TT_FLUX_1_SCHNELL = "tt-flux.1-schnell"
    TT_MOCHI_1 = "tt-mochi-1"
    TT_WAN_2_2 = "tt-wan2.2"
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
    ModelServices.IMAGE: {
        ModelRunners.TT_SDXL_IMAGE_TO_IMAGE,
        ModelRunners.TT_SDXL_TRACE,
        ModelRunners.TT_SD3_5,
        ModelRunners.TT_FLUX_1_DEV,
        ModelRunners.TT_FLUX_1_SCHNELL,
        ModelRunners.TT_MOCHI_1,
        ModelRunners.TT_WAN_2_2,
    },
    ModelServices.AUDIO: {
        ModelRunners.TT_WHISPER
    },
    ModelServices.CNN: {
        ModelRunners.TT_XLA_RESNET,
        ModelRunners.TT_XLA_VOVNET,
        ModelRunners.TT_XLA_MOBILENETV2,
        ModelRunners.TT_YOLOV4},
}

MODEL_RUNNER_TO_MODEL_NAMES_MAP = {
    ModelRunners.TT_SDXL_IMAGE_TO_IMAGE: {
        ModelNames.STABLE_DIFFUSION_XL_IMG2IMG
    },
    ModelRunners.TT_SDXL_TRACE: {
        ModelNames.STABLE_DIFFUSION_XL_BASE
    },
    ModelRunners.TT_SD3_5: {
        ModelNames.STABLE_DIFFUSION_3_5_LARGE
    },
    ModelRunners.TT_FLUX_1_DEV: {
        ModelNames.FLUX_1_DEV
    },
    ModelRunners.TT_FLUX_1_SCHNELL: {
        ModelNames.FLUX_1_SCHNELL
    },
    ModelRunners.TT_MOCHI_1: {
        ModelNames.MOCHI_1
    },
    ModelRunners.TT_WAN_2_2: {
        ModelNames.WAN_2_2
    },
    ModelRunners.TT_WHISPER: {
        ModelNames.DISTIL_WHISPER_LARGE_V3,
        ModelNames.OPENAI_WHISPER_LARGE_V3,
    },
    ModelRunners.TT_XLA_RESNET: {
        ModelNames.MICROSOFT_RESNET_50
    },
    ModelRunners.TT_XLA_VOVNET: {
        ModelNames.VOVNET
    },
    ModelRunners.TT_XLA_MOBILENETV2: {
        ModelNames.MOBILENETV2
    },
}

# DEVICE environment variable
class DeviceTypes(Enum):
    N150 = "n150"
    N300 = "n300"
    GALAXY = "galaxy"
    T3K = "t3k"

class DeviceIds(Enum):
    DEVICE_IDS_1 = "(0)"
    DEVICE_IDS_2 = "(0),(1)"
    DEVICE_IDS_4 = "(0),(1),(2),(3)"
    DEVICE_IDS_16 = (
        "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)"
    )
    DEVICE_IDS_32 = "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)"
    DEVICE_IDS_ALL = "" #HACK to use all devices. device id split will return and empty string to be passed to os.environ[TT_VISIBLE_DEVICES] in device_worker.py


# Combined model-device specific configurations
# useful when whole device is being used by a single model type
# also for CI testing
ModelConfigs = {
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_16.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_16.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SD3_5, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1
    },
    (ModelRunners.TT_SD3_5, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_16.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    }
}

for runner in [ModelRunners.TT_XLA_RESNET,ModelRunners.TT_XLA_VOVNET,ModelRunners.TT_XLA_MOBILENETV2]:
    ModelConfigs[(runner, DeviceTypes.N150)] = {
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
    }
    ModelConfigs[(runner, DeviceTypes.N300)] = {
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
    }
