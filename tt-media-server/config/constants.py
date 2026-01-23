# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import Enum


class SupportedModels(Enum):
    STABLE_DIFFUSION_XL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_IMG2IMG = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_INPAINTING = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    STABLE_DIFFUSION_3_5_LARGE = "stabilityai/stable-diffusion-3.5-large"
    FLUX_1_DEV = "black-forest-labs/FLUX.1-dev"
    FLUX_1_SCHNELL = "black-forest-labs/FLUX.1-schnell"
    MOTIF_IMAGE_6B_PREVIEW = "Motif-Technologies/Motif-Image-6B-Preview"
    QWEN_IMAGE = "Qwen/Qwen-Image"
    QWEN_IMAGE_2512 = "Qwen/Qwen-Image-2512"
    MOCHI_1 = "genmo/mochi-1-preview"
    WAN_2_2 = "Wan2.2-T2V-A14B-Diffusers"
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    PYANNOTE_SPEAKER_DIARIZATION = "pyannote/speaker-diarization-3.0"
    QWEN_3_EMBEDDING_4B = "Qwen/Qwen3-Embedding-4B"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
    QWEN_3_4B = "Qwen/Qwen3-4B"
    SPEECHT5_TTS = "microsoft/speecht5_tts"


# MODEL environment variable
# Model names should be unique
class ModelNames(Enum):
    STABLE_DIFFUSION_XL_BASE = "stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_IMG2IMG = "stable-diffusion-xl-base-1.0-img-2-img"
    STABLE_DIFFUSION_XL_INPAINTING = "stable-diffusion-xl-1.0-inpainting-0.1"
    STABLE_DIFFUSION_3_5_LARGE = "stable-diffusion-3.5-large"
    FLUX_1_DEV = "flux.1-dev"
    FLUX_1_SCHNELL = "flux.1-schnell"
    MOTIF_IMAGE_6B_PREVIEW = "motif-image-6b-preview"
    QWEN_IMAGE = "qwen-image"
    QWEN_IMAGE_2512 = "qwen-image-2512"
    MOCHI_1 = "mochi-1-preview"
    WAN_2_2 = "Wan2.2-T2V-A14B-Diffusers"
    DISTIL_WHISPER_LARGE_V3 = "distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "whisper-large-v3"
    MICROSOFT_RESNET_50 = "resnet-50"
    VOVNET = "vovnet"
    MOBILENETV2 = "mobilenetv2"
    EFFICIENTNET = "efficientnet"
    SEGFORMER = "segformer"
    UNET = "unet"
    VIT = "vit"
    QWEN_3_EMBEDDING_4B = "Qwen3-Embedding-4B"
    BGE_LARGE_EN_V1_5 = "bge-large-en-v1.5"
    LLAMA_3_2_3B = "Llama-3.2-3B"
    QWEN_3_4B = "Qwen3-4B"
    SPEECHT5_TTS = "speecht5-tts"


class ModelRunners(Enum):
    TT_SDXL_TRACE = "tt-sdxl-trace"
    TT_SDXL_IMAGE_TO_IMAGE = "tt-sdxl-image-to-image"
    TT_SDXL_EDIT = "tt-sdxl-edit"
    TT_SD3_5 = "tt-sd3.5"
    TT_FLUX_1_DEV = "tt-flux.1-dev"
    TT_FLUX_1_SCHNELL = "tt-flux.1-schnell"
    TT_MOTIF_IMAGE_6B_PREVIEW = "tt-motif-image-6b-preview"
    TT_QWEN_IMAGE = "tt-qwen-image"
    TT_QWEN_IMAGE_2512 = "tt-qwen-image-2512"
    TT_MOCHI_1 = "tt-mochi-1"
    TT_WAN_2_2 = "tt-wan2.2"
    TT_WHISPER = "tt-whisper"
    VLLM = "vllm"
    VLLMForge_QWEN_EMBEDDING = "vllmforge_qwen_embedding"
    VLLMBGELargeEN_V1_5 = "vllm_bge_large_en_v1_5"
    TT_XLA_RESNET = "tt-xla-resnet"
    TT_XLA_VOVNET = "tt-xla-vovnet"
    TT_XLA_MOBILENETV2 = "tt-xla-mobilenetv2"
    TT_XLA_EFFICIENTNET = "tt-xla-efficientnet"
    TT_XLA_SEGFORMER = "tt-xla-segformer"
    TT_XLA_UNET = "tt-xla-unet"
    TT_XLA_VIT = "tt-xla-vit"
    LORA_TRAINER = "lora_trainer"
    MOCK = "mock"
    LLM_TEST = "llm_test"
    TT_SPEECHT5_TTS = "tt-speecht5-tts"


class ModelServices(Enum):
    IMAGE = "image"
    LLM = "llm"
    CNN = "cnn"
    AUDIO = "audio"
    VIDEO = "video"
    TRAINING = "training"
    TEXT_TO_SPEECH = "text_to_speech"


MODEL_SERVICE_RUNNER_MAP = {
    ModelServices.IMAGE: {
        ModelRunners.TT_SDXL_EDIT,
        ModelRunners.TT_SDXL_IMAGE_TO_IMAGE,
        ModelRunners.TT_SDXL_TRACE,
        ModelRunners.TT_SD3_5,
        ModelRunners.TT_FLUX_1_DEV,
        ModelRunners.TT_FLUX_1_SCHNELL,
        ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW,
        ModelRunners.TT_QWEN_IMAGE,
        ModelRunners.TT_QWEN_IMAGE_2512,
    },
    ModelServices.LLM: {
        ModelRunners.VLLM,
        ModelRunners.VLLMForge_QWEN_EMBEDDING,
        ModelRunners.VLLMBGELargeEN_V1_5,
        ModelRunners.LLM_TEST,
    },
    ModelServices.CNN: {
        ModelRunners.TT_XLA_RESNET,
        ModelRunners.TT_XLA_VOVNET,
        ModelRunners.TT_XLA_MOBILENETV2,
        ModelRunners.TT_XLA_EFFICIENTNET,
        ModelRunners.TT_XLA_SEGFORMER,
        ModelRunners.TT_XLA_UNET,
        ModelRunners.TT_XLA_VIT,
    },
    ModelServices.AUDIO: {
        ModelRunners.TT_WHISPER,
    },
    ModelServices.VIDEO: {
        ModelRunners.TT_MOCHI_1,
        ModelRunners.TT_WAN_2_2,
    },
    ModelServices.TRAINING: {
        ModelRunners.LORA_TRAINER,
    },
    ModelServices.TEXT_TO_SPEECH: {
        ModelRunners.TT_SPEECHT5_TTS,
    },
}


MODEL_RUNNER_TO_MODEL_NAMES_MAP = {
    ModelRunners.TT_SDXL_EDIT: {ModelNames.STABLE_DIFFUSION_XL_INPAINTING},
    ModelRunners.TT_SDXL_IMAGE_TO_IMAGE: {ModelNames.STABLE_DIFFUSION_XL_IMG2IMG},
    ModelRunners.TT_SDXL_TRACE: {ModelNames.STABLE_DIFFUSION_XL_BASE},
    ModelRunners.TT_SD3_5: {ModelNames.STABLE_DIFFUSION_3_5_LARGE},
    ModelRunners.TT_FLUX_1_DEV: {ModelNames.FLUX_1_DEV},
    ModelRunners.TT_FLUX_1_SCHNELL: {ModelNames.FLUX_1_SCHNELL},
    ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW: {ModelNames.MOTIF_IMAGE_6B_PREVIEW},
    ModelRunners.TT_QWEN_IMAGE: {ModelNames.QWEN_IMAGE},
    ModelRunners.TT_QWEN_IMAGE_2512: {ModelNames.QWEN_IMAGE_2512},
    ModelRunners.TT_MOCHI_1: {ModelNames.MOCHI_1},
    ModelRunners.TT_WAN_2_2: {ModelNames.WAN_2_2},
    ModelRunners.TT_WHISPER: {
        ModelNames.OPENAI_WHISPER_LARGE_V3,
        ModelNames.DISTIL_WHISPER_LARGE_V3,
    },
    ModelRunners.TT_XLA_RESNET: {ModelNames.MICROSOFT_RESNET_50},
    ModelRunners.TT_XLA_VOVNET: {ModelNames.VOVNET},
    ModelRunners.TT_XLA_MOBILENETV2: {ModelNames.MOBILENETV2},
    ModelRunners.TT_XLA_EFFICIENTNET: {ModelNames.EFFICIENTNET},
    ModelRunners.TT_XLA_SEGFORMER: {ModelNames.SEGFORMER},
    ModelRunners.TT_XLA_UNET: {ModelNames.UNET},
    ModelRunners.TT_XLA_VIT: {ModelNames.VIT},
    ModelRunners.VLLMForge_QWEN_EMBEDDING: {ModelNames.QWEN_3_EMBEDDING_4B},
    ModelRunners.VLLMBGELargeEN_V1_5: {ModelNames.BGE_LARGE_EN_V1_5},
    ModelRunners.VLLM: {ModelNames.LLAMA_3_2_3B, ModelNames.QWEN_3_4B},
    ModelRunners.TT_SPEECHT5_TTS: {ModelNames.SPEECHT5_TTS},
}


# DEVICE environment variable
class DeviceTypes(Enum):
    N150 = "n150"
    N300 = "n300"
    GALAXY = "galaxy"
    T3K = "t3k"
    QBGE = "qbge"
    P300 = "p300"


class DeviceIds(Enum):
    DEVICE_IDS_1 = "(0)"
    DEVICE_IDS_2 = "(0),(1)"
    DEVICE_IDS_4 = "(0),(1),(2),(3)"
    DEVICE_IDS_16 = (
        "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)"
    )
    DEVICE_IDS_32 = "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)"
    DEVICE_IDS_ALL = ""  # HACK to use all devices. device id split will return and empty string to be passed to os.environ[TT_VISIBLE_DEVICES] in device_worker.py


class AudioTasks(Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class ResponseFormat(Enum):
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"
    AUDIO = "audio"


class JobTypes(Enum):
    VIDEO = "video"
    TRAINING = "training"


# Combined model-device specific configurations
# useful when whole device is being used by a single model type
# also for CI testing
ModelConfigs = {
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_16.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
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
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
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
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_SD3_5, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.QBGE): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P300): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
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
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.QBGE): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P300): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE_2512, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE_2512, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
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
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.QBGE): {
        "device_mesh_shape": (1, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMBGELargeEN_V1_5, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 8,
        "default_throttle_level": 0,
    },
    (ModelRunners.VLLMBGELargeEN_V1_5, DeviceTypes.N300): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
        "max_batch_size": 16,
        "default_throttle_level": 0,
    },
    (ModelRunners.VLLMBGELargeEN_V1_5, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 16,
        "default_throttle_level": 0,
    },
    (ModelRunners.VLLMBGELargeEN_V1_5, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 8,
        "default_throttle_level": 0,
    },
}

for runner in [
    ModelRunners.TT_XLA_RESNET,
    ModelRunners.TT_XLA_VOVNET,
    ModelRunners.TT_XLA_MOBILENETV2,
    ModelRunners.TT_XLA_EFFICIENTNET,
    ModelRunners.TT_XLA_SEGFORMER,
    ModelRunners.TT_XLA_UNET,
    ModelRunners.TT_XLA_VIT,
]:
    ModelConfigs[(runner, DeviceTypes.N150)] = {
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
    }
    ModelConfigs[(runner, DeviceTypes.N300)] = {
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": DeviceIds.DEVICE_IDS_ALL.value,
    }
