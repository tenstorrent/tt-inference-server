# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from enum import Enum


class SupportedModels(Enum):
    STABLE_DIFFUSION_XL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_512 = "hotshotco/SDXL-512"
    STABLE_DIFFUSION_XL_IMG2IMG = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_INPAINTING = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    STABLE_DIFFUSION_3_5_LARGE = "stabilityai/stable-diffusion-3.5-large"
    FLUX_1_DEV = "black-forest-labs/FLUX.1-dev"
    FLUX_1_SCHNELL = "black-forest-labs/FLUX.1-schnell"
    MOTIF_IMAGE_6B_PREVIEW = "Motif-Technologies/Motif-Image-6B-Preview"
    QWEN_IMAGE = "Qwen/Qwen-Image"
    QWEN_IMAGE_2512 = "Qwen/Qwen-Image-2512"
    MOCHI_1 = "genmo/mochi-1-preview"
    WAN_2_2 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    DISTIL_WHISPER_LARGE_V3 = "distil-whisper/distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    PYANNOTE_SPEAKER_DIARIZATION = "pyannote/speaker-diarization-3.0"
    QWEN_3_EMBEDDING_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN_3_EMBEDDING_8B = "Qwen/Qwen3-Embedding-8B"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
    BGE_M3 = "BAAI/bge-m3"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"
    LLAMA_3_2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA_3_1_70B = "meta-llama/Llama-3.1-70B"
    QWEN_3_4B = "Qwen/Qwen3-4B"
    QWEN_3_8B = "Qwen/Qwen3-8B"
    SPEECHT5_TTS = "microsoft/speecht5_tts"
    GEMMA_1_1_2B_IT = "google/gemma-1.1-2b-it"


# MODEL environment variable
# Model names should be unique
class ModelNames(Enum):
    STABLE_DIFFUSION_XL_BASE = "stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_XL_512 = "SDXL-512"
    STABLE_DIFFUSION_XL_IMG2IMG = "stable-diffusion-xl-base-1.0-img-2-img"
    STABLE_DIFFUSION_XL_INPAINTING = "stable-diffusion-xl-1.0-inpainting-0.1"
    STABLE_DIFFUSION_3_5_LARGE = "stable-diffusion-3.5-large"
    FLUX_1_DEV = "FLUX.1-dev"
    FLUX_1_SCHNELL = "FLUX.1-schnell"
    MOTIF_IMAGE_6B_PREVIEW = "Motif-Image-6B-Preview"
    QWEN_IMAGE = "Qwen-Image"
    QWEN_IMAGE_2512 = "Qwen-Image-2512"
    MOCHI_1 = "mochi-1-preview"
    WAN_2_2 = "Wan2.2-T2V-A14B-Diffusers"
    DISTIL_WHISPER_LARGE_V3 = "distil-large-v3"
    OPENAI_WHISPER_LARGE_V3 = "whisper-large-v3"
    MICROSOFT_RESNET_50 = "resnet-50"
    VOVNET = "vovnet"
    MOBILENETV2 = "mobilenetv2"
    YOLOV4 = "yolov4"
    EFFICIENTNET = "efficientnet"
    SEGFORMER = "segformer"
    UNET = "unet"
    VIT = "vit"
    QWEN_3_EMBEDDING_4B = "Qwen3-Embedding-4B"
    QWEN_3_EMBEDDING_8B = "Qwen3-Embedding-8B"
    BGE_LARGE_EN_V1_5 = "bge-large-en-v1.5"
    BGE_M3 = "bge-m3"
    LLAMA_3_2_3B = "Llama-3.2-3B"
    LLAMA_3_1_8B = "Llama-3.1-8B"
    LLAMA_3_2_3B_INSTRUCT = "Llama-3.2-3B-Instruct"
    LLAMA_3_1_8B_INSTRUCT = "Llama-3.1-8B-Instruct"
    LLAMA_3_1_70B = "Llama-3.1-70B"
    QWEN_3_4B = "Qwen3-4B"
    QWEN_3_8B = "Qwen3-8B"
    SPEECHT5_TTS = "speecht5_tts"
    GEMMA_1_1_2B_IT = "gemma-1.1-2b-it"


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
    VLLMForge = "vllm_forge"
    TT_YOLOV4 = "tt-yolov4"
    VLLMForge_QWEN_EMBEDDING = "vllmforge_qwen_embedding"
    VLLMForge_LLAMA_70B = "vllm_forge_llama_70b"
    QWEN_EMBEDDING_8B = "qwen_embedding_8b"
    BGELargeEN_V1_5 = "bge_large_en_v1_5"
    BGEM3 = "bge-m3"
    TT_XLA_RESNET = "tt-xla-resnet"
    TT_XLA_VOVNET = "tt-xla-vovnet"
    TT_XLA_MOBILENETV2 = "tt-xla-mobilenetv2"
    TT_XLA_EFFICIENTNET = "tt-xla-efficientnet"
    TT_XLA_SEGFORMER = "tt-xla-segformer"
    TT_XLA_UNET = "tt-xla-unet"
    TT_XLA_VIT = "tt-xla-vit"
    TRAINING_LLAMA_LORA = "training-llama-lora"
    TRAINING_GEMMA_LORA = "training-gemma-lora"
    LORA_SINGLE_CHIP = "lora-single-chip"
    MOCK = "mock"
    SP_RUNNER = "sp_runner"
    LLM_TEST = "llm_test"
    LLAMA_RUNNER = "llama_runner"
    TT_SPEECHT5_TTS = "tt-speecht5-tts"
    TT_XLA_SDXL = "tt-xla-sdxl"


class ModelServices(Enum):
    IMAGE = "image"
    LLM = "llm"
    CNN = "cnn"
    AUDIO = "audio"
    VIDEO = "video"
    TRAINING = "training"
    TEXT_TO_SPEECH = "text_to_speech"
    EMBEDDING = "embedding"


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
        ModelRunners.TT_XLA_SDXL,
    },
    ModelServices.LLM: {
        ModelRunners.VLLMForge,
        ModelRunners.VLLMForge_LLAMA_70B,
        ModelRunners.LLM_TEST,
        ModelRunners.LLAMA_RUNNER,
        ModelRunners.LORA_SINGLE_CHIP,
    },
    ModelServices.EMBEDDING: {
        ModelRunners.VLLMForge_QWEN_EMBEDDING,
        ModelRunners.QWEN_EMBEDDING_8B,
        ModelRunners.BGELargeEN_V1_5,
        ModelRunners.BGEM3,
    },
    ModelServices.CNN: {
        ModelRunners.TT_XLA_RESNET,
        ModelRunners.TT_XLA_VOVNET,
        ModelRunners.TT_XLA_MOBILENETV2,
        ModelRunners.TT_XLA_EFFICIENTNET,
        ModelRunners.TT_XLA_SEGFORMER,
        ModelRunners.TT_XLA_UNET,
        ModelRunners.TT_XLA_VIT,
        ModelRunners.TT_YOLOV4,
    },
    ModelServices.AUDIO: {
        ModelRunners.TT_WHISPER,
    },
    ModelServices.VIDEO: {
        ModelRunners.TT_MOCHI_1,
        ModelRunners.TT_WAN_2_2,
        ModelRunners.SP_RUNNER,
    },
    ModelServices.TRAINING: {
        ModelRunners.TRAINING_GEMMA_LORA,
        ModelRunners.TRAINING_LLAMA_LORA,
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
    ModelRunners.SP_RUNNER: {ModelNames.WAN_2_2, ModelNames.MOCHI_1},
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
    ModelRunners.VLLMForge_LLAMA_70B: {ModelNames.LLAMA_3_1_70B},
    ModelRunners.QWEN_EMBEDDING_8B: {ModelNames.QWEN_3_EMBEDDING_8B},
    ModelRunners.BGELargeEN_V1_5: {ModelNames.BGE_LARGE_EN_V1_5},
    ModelRunners.BGEM3: {ModelNames.BGE_M3},
    ModelRunners.VLLMForge: {
        ModelNames.LLAMA_3_2_3B,
        ModelNames.LLAMA_3_2_3B_INSTRUCT,
        ModelNames.LLAMA_3_1_8B_INSTRUCT,
        ModelNames.QWEN_3_4B,
        ModelNames.QWEN_3_8B,
    },
    ModelRunners.TT_SPEECHT5_TTS: {ModelNames.SPEECHT5_TTS},
    ModelRunners.TRAINING_GEMMA_LORA: {ModelNames.GEMMA_1_1_2B_IT},
    ModelRunners.TRAINING_LLAMA_LORA: {ModelNames.LLAMA_3_1_8B},
    ModelRunners.LORA_SINGLE_CHIP: {ModelNames.GEMMA_1_1_2B_IT},
    ModelRunners.TT_XLA_SDXL: {
        ModelNames.STABLE_DIFFUSION_XL_BASE,
        ModelNames.STABLE_DIFFUSION_XL_512,
    },
}


# DEVICE environment variable
class DeviceTypes(Enum):
    N150 = "n150"
    N300 = "n300"
    GALAXY = "galaxy"
    T3K = "t3k"
    P300 = "p300"
    P150 = "p150"
    P150X4 = "p150x4"  # 4x P150 cards (1,4 mesh)
    P150X8 = "p150x8"  # BH LoudBox - 8x P150 (2,4 mesh)
    P300X2 = "p300x2"  # BH QuietBox GE - 2x P300 cards (2,2 mesh)
    BLACKHOLE_GALAXY = "bh-galaxy"


class QueueType(Enum):
    MemoryQueue = "MemoryQueue"
    FasterFifo = "FasterFifo"
    BatchFifo = "BatchFifo"
    TTQueue = "TTQueue"


class DeviceIds(Enum):
    DEVICE_IDS_1 = "(0)"
    DEVICE_IDS_2 = "(0),(1)"
    DEVICE_IDS_2_GROUP = "(0,1)"
    DEVICE_IDS_2X2_GROUP = "(0,1),(2,3)"
    DEVICE_IDS_4 = "(0),(1),(2),(3)"
    DEVICE_IDS_4_GROUP = "(0,1,2,3)"
    DEVICE_IDS_4x2_GROUP = "(0,1),(2,3),(4,5),(6,7)"
    DEVICE_IDS_8_GROUP = "(0,1,2,3,4,5,6,7)"
    DEVICE_IDS_16 = (
        "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15)"
    )
    DEVICE_IDS_32 = "(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)"
    DEVICE_IDS_32_GROUP = "(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)"


class AudioTasks(Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class ResponseFormat(Enum):
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"


class AudioResponseFormat(Enum):
    """TTS workflow: supported binary response formats."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"


SDXL_VALID_IMAGE_RESOLUTIONS = frozenset({(1024, 1024), (512, 512)})
AUDIO_RESPONSE_FORMATS = frozenset(e.value for e in AudioResponseFormat)

# TTS formats that require ffmpeg for encoding (WAV does not)
FFMPEG_REQUIRED_FORMATS = frozenset(
    (AudioResponseFormat.MP3.value, AudioResponseFormat.OGG.value)
)

# TTS: all allowed response_format values (binary + JSON)
TTS_RESPONSE_FORMATS = AUDIO_RESPONSE_FORMATS | frozenset(
    (ResponseFormat.JSON.value, ResponseFormat.VERBOSE_JSON.value)
)


class JobTypes(Enum):
    VIDEO = "video"
    TRAINING = "training"


class DatasetLoaders(Enum):
    SST2 = "SST2"
    ALPACA = "Alpaca"


class TrainingTrainers(Enum):
    LORA = "lora"
    SFT = "sft"


class ModelDisplayNames(Enum):
    GEMMA_1_1_2B_IT = "Gemma 1.1 2B Instruct"
    LLAMA_3_1_8B = "Llama 3.1 8B"


class TrainingOptimizers(Enum):
    ADAMW = "adamw"


# Base directory for storing fine-tuned adapter outputs.
TRAINING_STORE_ADAPTERS_DIR = "model_store/"


# Helper function to create vLLM configuration with late import to avoid circular imports
def _vllm_config(
    model: str,
    max_model_length: int,
    max_num_batched_tokens: int,
    min_context_length: int = 32,
    max_num_seqs: int = 1,
):
    from config.vllm_settings import VLLMSettings

    return VLLMSettings(
        model=model,
        max_model_length=max_model_length,
        max_num_batched_tokens=max_num_batched_tokens,
        min_context_length=min_context_length,
        max_num_seqs=max_num_seqs,
    )


# Combined model-device specific configurations
# useful when whole device is being used by a single model type
# also for CI testing

ModelConfigs = {
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.N300): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_EDIT, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.N300): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_IMAGE_TO_IMAGE, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 1),
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
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 3000,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.P150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.P300): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.P300X2): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2X2_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.P150X4): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2X2_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.P150X8): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4x2_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SDXL_TRACE, DeviceTypes.BLACKHOLE_GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SD3_5, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_SD3_5, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 3000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P150X4): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P150X8): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_8_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P300): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P300X2): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 3000,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P150X4): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P150X8): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_8_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P300): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P300X2): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW, DeviceTypes.P150X8): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_8_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW, DeviceTypes.P300X2): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 2000,
    },
    (ModelRunners.TT_QWEN_IMAGE, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE_2512, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_QWEN_IMAGE_2512, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.P150X4): {
        "device_mesh_shape": (1, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.P150X8): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_8_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_MOCHI_1, DeviceTypes.P300X2): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.GALAXY): {
        "device_mesh_shape": (4, 8),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_32_GROUP.value,
        "max_batch_size": 1,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.P150X4): {
        "device_mesh_shape": (1, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.P150X8): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_8_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.P300X2): {
        "device_mesh_shape": (2, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.SP_RUNNER, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.SP_RUNNER, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 4),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4_GROUP.value,
        "max_batch_size": 1,
        "download_weights_from_service": False,
    },
    (ModelRunners.TT_WAN_2_2, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,  # "(0)"
        "max_batch_size": 1,
        "download_weights_from_service": False,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SPEECHT5_TTS, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SPEECHT5_TTS, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SPEECHT5_TTS, DeviceTypes.P150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SPEECHT5_TTS, DeviceTypes.P300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_SPEECHT5_TTS, DeviceTypes.P300X2): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 2,
        "queue_for_multiprocessing": QueueType.BatchFifo.value,
        "request_processing_timeout_seconds": 5000,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 2,
        "queue_for_multiprocessing": QueueType.BatchFifo.value,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.P150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.P300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_WHISPER, DeviceTypes.P300X2): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
        "vllm": {
            "model": SupportedModels.QWEN_3_EMBEDDING_4B.value,
            "max_model_length": 1024,
            "max_num_batched_tokens": 1024,
            "min_context_length": 32,
            "max_num_seqs": 1,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
        "vllm": {
            "model": SupportedModels.QWEN_3_EMBEDDING_4B.value,
            "max_model_length": 1024,
            "max_num_batched_tokens": 1024,
            "min_context_length": 32,
            "max_num_seqs": 1,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
        "vllm": {
            "model": SupportedModels.QWEN_3_EMBEDDING_4B.value,
            "max_model_length": 1024,
            "max_num_batched_tokens": 1024,
            "min_context_length": 32,
            "max_num_seqs": 1,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.VLLMForge_QWEN_EMBEDDING, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
        "vllm": {
            "model": SupportedModels.QWEN_3_EMBEDDING_4B.value,
            "max_model_length": 1024,
            "max_num_batched_tokens": 1024,
            "min_context_length": 32,
            "max_num_seqs": 1,
        },
    },
    (ModelRunners.VLLMForge_LLAMA_70B, DeviceTypes.T3K): {
        "device_mesh_shape": (4, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
        "vllm": {
            "model": SupportedModels.LLAMA_3_1_70B.value,
            "max_model_length": 1024,
            "max_num_batched_tokens": 1024,
            "min_context_length": 32,
            "max_num_seqs": 1,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.QWEN_EMBEDDING_8B, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
        "use_queue_per_worker": True,
        "default_throttle_level": 0,
        "request_processing_timeout_seconds": 2000,
        "vllm": _vllm_config(
            model=SupportedModels.QWEN_3_EMBEDDING_8B.value,
            max_model_length=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
        ),
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.QWEN_EMBEDDING_8B, DeviceTypes.N300): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 2,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
        "request_processing_timeout_seconds": 2000,
        "vllm": _vllm_config(
            model=SupportedModels.QWEN_3_EMBEDDING_8B.value,
            max_model_length=4096,
            max_num_batched_tokens=8192,
            max_num_seqs=2,
        ),
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.QWEN_EMBEDDING_8B, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 2,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
        "request_processing_timeout_seconds": 2000,
        "vllm": _vllm_config(
            model=SupportedModels.QWEN_3_EMBEDDING_8B.value,
            max_model_length=4096,
            max_num_batched_tokens=8192,
            max_num_seqs=2,
        ),
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.QWEN_EMBEDDING_8B, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
        "request_processing_timeout_seconds": 2000,
        "vllm": _vllm_config(
            model=SupportedModels.QWEN_3_EMBEDDING_8B.value,
            max_model_length=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
        ),
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
    },
    (ModelRunners.BGELargeEN_V1_5, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 8,
        "vllm": {
            "model": SupportedModels.BGE_LARGE_EN_V1_5.value,
            "max_model_length": 384,
            "max_num_batched_tokens": 384 * 8,
            "min_context_length": 32,
            "max_num_seqs": 8,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGELargeEN_V1_5, DeviceTypes.N300): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 16,
        "vllm": {
            "model": SupportedModels.BGE_LARGE_EN_V1_5.value,
            "max_model_length": 384,
            "max_num_batched_tokens": 384 * 8,
            "min_context_length": 32,
            "max_num_seqs": 8,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGELargeEN_V1_5, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "vllm": {
            "model": SupportedModels.BGE_LARGE_EN_V1_5.value,
            "max_model_length": 384,
            "max_num_batched_tokens": 384 * 8,
            "min_context_length": 32,
            "max_num_seqs": 8,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
        "max_batch_size": 16,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGELargeEN_V1_5, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "vllm": {
            "model": SupportedModels.BGE_LARGE_EN_V1_5.value,
            "max_model_length": 384,
            "max_num_batched_tokens": 384 * 8,
            "min_context_length": 32,
            "max_num_seqs": 8,
        },
        "queue_for_multiprocessing": QueueType.FasterFifo.value,
        "max_batch_size": 8,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGEM3, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 32,
        "vllm": {
            "model": SupportedModels.BGE_M3.value,
            "max_model_length": 8192,
            "max_num_batched_tokens": 8192 * 32,
            "min_context_length": 32,
            "max_num_seqs": 32,
        },
        "queue_for_multiprocessing": QueueType.BatchFifo.value,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGEM3, DeviceTypes.N300): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 32,
        "vllm": {
            "model": SupportedModels.BGE_M3.value,
            "max_model_length": 8192,
            "max_num_batched_tokens": 8192 * 32,
            "min_context_length": 32,
            "max_num_seqs": 32,
        },
        "queue_for_multiprocessing": QueueType.BatchFifo.value,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGEM3, DeviceTypes.T3K): {
        "device_mesh_shape": (2, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "vllm": {
            "model": SupportedModels.BGE_M3.value,
            "max_model_length": 8192,
            "max_num_batched_tokens": 8192 * 32,
            "min_context_length": 32,
            "max_num_seqs": 32,
        },
        "queue_for_multiprocessing": QueueType.BatchFifo.value,
        "max_batch_size": 32,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.BGEM3, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "vllm": {
            "model": SupportedModels.BGE_M3.value,
            "max_model_length": 8192,
            "max_num_batched_tokens": 8192 * 32,
            "min_context_length": 32,
            "max_num_seqs": 32,
        },
        "queue_for_multiprocessing": QueueType.BatchFifo.value,
        "max_batch_size": 32,
        "default_throttle_level": 0,
        "use_queue_per_worker": True,
    },
    (ModelRunners.VLLMForge, DeviceTypes.N150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge, DeviceTypes.N300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge, DeviceTypes.T3K): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge, DeviceTypes.GALAXY): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": True,
        "device_ids": DeviceIds.DEVICE_IDS_32.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge, DeviceTypes.P150): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge, DeviceTypes.P300): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2.value,
        "max_batch_size": 1,
    },
    (ModelRunners.VLLMForge, DeviceTypes.P300X2): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_XLA_SDXL, DeviceTypes.P150X4): {
        "device_mesh_shape": (1, 1),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_4.value,
        "max_batch_size": 1,
    },
    (ModelRunners.TT_XLA_SDXL, DeviceTypes.P300X2): {
        "device_mesh_shape": (1, 2),
        "is_galaxy": False,
        "device_ids": DeviceIds.DEVICE_IDS_2X2_GROUP.value,
        "max_batch_size": 1,
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
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
    }
    ModelConfigs[(runner, DeviceTypes.N300)] = {
        "is_galaxy": False,
        "device_mesh_shape": (1, 1),
        "device_ids": DeviceIds.DEVICE_IDS_1.value,
    }


# Per-model overrides applied after device config (keyed by ModelNames enum value)
MODEL_NAME_OVERRIDES = {
    ModelNames.QWEN_3_4B: {"chat_template_kwargs": {"enable_thinking": False}},
    ModelNames.QWEN_3_8B: {"chat_template_kwargs": {"enable_thinking": False}},
}


# Default sampling parameters for vLLM inference
# These values are used when request parameters are not specified
_DEFAULT_SAMPLING_PARAMS = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": None,
    "stop": [],
    "stop_token_ids": [],
    "bad_words": [],
    "max_tokens": 65536,
    "logprobs": None,
    "truncate_prompt_tokens": None,
    "guided_decoding": None,
    "extra_args": None,
}

# Sentinel object for worker shutdown signaling
SHUTDOWN_SIGNAL = {"__shutdown__": True}
