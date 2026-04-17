# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelRunners
from config.settings import settings
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.logger import TTLogger

AVAILABLE_RUNNERS = {
    ModelRunners.TT_SDXL_TRACE: lambda wid: __import__(
        "tt_model_runners.sdxl_generate_runner_trace",
        fromlist=["TTSDXLGenerateRunnerTrace"],
    ).TTSDXLGenerateRunnerTrace(wid),
    ModelRunners.TT_SDXL_IMAGE_TO_IMAGE: lambda wid: __import__(
        "tt_model_runners.sdxl_image_to_image_runner_trace",
        fromlist=["TTSDXLImageToImageRunner"],
    ).TTSDXLImageToImageRunner(wid),
    ModelRunners.TT_SDXL_EDIT: lambda wid: __import__(
        "tt_model_runners.sdxl_edit_runner_trace", fromlist=["TTSDXLEditRunner"]
    ).TTSDXLEditRunner(wid),
    ModelRunners.TT_SD3_5: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTSD35Runner"]
    ).TTSD35Runner(wid),
    ModelRunners.TT_FLUX_1_DEV: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTFlux1Runner"]
    ).TTFlux1Runner(wid),
    ModelRunners.TT_FLUX_1_SCHNELL: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTFlux1Runner"]
    ).TTFlux1Runner(wid),
    ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTMotifImage6BPreviewRunner"]
    ).TTMotifImage6BPreviewRunner(wid),
    ModelRunners.TT_QWEN_IMAGE: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTQwenImageRunner"]
    ).TTQwenImageRunner(wid),
    ModelRunners.TT_QWEN_IMAGE_2512: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTQwenImage2512Runner"]
    ).TTQwenImageRunner(wid),
    ModelRunners.TT_MOCHI_1: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTMochi1Runner"]
    ).TTMochi1Runner(wid),
    ModelRunners.TT_WAN_2_2: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTWan22Runner"]
    ).TTWan22Runner(wid),
    ModelRunners.TT_WHISPER: lambda wid: __import__(
        "tt_model_runners.whisper_runner", fromlist=["TTWhisperRunner"]
    ).TTWhisperRunner(wid),
    ModelRunners.VLLMForge: lambda wid: __import__(
        "tt_model_runners.vllm_runner", fromlist=["VLLMForgeRunner"]
    ).VLLMForgeRunner(wid),
    ModelRunners.BGELargeEN_V1_5: lambda wid: __import__(
        "tt_model_runners.embedding_runner", fromlist=["BGELargeENRunner"]
    ).BGELargeENRunner(wid),
    ModelRunners.BGEM3: lambda wid: __import__(
        "tt_model_runners.embedding_runner", fromlist=["BGEM3Runner"]
    ).BGEM3Runner(wid),
    ModelRunners.LLM_TEST: lambda wid: __import__(
        "tt_model_runners.llm_test_runner", fromlist=["LLMTestRunner"]
    ).LLMTestRunner(wid),
    ModelRunners.QWEN_EMBEDDING_8B: lambda wid: __import__(
        "tt_model_runners.embedding_runner",
        fromlist=["Qwen3Embedding8BRunner"],
    ).Qwen3Embedding8BRunner(wid),
    ModelRunners.TT_YOLOV4: lambda wid: __import__(
        "tt_model_runners.yolov4_runner", fromlist=["TTYolov4Runner"]
    ).TTYolov4Runner(wid),
    ModelRunners.VLLMForge_QWEN_EMBEDDING: lambda wid: __import__(
        "tt_model_runners.vllm_forge_qwen_embedding_runner",
        fromlist=["VLLMForgeEmbeddingQwenRunner"],
    ).VLLMForgeEmbeddingQwenRunner(wid),
    ModelRunners.VLLMForge_LLAMA_70B: lambda wid: __import__(
        "tt_model_runners.vllm_forge_llama_70b",
        fromlist=["VLLMForgeLlama70BRunner"],
    ).VLLMForgeLlama70BRunner(wid),
    ModelRunners.TT_XLA_RESNET: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeResnetRunner"]
    ).ForgeResnetRunner(wid),
    ModelRunners.TT_XLA_VOVNET: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeVovnetRunner"]
    ).ForgeVovnetRunner(wid),
    ModelRunners.TT_XLA_MOBILENETV2: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeMobilenetv2Runner"]
    ).ForgeMobilenetv2Runner(wid),
    ModelRunners.TT_XLA_EFFICIENTNET: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeEfficientnetRunner"]
    ).ForgeEfficientnetRunner(wid),
    ModelRunners.TT_XLA_SEGFORMER: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeSegformerRunner"]
    ).ForgeSegformerRunner(wid),
    ModelRunners.TT_XLA_UNET: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeUnetRunner"]
    ).ForgeUnetRunner(wid),
    ModelRunners.TT_XLA_VIT: lambda wid: __import__(
        "tt_model_runners.forge_runners.runners", fromlist=["ForgeVitRunner"]
    ).ForgeVitRunner(wid),
    ModelRunners.TRAINING_GEMMA_LORA: lambda wid: __import__(
        "tt_model_runners.forge_training_runners.training_gemma_lora_runner",
        fromlist=["TrainingGemmaLoraRunner"],
    ).TrainingGemmaLoraRunner(wid),
    ModelRunners.LORA_SINGLE_CHIP: lambda wid: __import__(
        "tt_model_runners.forge_runners.lora_single_chip_runner",
        fromlist=["LoraSingleChipRunner"],
    ).LoraSingleChipRunner(wid),
    ModelRunners.VLLM_LORA: lambda wid: __import__(
        "tt_model_runners.vllm_lora_basic_runner",
        fromlist=["VLLMLoraRunner"],
    ).VLLMLoraRunner(wid),
    ModelRunners.TRAINING_LLAMA_LORA: lambda wid: __import__(
        "tt_model_runners.forge_training_runners.training_llama_lora_runner",
        fromlist=["TrainingLlamaLoraRunner"],
    ).TrainingLlamaLoraRunner(wid),
    ModelRunners.MOCK: lambda wid: __import__(
        "tt_model_runners.mock_runner", fromlist=["MockRunner"]
    ).MockRunner(wid),
    ModelRunners.SP_RUNNER: lambda wid: __import__(
        "tt_model_runners.sp_runner", fromlist=["SPRunner"]
    ).SPRunner(wid),
    ModelRunners.TT_SPEECHT5_TTS: lambda wid: __import__(
        "tt_model_runners.speecht5_runner", fromlist=["TTSpeechT5Runner"]
    ).TTSpeechT5Runner(wid),
    ModelRunners.TT_XLA_SDXL: lambda wid: __import__(
        "tt_model_runners.forge_runners.sdxl_forge_runner",
        fromlist=["SDXLForgeRunner"],
    ).SDXLForgeRunner(wid),
}


def get_device_runner(worker_id: str) -> BaseDeviceRunner:
    _logger = TTLogger()
    model_runner = settings.model_runner
    _logger.info(
        f"get_device_runner: worker_id={worker_id!r}, model_runner={model_runner!r}"
    )
    try:
        model_runner_enum = ModelRunners(model_runner)
        runner = AVAILABLE_RUNNERS[model_runner_enum](worker_id)
        _logger.info(
            f"get_device_runner: created {type(runner).__name__} for worker {worker_id}"
        )
        return runner
    except ValueError:
        raise ValueError(f"Unknown model runner: {model_runner}")
    except KeyError:
        raise ValueError(
            f"Unsupported model runner: {model_runner}. Available: {', '.join(AVAILABLE_RUNNERS.keys())}"
        )
    except ImportError as e:
        raise ImportError(f"Failed to load model runner {model_runner}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create model runner {model_runner}: {e}")
