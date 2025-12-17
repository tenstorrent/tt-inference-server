# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelRunners
from config.settings import settings
from tt_model_runners.base_device_runner import BaseDeviceRunner

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
        "tt_model_runners.dit_runners", fromlist=["TTFlux1DevRunner"]
    ).TTFlux1DevRunner(wid),
    ModelRunners.TT_FLUX_1_SCHNELL: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTFlux1SchnellRunner"]
    ).TTFlux1SchnellRunner(wid),
    ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW: lambda wid: __import__(
        "tt_model_runners.dit_runners", fromlist=["TTMotifImage6BPreviewRunner"]
    ).TTMotifImage6BPreviewRunner(wid),
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
        "tt_model_runners.vllm_forge_runner", fromlist=["VLLMForgeRunner"]
    ).VLLMForgeRunner(wid),
    ModelRunners.VLLMBGELargeEN_V1_5: lambda wid: __import__(
        "tt_model_runners.vllm_bge_large_en_runner", fromlist=["VLLMBGELargeENRunner"]
    ).VLLMBGELargeENRunner(wid),
    ModelRunners.TEST: lambda wid: __import__(
        "tt_model_runners.test_runner", fromlist=["TestRunner"]
    ).TestRunner(wid),
    ModelRunners.VLLMForge_QWEN_EMBEDDING: lambda wid: __import__(
        "tt_model_runners.vllm_forge_qwen_embedding_runner",
        fromlist=["VLLMForgeEmbeddingQwenRunner"],
    ).VLLMForgeEmbeddingQwenRunner(wid),
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
    ModelRunners.MOCK: lambda wid: __import__(
        "tt_model_runners.mock_runner", fromlist=["MockRunner"]
    ).MockRunner(wid),
}


def get_device_runner(worker_id: str) -> BaseDeviceRunner:
    model_runner = settings.model_runner
    try:
        model_runner_enum = ModelRunners(model_runner)
        return AVAILABLE_RUNNERS[model_runner_enum](worker_id)
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
