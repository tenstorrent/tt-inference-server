# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelRunners
from config.settings import settings
from tt_model_runners.base_device_runner import BaseDeviceRunner

AVAILABLE_RUNNERS = {
    ModelRunners.TT_SDXL: lambda wid: __import__("tt_model_runners.sdxl_runner", fromlist=["TTSDXLRunner"]).TTSDXLRunner(wid),
    ModelRunners.TT_SDXL_TRACE: lambda wid: __import__("tt_model_runners.sdxl_runner_trace", fromlist=["TTSDXLRunnerTrace"]).TTSDXLRunnerTrace(wid),
    ModelRunners.TT_SD3_5: lambda wid: __import__("tt_model_runners.sd35_runner", fromlist=["TTSD35Runner"]).TTSD35Runner(wid),
    ModelRunners.TT_WHISPER: lambda wid: __import__("tt_model_runners.whisper_runner", fromlist=["TTWhisperRunner"]).TTWhisperRunner(wid),
    ModelRunners.FORGE: lambda wid: __import__("tt_model_runners.forge_runners.forge_runner", fromlist=["ForgeRunner"]).ForgeRunner(wid),
    ModelRunners.MOCK: lambda wid: __import__("tt_model_runners.mock_runner", fromlist=["MockRunner"]).MockRunner(wid),
}

def get_device_runner(worker_id: str) -> BaseDeviceRunner:
    model_runner = settings.model_runner

    try:
        model_runner_enum = ModelRunners(model_runner)
        return AVAILABLE_RUNNERS[model_runner_enum](worker_id)
    except ValueError:
        raise ValueError(f"Unknown model runner: {model_runner}")
    except KeyError:
        raise ValueError(f"Unsupported model runner: {model_runner}. Available: {', '.join(AVAILABLE_RUNNERS.keys())}")
    except ImportError as e:
        raise ImportError(f"Failed to load model runner '{model_runner}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create model runner '{model_runner}': {e}")
