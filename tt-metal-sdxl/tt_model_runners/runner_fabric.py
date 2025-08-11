# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.settings import settings
from tt_model_runners.base_device_runner import DeviceRunner

def get_device_runner(worker_id: str) -> DeviceRunner:
    model_runner = settings.model_runner
    runners = {
        "tt-sdxl": lambda wid: __import__("tt_model_runners.sdxl_runner", fromlist=["TTSDXLRunner"]).TTSDXLRunner(wid),
        "tt-sdxl-trace": lambda wid: __import__("tt_model_runners.sdxl_runner_trace", fromlist=["TTSDXLRunnerTrace"]).TTSDXLRunnerTrace(wid),
        "tt-sd3.5": lambda wid: __import__("tt_model_runners.sd35_runner", fromlist=["TTSD35Runner"]).TTSD35Runner(wid),
        "tt-whisper": lambda wid: __import__("tt_model_runners.whisper_runner", fromlist=["TTWhisperRunner"]).TTWhisperRunner(wid),
        "mock": lambda wid: __import__("tt_model_runners.mock_runner", fromlist=["MockRunner"]).MockRunner(wid),
    }
    try:
        return runners[model_runner](worker_id)
    except KeyError:
        raise ValueError(f"Unsupported model runner: {model_runner}")