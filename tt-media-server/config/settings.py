# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from functools import lru_cache
from typing import Optional

from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    MODEL_SERVICE_RUNNER_MAP,
    AudioTasks,
    DeviceIds,
    DeviceTypes,
    ModelConfigs,
    ModelNames,
    ModelRunners,
    ModelServices,
    QueueType,
    SupportedModels,
)
from config.vllm_settings import VLLMSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from utils.device_manager import DeviceManager


class Settings(BaseSettings):
    # General settings
    environment: str = "development"
    device: Optional[str] = None

    # Device settings
    device_ids: str = DeviceIds.DEVICE_IDS_32.value
    is_galaxy: bool = True  # used for graph device split and class init
    device_mesh_shape: tuple = (1, 1)
    reset_device_command: str = "tt-smi -r"
    reset_device_sleep_time: float = 5.0
    allow_deep_reset: bool = False
    use_greedy_based_allocation: bool = True

    # Model settings
    model_runner: str = ModelRunners.TT_SDXL_TRACE.value
    model_service: Optional[str] = (
        None  # model_service can be deduced from model_runner using MODEL_SERVICE_RUNNER_MAP
    )
    model_weights_path: str = ""
    preprocessing_model_weights_path: str = ""
    trace_region_size: int = 34541598
    download_weights_from_service: bool = True
    use_queue_per_worker: bool = False
    queue_for_multiprocessing: str = QueueType.TTQueue.value

    # Queue and batch settings
    max_queue_size: int = 5000
    max_batch_size: int = 1
    max_batch_delay_time_ms: Optional[int] = None
    use_dynamic_batcher: bool = False

    # Worker management settings
    new_device_delay_seconds: int = 0
    new_runner_delay_seconds: int = 2
    mock_devices_count: int = 5
    max_worker_restart_count: int = 5
    worker_check_sleep_timeout: float = 30.0
    default_throttle_level: str = "0"

    # Timeout settings
    request_processing_timeout_seconds: int = 1000

    # Job management settings
    max_jobs: int = 10000
    job_cleanup_interval_seconds: int = 300
    job_retention_seconds: int = 86400
    job_max_stuck_time_seconds: int = 10800
    enable_job_persistence: bool = False

    vllm: VLLMSettings = VLLMSettings()

    # Audio processing settings
    allow_audio_preprocessing: bool = True
    audio_chunk_duration_seconds: Optional[int] = None
    max_audio_duration_seconds: float = 60.0
    max_audio_duration_with_preprocessing_seconds: float = (
        300.0  # 5 minutes when preprocessing enabled
    )
    max_audio_size_bytes: int = 50 * 1024 * 1024
    default_sample_rate: int = 16000
    audio_task: str = AudioTasks.TRANSCRIBE.value
    audio_language: str = "English"

    # Telemetry settings
    enable_telemetry: bool = False
    prometheus_endpoint: str = "/metrics"

    model_config = SettingsConfigDict(env_file=".env")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_to_run = os.getenv("MODEL")
        if model_to_run and self.device:
            self._set_config_overrides(model_to_run, self.device)
        self._set_mesh_overrides()

        if self.model_service is None:
            found = False
            for service, runners in MODEL_SERVICE_RUNNER_MAP.items():
                if any(self.model_runner == r.value for r in runners):
                    self.model_service = service.value
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Model service could not be deduced from model runner {self.model_runner}."
                )

        # set model weights path using model name
        if self.model_weights_path is None or self.model_weights_path == "":
            # Convert string to enum first
            model_runner_enum = ModelRunners(self.model_runner)

            # Use dictionary key access
            model_names_set = MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(model_runner_enum)

            if model_names_set:
                # Get first model name from the set
                model_name = list(model_names_set)[0]
                if model_name:
                    supported_model = getattr(SupportedModels, model_name.name, None)
                    if supported_model:
                        self.model_weights_path = supported_model.value

        # use throttling overrides until we confirm is no-throttling a stable approach
        self._set_throttling_overrides()
        self._set_device_pairs_overrides()
        if (
            self.model_service == ModelServices.AUDIO.value
            and self.audio_chunk_duration_seconds is None
        ):
            self._calculate_audio_chunk_duration()

    def _set_device_pairs_overrides(self):
        if self.is_galaxy:
            device_manager = DeviceManager()
            devices = None
            if self.device_mesh_shape == (1, 1) and self.use_greedy_based_allocation:
                # use device manager to use all the available devices
                devices = device_manager.get_single_devices_from_system()
            if self.device_mesh_shape == (2, 1):
                # use device manager to pair devices
                devices = device_manager.get_device_pairs_from_system()
            elif self.device_mesh_shape == (2, 4):
                devices = device_manager.get_device_groups_of_eight_from_system()
            if devices:
                self.device_ids = ",".join([f"({device})" for device in devices])

    def _set_throttling_overrides(self):
        if self.model_runner in [
            ModelRunners.TT_SD3_5.value,
            ModelRunners.TT_FLUX_1_SCHNELL.value,
            ModelRunners.TT_FLUX_1_DEV.value,
            ModelRunners.TT_QWEN_IMAGE.value,
            ModelRunners.TT_MOCHI_1.value,
            ModelRunners.TT_WAN_2_2.value,
        ]:
            self.default_throttle_level = None

    def _set_mesh_overrides(self):
        env_mesh_map = {
            "SD_3_5_FAST": (4, 8),
            "SD_3_5_BASE": (2, 4),
            "TP2": (2, 1),
        }
        for env_var, mesh_shape in env_mesh_map.items():
            value = os.getenv(env_var)
            if value and value.lower() == "true":
                setattr(self, "device_mesh_shape", mesh_shape)
                break

    def _calculate_audio_chunk_duration(self):
        worker_count = len(self.device_ids.replace(" ", "").split("),("))
        self.audio_chunk_duration_seconds = (
            # temporary setup until we get dynamic chunking
            15 if worker_count >= 8 else 15 if worker_count >= 4 else 15
        )

    def _set_config_overrides(self, model_to_run: str, device: str):
        model_name_enum = ModelNames(model_to_run)

        # Find the appropriate model runner for this model name
        model_runner_enum = None
        for runner, model_names in MODEL_RUNNER_TO_MODEL_NAMES_MAP.items():
            if model_name_enum in model_names:
                model_runner_enum = runner
                break

        if model_runner_enum:
            matching_config = ModelConfigs.get((model_runner_enum, DeviceTypes(device)))
        else:
            raise ValueError(f"No model runner found for model {model_to_run}.")

        if matching_config:
            self.model_runner = model_runner_enum.value

            supported_model = getattr(SupportedModels, model_name_enum.name, None)
            if supported_model:
                self.model_weights_path = supported_model.value

            # Apply all configuration values
            for key, value in matching_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)


settings = Settings()


@lru_cache()
def get_settings() -> Settings:
    return settings
