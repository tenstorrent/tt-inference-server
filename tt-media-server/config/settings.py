# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from functools import lru_cache
import os
from typing import Optional
from config.constants import DeviceIds, DeviceTypes, ModelConfigs, ModelNames, ModelRunners, MODEL_SERVICE_RUNNER_MAP, MODEL_RUNNER_TO_MODEL_NAMES_MAP, SupportedModels
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    log_level: str = "INFO"
    environment: str = "development"
    device_ids: str = DeviceIds.DEVICE_IDS_32.value
    device: Optional[str] = None
    max_queue_size: int = 64
    max_batch_size: int = 1
    model_runner: str = ModelRunners.TT_SDXL_TRACE.value
    model_service: Optional[str] = None # model_service can be deduced from model_runner using MODEL_SERVICE_RUNNER_MAP
    is_galaxy: bool = False # used for graph device split and class init
    model_weights_path: str = ""
    preprocessing_model_weights_path: str = ""
    trace_region_size: int = 34541598
    log_file: Optional[str] = None
    device_mesh_shape: tuple = (1, 1)
    new_device_delay_seconds: int = 30
    mock_devices_count: int = 5
    reset_device_command: str = "tt-smi -r"
    reset_device_sleep_time: float = 5.0
    max_worker_restart_count: int = 5
    worker_check_sleep_timeout: float = 30.0
    default_inference_timeout_seconds: int = 90
    allow_deep_reset: bool = False
    default_throttle_level= "5"
    # image specific settings
    num_inference_steps: int = 20 # has to be hardcoded since we cannot allow per image currently
    # audio specific settings
    allow_audio_preprocessing: bool = True
    max_audio_duration_seconds: float = 60.0
    max_audio_duration_with_preprocessing_seconds: float = 300.0  # 5 minutes when preprocessing enabled
    max_audio_size_bytes: int = 50 * 1024 * 1024
    default_sample_rate: int = 16000
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
                raise ValueError(f"Model service could not be deduced from model runner '{self.model_runner}'.")

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
            raise ValueError(f"No model runner found for model '{model_to_run}'.")

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
