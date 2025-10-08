# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from functools import lru_cache
import os
from typing import Optional
from config.constants import DeviceTypes, ModelConfigs, ModelRunners, MODEL_SERVICE_RUNNER_MAP, SupportedModels
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    log_level: str = "INFO"
    environment: str = "development"
    device_ids: str = Field(default="(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)", alias="DEVICE_IDS")
    device: Optional[str] = os.getenv("DEVICE") or None
    max_queue_size: int = 64
    max_batch_size: int = 1
    model_runner: str = os.getenv("MODEL_RUNNER", ModelRunners.TT_SDXL_TRACE.value)
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
    # image specific settings
    num_inference_steps: int = 20 # has to be hardcoded since we cannnot allow per image currently
    # audio specific setttings
    enable_audio_preprocessing: bool = True
    max_audio_duration_seconds: float = 60.0
    max_audio_duration_with_preprocessing_seconds: float = 300.0  # 5 minutes when preprocessing enabled
    max_audio_size_bytes: int = 50 * 1024 * 1024
    default_sample_rate: int = 16000
    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True) 
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_to_run = os.getenv("MODEL") or None
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
        matching_config = ModelConfigs.get((SupportedModels(model_to_run), DeviceTypes(device)))
        if matching_config:
            # Apply all configuration values, but respect explicitly set environment variables
            for key, value in matching_config.items():
                if hasattr(self, key):
                    # Check if this field has an explicit environment variable set
                    if key == "device_ids" and os.getenv("DEVICE_IDS"):
                        # Skip overriding device_ids if DEVICE_IDS environment variable is set
                        continue
                    setattr(self, key, value)

settings = Settings()

@lru_cache()
def get_settings() -> Settings:
    return settings
