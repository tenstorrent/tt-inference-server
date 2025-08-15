# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_service:str = "image"
    log_level:str = "INFO"
    environment:str = "development"
    device_ids:str = "0"
    devices_per_runner:int = 1
    max_queue_size:int = 64
    max_batch_size:int = 32
    model_runner:str = "tt-sdxl"
    num_inference_steps:int = 20 # has to be hardcoded since we cannnot allow per image currently
    log_file: Optional[str] = None
    device_mesh_shape:tuple = (1, 1)
    mock_devices_count:int = 5
    model_config = SettingsConfigDict(env_file=".env") 
    reset_device_command: str = "tt-smi -r"
    reset_device_sleep_time: float = 5.0
    max_worker_restart_count: int = 5
    worker_check_sleep_timeout: float = 30.0
    max_audio_duration_seconds: float = 60.0
    max_audio_size_bytes: int = 50 * 1024 * 1024
    default_sample_rate: int = 16000

settings = Settings()

@lru_cache()
def get_settings() -> Settings:
    return Settings()