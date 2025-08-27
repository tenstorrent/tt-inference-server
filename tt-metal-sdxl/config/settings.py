# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from functools import lru_cache
import os
from typing import Optional
from config.constants import ModelConfigs, ModelRunners, ModelServices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_service:str = ModelServices.IMAGE.value
    log_level:str = "INFO"
    environment:str = "development"
    device_ids:str = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    #device_ids:str = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    max_queue_size:int = 64
    max_batch_size:int = 1
    model_runner:str = ModelRunners.TT_SDXL_TRACE.value
    use_graph_device_split: bool = True
    model_weights_path:str = "stabilityai/stable-diffusion-xl-base-1.0"
    trace_region_size:int = 34502656
    log_file: Optional[str] = None
    device_mesh_shape:tuple = (1, 1)
    new_device_delay_seconds:int = 30
    mock_devices_count:int = 5
    reset_device_command: str = "tt-smi -r"
    reset_device_sleep_time: float = 5.0
    max_worker_restart_count: int = 5
    worker_check_sleep_timeout: float = 30.0
    # image specific settings
    num_inference_steps:int = 20 # has to be hardcoded since we cannnot allow per image currently
    # audio specific setttings
    max_audio_duration_seconds: float = 60.0
    max_audio_size_bytes: int = 50 * 1024 * 1024
    default_sample_rate: int = 16000
    model_config = SettingsConfigDict(env_file=".env") 
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Check for MODEL and DEVICE environment override 
        model_to_run = os.getenv("MODEL") or None
        device = os.getenv("DEVICE") or None
        # override config only if both are set
        if model_to_run and device:
            self._set_config_overrides(model_to_run, device)

    def _set_config_overrides(self, model_to_run: str, device: str):
        # Search for matching config by values directly in ModelConfigs
        matching_config = None
        for (model_enum, device_enum), config in ModelConfigs.items():
            if model_enum.value == model_to_run and device_enum.value == device:
                matching_config = config
                break
        
        if matching_config:
            # Apply all configuration values
            for key, value in matching_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

settings = Settings()

@lru_cache()
def get_settings() -> Settings:
    return settings
