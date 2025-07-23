from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_in_use:str = "SDXL"
    log_level:str = "INFO"
    environment:str = "development"
    device_ids:str = "0"
    devices_per_runner:int = 1
    max_queue_size:int = 4
    model_runner:str = "tt-sdxl"

    model_config = SettingsConfigDict(env_file=".env") 

settings = Settings()

@lru_cache()
def get_settings() -> Settings:
    return Settings()