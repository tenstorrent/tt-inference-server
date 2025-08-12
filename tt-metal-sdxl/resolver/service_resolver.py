# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from model_services.image_service import ImageService
from model_services.audio_service import AudioService
from config.settings import settings
from utils.logger import TTLogger
import threading

# Singleton holders per service type
_service_holders = {}
logger = TTLogger()
_service_holders_lock = threading.Lock()

def service_resolver() -> BaseService:
    """
    Resolves and returns the appropriate model service singleton.
    This ensures we only create one instance of each model type.
    """
    model_service = settings.model_service
    with _service_holders_lock:
        if model_service not in _service_holders:
            if model_service == "image":
                logger.info("Creating new ImageService instance")
                _service_holders[model_service] = ImageService()
            elif model_service == "audio":
                logger.info("Creating new AudioService instance")
                _service_holders[model_service] = AudioService()
            else:
                raise ValueError(f"Unsupported model service: {model_service}")
    return _service_holders[model_service]