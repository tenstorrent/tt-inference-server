# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelServices
from model_services.base_service import BaseService
from config.settings import settings
from utils.logger import TTLogger
import threading

# Supported model services with factory functions
_SUPPORTED_MODEL_SERVICES = {
    ModelServices.IMAGE: lambda: __import__('model_services.image_service', fromlist=['ImageService']).ImageService(),
    ModelServices.AUDIO: lambda: __import__('model_services.audio_service', fromlist=['AudioService']).AudioService(),
    ModelServices.CNN: lambda: __import__('model_services.cnn_service', fromlist=['CNNService']).CNNService(),
}

# Singleton holders per service type
_service_holders = {}
logger = TTLogger()
_service_holders_lock = threading.Lock()

def service_resolver() -> BaseService:
    """
    Resolves and returns the appropriate model service singleton.
    This ensures we only create one instance of each model type.
    """
    model_service = ModelServices(settings.model_service)
    with _service_holders_lock:
        if model_service not in _service_holders:
            if model_service in _SUPPORTED_MODEL_SERVICES:
                logger.info(f"Creating new {model_service.value.title()} service instance")
                _service_holders[model_service] = _SUPPORTED_MODEL_SERVICES[model_service]()
            else:
                raise ValueError(
                    f"Unsupported model service: {model_service}. "
                    f"Supported services: {', '.join([s.value for s in _SUPPORTED_MODEL_SERVICES.keys()])}"
                )
    return _service_holders[model_service]
