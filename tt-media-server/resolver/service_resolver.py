# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelServices
from domain.base_image_generate_request import BaseImageGenerateRequest
from model_services.base_service import BaseService
from config.settings import settings
from utils.logger import TTLogger
import threading

# Supported model services with factory functions
_SUPPORTED_MODEL_SERVICES = {
    ModelServices.IMAGE_SDXL: lambda: __import__('model_services.sdxl_image_service', fromlist=['SDXLImageService']).SDXLImageService(),
    ModelServices.IMAGE_SD35: lambda: __import__('model_services.sd35_image_service', fromlist=['SD35ImageService']).SD35ImageService(),
    ModelServices.AUDIO: lambda: __import__('model_services.audio_service', fromlist=['AudioService']).AudioService(),
    ModelServices.CNN: lambda: __import__('model_services.cnn_service', fromlist=['CNNService']).CNNService(),
}

# Singleton holders per service type
_service_holders = {}
logger = TTLogger()
_service_holders_lock = threading.Lock()

def get_image_request_model():
    """
    Returns the appropriate image request model class based on the configured model service.
    """
    model_service = ModelServices(settings.model_service)
    if model_service == ModelServices.IMAGE_SDXL:
        return __import__('domain.sdxl_image_generate_request', fromlist=['SDXLImageRequest']).SDXLImageRequest
    elif model_service == ModelServices.IMAGE_SD35:
        return __import__('domain.sd35_image_generate_request', fromlist=['SD35ImageRequest']).SD35ImageRequest
    else:
        # Fallback to base class
        return BaseImageGenerateRequest

def service_resolver() -> BaseService:
    """
    Resolves and returns the appropriate model service singleton.
    This ensures we only create one instance of each model type.
    """
    model_service = ModelServices(settings.model_service)
    with _service_holders_lock:
        if model_service not in _service_holders:
            if not model_service in _SUPPORTED_MODEL_SERVICES:
                raise ValueError(
                    f"Unsupported model service: {settings.model_service}. "
                    f"Supported services: {', '.join([s.value for s in _SUPPORTED_MODEL_SERVICES.keys()])}"
                )
            logger.info(f"Creating new {model_service.value.title()} service instance")
            _service_holders[model_service] = _SUPPORTED_MODEL_SERVICES[model_service]()
    return _service_holders[model_service]


