# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import threading

from config.constants import ModelServices
from config.settings import settings
from model_services.base_service import BaseService
from utils.logger import TTLogger

# Supported model services with factory functions
_SUPPORTED_MODEL_SERVICES = {
    ModelServices.IMAGE: lambda: __import__(
        "model_services.image_service", fromlist=["ImageService"]
    ).ImageService(),
    ModelServices.LLM: lambda: __import__(
        "model_services.llm_service", fromlist=["LLMService"]
    ).LLMService(),
    ModelServices.CNN: lambda: __import__(
        "model_services.cnn_service", fromlist=["CNNService"]
    ).CNNService(),
    ModelServices.AUDIO: lambda: __import__(
        "model_services.audio_service", fromlist=["AudioService"]
    ).AudioService(),
    ModelServices.VIDEO: lambda: __import__(
        "model_services.video_service", fromlist=["VideoService"]
    ).VideoService(),
    ModelServices.TRAINING: lambda: __import__(
        "model_services.training_service", fromlist=["TrainingService"]
    ).TrainingService(),
    ModelServices.TEXT_TO_SPEECH: lambda: __import__(
        "model_services.text_to_speech_service", fromlist=["TextToSpeechService"]
    ).TextToSpeechService(),
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
            if model_service not in _SUPPORTED_MODEL_SERVICES:
                raise ValueError(
                    f"Unsupported model service: {settings.model_service}. "
                    f"Supported services: {', '.join([s.value for s in _SUPPORTED_MODEL_SERVICES.keys()])}"
                )
            logger.info(f"Creating new {model_service.value.title()} service instance")
            _service_holders[model_service] = _SUPPORTED_MODEL_SERVICES[model_service]()
    return _service_holders[model_service]
