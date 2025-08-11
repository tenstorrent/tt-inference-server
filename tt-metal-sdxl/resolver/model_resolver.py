# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_model import BaseModel
from model_services.image_service import ImageService
from model_services.audio_service import AudioService
from config.settings import settings
from utils.logger import TTLogger

BASE_MODEL_KEY = "base"

# Singleton holders per service type
_model_holders = {}
logger = TTLogger()

def model_resolver() -> BaseModel:
    """
    Resolves and returns the appropriate model service singleton.
    This ensures we only create one instance of each model type.
    """
    model_service = settings.model_service
    if model_service not in _model_holders:
        if model_service == "image":
            logger.info("Creating new ImageService instance")
            _model_holders[model_service] = ImageService()
        elif model_service == "audio":
            logger.info("Creating new AudioService instance")
            _model_holders[model_service] = AudioService()
        else:
            logger.info("Creating new BaseModel instance")
            _model_holders[BASE_MODEL_KEY] = BaseModel()
    if model_service in _model_holders:
        return _model_holders[model_service]
    else:
        return _model_holders[BASE_MODEL_KEY]