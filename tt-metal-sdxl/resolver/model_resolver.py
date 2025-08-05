# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_model import BaseModel
from model_services.image_service import ImageService
from config.settings import settings
from utils.logger import TTLogger
# from model_services.task_worker import TaskWorker

# model and worker are singleton
current_model_holder = None
logger = TTLogger()

def model_resolver() -> BaseModel:
    """
    Resolves and returns the appropriate model service singleton.
    This ensures we only create one instance of each model type.
    """
    global current_model_holder, logger
    model_service = settings.model_service
    if model_service == "image":
        if (current_model_holder is None):
            logger.info("Creating new ImageService instance")
            current_model_holder = ImageService()
        return current_model_holder    
    return BaseModel()