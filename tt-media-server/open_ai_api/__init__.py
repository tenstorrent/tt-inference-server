# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelServices, ModelRunners
from fastapi import APIRouter
from config.settings import settings

api_router = APIRouter()

from open_ai_api import audio, cnn, image_to_image, text_to_image, tt_maintenance_api

if settings.model_service == ModelServices.IMAGE.value:
    if settings.model_runner == ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value:
        api_router.include_router(image_to_image.router, prefix='/image', tags=['Image processing'])
    else:
        api_router.include_router(text_to_image.router, prefix='/image', tags=['Image processing'])
elif settings.model_service == ModelServices.AUDIO.value:
    api_router.include_router(audio.router, prefix='/audio', tags=['Audio processing'])
elif settings.model_service == ModelServices.CNN.value:
    api_router.include_router(cnn.router, prefix='/cnn', tags=['CNN processing'])

# Maintenance endpoints are always included
api_router.include_router(tt_maintenance_api.router, prefix='', tags=['Maintenance'])
