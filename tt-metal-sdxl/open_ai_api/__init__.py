# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from fastapi import APIRouter
from config.settings import settings

api_router = APIRouter()

from open_ai_api import audio, cnn, image, tt_maintenance_api

if (settings.model_service == "image"):
    api_router.include_router(image.router, prefix='/image', tags=['Image processing'])
elif (settings.model_service == "audio"):
    api_router.include_router(audio.router, prefix='/audio', tags=['Audio processing'])
elif (settings.model_service == "cnn"):
    api_router.include_router(cnn.router, prefix='/cnn', tags=['CNN processing'])

# Maintenance endpoints are always included
api_router.include_router(tt_maintenance_api.router, prefix='', tags=['Maintenance'])
