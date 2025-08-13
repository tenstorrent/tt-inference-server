# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from fastapi import APIRouter

api_router = APIRouter()

from open_ai_api import audio, image, tt_maintenance_api

api_router.include_router(audio.router, prefix='/audio', tags=['Audio processing'])
api_router.include_router(image.router, prefix='/image', tags=['Image processing'])
api_router.include_router(tt_maintenance_api.router, prefix='', tags=['Maintenance'])
