# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from fastapi import APIRouter

api_router = APIRouter()

from open_ai_api import image, llm, cnn

api_router.include_router(image.router, prefix='/image', tags=['Image processing'])
api_router.include_router(cnn.router, prefix='/cnn', tags=['CNN processing'])
api_router.include_router(llm.router, prefix='', tags=['Language processing'])
