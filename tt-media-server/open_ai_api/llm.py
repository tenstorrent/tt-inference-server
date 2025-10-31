# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_completion_request import TextCompletionRequest
from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.responses import JSONResponse
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/completions')
async def complete_text(
    text_completion_request: TextCompletionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    try:
        return await service.process_request(text_completion_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))