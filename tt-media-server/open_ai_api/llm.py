# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.settings import settings
from config.constants import ModelRunners, ModelServices
from domain.text_completion_request import TextCompletionRequest
from domain.text_embedding_request import TextEmbeddingRequest
from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.responses import JSONResponse
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

completions_router = APIRouter()

@completions_router.post('/completions')
async def complete_text(
    text_completion_request: TextCompletionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    try:
        return await service.process_request(text_completion_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

embedding_router = APIRouter()

@embedding_router.post('/embeddings')
async def create_embedding(
    text_embedding_request: TextEmbeddingRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Create text embeddings based on the provided request.
    Returns:
        JSONResponse: The generated embeddings as a list of float vectors.
    Raises:
        HTTPException: If embedding generation fails.
    """
    try:
        return await service.process_request(text_embedding_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
router = APIRouter()
if settings.model_runner == ModelRunners.VLLMForge_QWEN_EMBEDDING.value:
    router.include_router(embedding_router)
else:
    router.include_router(completions_router)