# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelRunners
from config.settings import settings
from domain.completion_request import CompletionRequest
from domain.text_embedding_request import TextEmbeddingRequest
from fastapi import APIRouter, Depends, HTTPException, Response, Security
from fastapi.responses import StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

completions_router = APIRouter()


@completions_router.post("/completions")
async def complete_text(
    completion_request: CompletionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a completion for the provided prompt and parameters.

    OpenAI-compatible endpoint for text completions.

    Note: This endpoint is considered legacy according to OpenAI documentation.
    Most developers should use the Chat Completions API to leverage the best and newest models.
    See: https://platform.openai.com/docs/api-reference/completions
    """
    try:
        if not completion_request.stream:
            result = await service.process_request(completion_request)
            return Response(content=result.text, media_type="text/plain")

        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def result_stream():
            import json

            async for partial in service.process_streaming_request(completion_request):
                chunk = {"choices": [partial.to_dict()]}
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(
            result_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


embedding_router = APIRouter()


@embedding_router.post("/embeddings")
async def create_embedding(
    text_embedding_request: TextEmbeddingRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create text embeddings based on the provided request.
    Returns:
        JSONResponse: The generated embeddings as a list of float vectors.
    Raises:
        HTTPException: If embedding generation fails.
    """
    try:
        response = await service.process_request(text_embedding_request)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": response.embedding, "index": 0}
            ],
            "model": text_embedding_request.model,
            "usage": {
                "total_tokens": response.total_tokens,
                "prompt_tokens": response.total_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


router = APIRouter()
if settings.model_runner in [
    ModelRunners.VLLMForge_QWEN_EMBEDDING.value,
    ModelRunners.VLLMBGELargeEN_V1_5.value,
]:
    router.include_router(embedding_router)
else:
    router.include_router(completions_router)
