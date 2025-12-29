# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import time
import uuid

from config.constants import ModelRunners
from config.settings import settings
from domain.completion_request import CompletionRequest
from domain.text_embedding_request import TextEmbeddingRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

completions_router = APIRouter()


def create_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def create_completion_response(
    completion_id: str,
    model: str,
    text: str,
    created: int,
    finish_reason: str = "stop",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    """Create OpenAI-compatible completion response."""
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_stream_chunk(
    completion_id: str,
    model: str,
    text: str,
    created: int,
    index: int = 0,
    finish_reason: str | None = None,
) -> dict:
    """Create OpenAI-compatible streaming chunk."""
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


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
    completion_id = create_completion_id()
    created = int(time.time())
    model = completion_request.model or "default"

    try:
        if not completion_request.stream:
            result = await service.process_request(completion_request)
            response = create_completion_response(
                completion_id=completion_id,
                model=model,
                text=result.text,
                created=created,
                finish_reason=result.finish_reason or "stop",
            )
            return JSONResponse(content=response)

        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def result_stream():
            async for partial in service.process_streaming_request(completion_request):
                chunk = create_stream_chunk(
                    completion_id=completion_id,
                    model=model,
                    text=partial.text,
                    created=created,
                    index=partial.index or 0,
                    finish_reason=partial.finish_reason,
                )
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            result_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
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
