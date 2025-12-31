# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import time
import uuid

from config.constants import ModelRunners
from config.settings import settings
from domain.chat_completion_request import ChatCompletionRequest
from domain.completion_request import CompletionRequest
from domain.text_embedding_request import TextEmbeddingRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
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
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = completion_request.model or "default"

    try:
        if not completion_request.stream:
            result = await service.process_request(completion_request)
            response = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "text": result.text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": result.finish_reason or "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            return JSONResponse(content=response)

        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def result_stream():
            async for partial in service.process_streaming_request(completion_request):
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "text": partial.text,
                            "index": partial.index or 0,
                            "logprobs": None,
                            "finish_reason": partial.finish_reason,
                        }
                    ],
                }
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


@completions_router.post("/chat/completions")
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a chat completion for the provided messages and parameters.

    OpenAI-compatible endpoint for chat completions.
    
    This is the recommended endpoint for text generation.
    See: https://platform.openai.com/docs/api-reference/chat/create
    """
    try:
        # Convert chat messages to a single prompt
        prompt = chat_request.to_prompt()
        
        # Create a CompletionRequest from the ChatCompletionRequest
        completion_request = CompletionRequest(
            model=chat_request.model,
            prompt=prompt,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            frequency_penalty=chat_request.frequency_penalty,
            presence_penalty=chat_request.presence_penalty,
            stop=chat_request.stop,
            stream=chat_request.stream,
            stream_options=chat_request.stream_options,
            seed=chat_request.seed,
            user=chat_request.user,
            n=chat_request.n,
        )
        
        if not chat_request.stream:
            result = await service.process_request(completion_request)
            # Return in chat completion format
            return {
                "id": "chatcmpl-" + completion_request._task_id,
                "object": "chat.completion",
                "created": 0,
                "model": chat_request.model or "unknown",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.text,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def chat_result_stream():
            import json

            async for partial in service.process_streaming_request(completion_request):
                # Format as chat completion chunk
                chunk = {
                    "id": "chatcmpl-" + completion_request._task_id,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": chat_request.model or "unknown",
                    "choices": [
                        {
                            "index": partial.index or 0,
                            "delta": {
                                "content": partial.text,
                            },
                            "finish_reason": partial.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send the final [DONE] message
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            chat_result_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
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
