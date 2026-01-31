# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import time
import uuid

from domain.completion_request import CompletionRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


@router.post("/completions")
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
