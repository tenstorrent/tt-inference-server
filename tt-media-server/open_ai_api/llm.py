# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import json
import time
import uuid

from config.settings import settings
from domain.completion_request import CompletionRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from utils.logger import TTLogger

from open_ai_api.chat import _count_tokens

logger = TTLogger()
router = APIRouter()


def _split_batched_prompts(req: CompletionRequest) -> list[CompletionRequest]:
    """Split a batched-prompt request into N single-prompt requests.

    OpenAI /v1/completions accepts prompt as List[str] or List[List[int]] for
    batched submission. The downstream runner accepts only a single prompt
    per request, so split here and aggregate results into N choices.
    """
    p = req.prompt
    if isinstance(p, list) and p and isinstance(p[0], (list, str)):
        return [req.model_copy(update={"prompt": item}) for item in p]
    return [req]


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

    sub_requests = _split_batched_prompts(completion_request)

    # Compute prompt token counts (for context-window validation and usage stats).
    # vLLM requires `prompt + max_tokens <= max_model_len`; a prompt that fits
    # the context window exactly (prompt == max_model_len) leaves zero room for
    # output and fails deep in the engine as an HTTP 500. Validate the combined
    # length up front so the failure mode is a clean 400 from the handler.
    prompt_tokens_total = 0
    try:
        max_model_len = settings.vllm.max_model_length
        for r in sub_requests:
            p = r.prompt
            if isinstance(p, str):
                prompt_tokens = _count_tokens(p)
            elif isinstance(p, list):
                prompt_tokens = len(p)
            else:
                prompt_tokens = 0
            prompt_tokens_total += prompt_tokens
            # Default to 1 when max_tokens is unset; vLLM needs >= 1 output token.
            # Pass explicit values through (incl. 0) so the check matches what the
            # engine will actually try to allocate.
            output_tokens_needed = r.max_tokens if r.max_tokens is not None else 1
            if prompt_tokens + output_tokens_needed > max_model_len:
                logger.warning(
                    f"Rejected prompt: length ({prompt_tokens}) + max_tokens "
                    f"({output_tokens_needed}) exceeds max model length ({max_model_len})"
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Prompt length ({prompt_tokens}) + max_tokens "
                        f"({output_tokens_needed}) exceeds max model length ({max_model_len})"
                    ),
                )
    except HTTPException:
        raise
    except Exception:
        pass  # Skip validation if tokenizer unavailable (e.g., test/mock runners)

    try:
        if not completion_request.stream:
            results = await asyncio.gather(
                *(service.process_request(r) for r in sub_requests)
            )
            completion_tokens = sum(_count_tokens(r.text) for r in results if r.text)
            response = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "text": r.text,
                        "index": i,
                        "logprobs": None,
                        "finish_reason": r.finish_reason or "stop",
                    }
                    for i, r in enumerate(results)
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens_total,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens_total + completion_tokens,
                },
            }
            return JSONResponse(content=response)

        # Streaming + batched prompts: not implemented. Multiplexing N async
        # generators into one SSE stream is significant work and no current
        # client (lm-eval-harness, vllm bench) sends batched + streaming.
        if len(sub_requests) > 1:
            raise HTTPException(
                status_code=501,
                detail="Streaming responses are not supported for batched prompts.",
            )

        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def result_stream():
            async for partial in service.process_streaming_request(sub_requests[0]):
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
