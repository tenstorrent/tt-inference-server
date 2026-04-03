# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import time
import uuid
from functools import lru_cache

from config.settings import settings
from domain.chat_completion_request import ChatCompletionRequest
from domain.completion_request import CompletionRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from transformers import AutoTokenizer
from utils.logger import TTLogger

router = APIRouter()
logger = TTLogger()


@lru_cache(maxsize=1)
def _get_tokenizer():
    model_name = settings.model_weights_path
    logger.info(f"Loading tokenizer for chat template: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def _apply_chat_template(messages: list[dict]) -> str:
    tokenizer = _get_tokenizer()
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **settings.chat_template_kwargs,
    )


def _count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text))


def _build_completion_request(
    chat_request: ChatCompletionRequest, prompt: str
) -> CompletionRequest:
    return CompletionRequest(
        model=chat_request.model,
        prompt=prompt,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        top_k=chat_request.top_k,
        repetition_penalty=chat_request.repetition_penalty,
        frequency_penalty=chat_request.frequency_penalty,
        presence_penalty=chat_request.presence_penalty,
        stream=chat_request.stream,
        stop=chat_request.stop,
        n=chat_request.n,
        seed=chat_request.seed,
    )


@router.post("/chat/completions")
async def chat_completions(
    chat_request: ChatCompletionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a chat completion. OpenAI-compatible endpoint.

    Accepts messages in chat format, applies the model's chat template,
    and returns completions in the chat completions response format.
    See: https://platform.openai.com/docs/api-reference/chat/create
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = chat_request.model or (
        settings.vllm.model
        if hasattr(settings, "vllm")
        else settings.model_weights_path
    )

    # Convert chat messages to a prompt using the model's chat template
    messages = [{"role": m.role, "content": m.content} for m in chat_request.messages]
    prompt = _apply_chat_template(messages)
    prompt_tokens = _count_tokens(prompt)

    # Build an internal CompletionRequest to reuse the existing inference pipeline
    completion_request = _build_completion_request(chat_request, prompt)

    try:
        if not chat_request.stream:
            result = await service.process_request(completion_request)
            completion_tokens = _count_tokens(result.text) if result.text else 0
            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": result.finish_reason or "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return JSONResponse(content=response)

        # Streaming path
        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def result_stream():
            try:
                accumulated_text = ""
                async for partial in service.process_streaming_request(
                    completion_request
                ):
                    accumulated_text += partial.text
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": partial.text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk with finish_reason and usage stats
                completion_tokens = (
                    _count_tokens(accumulated_text) if accumulated_text else 0
                )
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n\n[Error: {e}]"},
                            "finish_reason": "error",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
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
