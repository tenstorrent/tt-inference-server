# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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


# Demo-safe sampling floor. The tt-cloud-console website's slider passes
# top_k=0 (= disabled) by default, leaving the long tail of the vocab fully
# samplable at temperature > 0. That occasionally lets the sampler pick a
# very-low-probability foreign-language / garbage token, which then steers
# subsequent tokens into incoherent output (the "tail cascade"). To keep
# users safe regardless of what the website sends, force top_k to a sane
# floor unless the user explicitly opts out by setting top_k > _DEMO_TOP_K.
# Set to None to disable the override entirely.
_DEMO_TOP_K = 50


def _build_completion_request(
    chat_request: ChatCompletionRequest, prompt: str
) -> CompletionRequest:
    # Apply demo-safe top_k floor: if the request didn't set top_k or set
    # it to disabled (0/None/-1), force it to _DEMO_TOP_K. This is invisible
    # to the website slider — values higher than _DEMO_TOP_K from the user
    # are still respected. Logs the override so it's auditable.
    requested_top_k = chat_request.top_k
    effective_top_k = requested_top_k
    if _DEMO_TOP_K is not None:
        if requested_top_k is None or requested_top_k == 0 or requested_top_k == -1:
            effective_top_k = _DEMO_TOP_K
            logger.info(
                f"top_k override: requested={requested_top_k} -> {_DEMO_TOP_K} "
                f"(server-side floor against tail-cascade gibberish)"
            )

    return CompletionRequest(
        model=chat_request.model,
        prompt=prompt,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        top_k=effective_top_k,
        repetition_penalty=chat_request.repetition_penalty,
        frequency_penalty=chat_request.frequency_penalty,
        presence_penalty=chat_request.presence_penalty,
        stream=chat_request.stream,
        stop=chat_request.stop,
        n=chat_request.n,
        seed=chat_request.seed,
        adapter=chat_request.adapter,
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

    stop_count = (
        len(chat_request.stop)
        if isinstance(chat_request.stop, list)
        else (1 if chat_request.stop else 0)
    )
    logger.info(
        f"Chat request: model={chat_request.model}, "
        f"prompt_tokens={prompt_tokens}, n_messages={len(chat_request.messages)}, "
        f"max_tokens={chat_request.max_tokens}, n={chat_request.n}, "
        f"stream={chat_request.stream}, "
        f"temp={chat_request.temperature}, "
        f"top_p={chat_request.top_p}, top_k={chat_request.top_k}, "
        f"freq_penalty={chat_request.frequency_penalty}, "
        f"pres_penalty={chat_request.presence_penalty}, "
        f"rep_penalty={chat_request.repetition_penalty}, "
        f"stop_count={stop_count}, seed={chat_request.seed}, "
        f"adapter={chat_request.adapter}"
    )

    # Reject prompts that exceed the model's context window
    max_model_len = settings.vllm.max_model_length
    if prompt_tokens > max_model_len:
        logger.warning(
            f"Rejected prompt: length ({prompt_tokens}) exceeds max model length ({max_model_len})"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length ({prompt_tokens}) exceeds max model length ({max_model_len})",
        )

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
            accumulated_text = ""
            async for partial in service.process_streaming_request(completion_request):
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
