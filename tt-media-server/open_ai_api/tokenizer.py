# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import Any

from config.constants import ModelNames, SupportedModels
from domain.detokenize_request import DetokenizeRequest
from domain.detokenize_response import DetokenizeResponse
from domain.tokenize_request import TokenizeRequest
from domain.tokenize_response import TokenizeResponse
from fastapi import APIRouter, HTTPException
from utils.logger import TTLogger
from transformers import AutoTokenizer

router = APIRouter()
logger = TTLogger()


def _resolve_model(model: str | None) -> SupportedModels:
    """Resolve model string to SupportedModels enum."""
    if model is None:
        # Default to a common LLM model if not specified
        raise HTTPException(status_code=400, detail="Model is required")

    # First try to match by SupportedModels enum value (HuggingFace path)
    for supported_model in SupportedModels:
        if supported_model.value == model:
            return supported_model

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported model: {model}. Supported models: {[m.value for m in SupportedModels]}",
    )


@router.post("/tokenize")
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    try:
        model = _resolve_model(request.model)
        tokenizer = AutoTokenizer.from_pretrained(model.value)

        token_ids = tokenizer.encode(
            request.prompt, add_special_tokens=request.add_special_tokens
        )

        response: TokenizeResponse = TokenizeResponse(
            count=len(token_ids),
            max_model_len=tokenizer.model_max_length,
            tokens=token_ids,
        )

        if request.return_token_strs:
            token_strs = tokenizer.convert_ids_to_tokens(token_ids)
            response.token_strs = token_strs

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        raise HTTPException(status_code=500, detail=f"Tokenization failed")


@router.post("/detokenize")
def detokenize(request: DetokenizeRequest) -> DetokenizeResponse:
    try:
        model = _resolve_model(request.model)
        tokenizer = AutoTokenizer.from_pretrained(model.value)

        # Detokenize the tokens
        text = tokenizer.decode(request.tokens, skip_special_tokens=False)

        return DetokenizeResponse(prompt=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detokenizing tokens: {e}")
        raise HTTPException(status_code=500, detail=f"Detokenization failed")
