# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from config.settings import settings
from fastapi import APIRouter, Security
from security.api_key_checker import get_api_key

router = APIRouter()


@router.get("/models")
async def list_models(api_key: str = Security(get_api_key)):
    """
    List available models. OpenAI-compatible endpoint.
    See: https://platform.openai.com/docs/api-reference/models/list
    """
    model_id = (
        settings.vllm.model
        if hasattr(settings, "vllm")
        else settings.model_weights_path
    )
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "tenstorrent",
            }
        ],
    }
