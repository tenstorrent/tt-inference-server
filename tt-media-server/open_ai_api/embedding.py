# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_embedding_request import TextEmbeddingRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


@router.post("/embeddings")
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
