# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import math
import uuid

from domain.rerank_request import RerankRequest, ScoreRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


def relevance_score(logit: float) -> float:
    """The model returns a relevance logit; relevance_score is its sigmoid (vLLM / Cohere convention)."""
    return 1.0 / (1.0 + math.exp(-logit))


async def _score_documents(
    service: BaseService, model: str, query: str, documents: list
):
    # The request models already rejected (422) empty document lists and unknown model names.
    request = RerankRequest(model=model, query=query, documents=documents)
    try:
        return await service.process_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rerank")
async def rerank(
    rerank_request: RerankRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Score each document against the query; results sorted by relevance, best first."""
    texts = rerank_request.document_texts()
    pairs = await _score_documents(
        service, rerank_request.model, rerank_request.query, texts
    )
    order = sorted(range(len(texts)), key=lambda i: -pairs[i].logit)
    if rerank_request.top_n is not None:
        order = order[: rerank_request.top_n]
    results = []
    for i in order:
        result = {
            "index": i,
            "relevance_score": relevance_score(pairs[i].logit),
            "logit": pairs[i].logit,
        }
        if rerank_request.return_documents:
            result["document"] = {"text": texts[i]}
        results.append(result)
    return {
        "id": f"rerank-{uuid.uuid4().hex}",
        "model": rerank_request.model,
        "usage": {"total_tokens": sum(p.total_tokens for p in pairs)},
        "results": results,
    }


@router.post("/score")
async def score(
    score_request: ScoreRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Score text_2 (one document or a list) against text_1, in input order."""
    texts = (
        [score_request.text_2]
        if isinstance(score_request.text_2, str)
        else list(score_request.text_2)
    )
    pairs = await _score_documents(
        service, score_request.model, score_request.text_1, texts
    )
    tokens = sum(p.total_tokens for p in pairs)
    return {
        "id": f"score-{uuid.uuid4().hex}",
        "object": "list",
        "model": score_request.model,
        "data": [
            {"index": i, "object": "score", "score": relevance_score(p.logit)}
            for i, p in enumerate(pairs)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
