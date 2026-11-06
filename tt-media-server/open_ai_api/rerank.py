# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import math
import uuid

from domain.rerank_request import RerankRequest, ScoreRequest, served_model_id
from fastapi import APIRouter, Depends, HTTPException, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


def relevance_score(logit: float) -> float:
    """The model returns a relevance logit; relevance_score is its sigmoid (vLLM / Cohere convention).
    Written per sign so math.exp never overflows for a large |logit|."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def _served_model(model):
    """The name to echo in the response: the request's, or the served model's when it was left out."""
    return model if model is not None else served_model_id()


async def _score_documents(service: BaseService, request: RerankRequest):
    """One RerankPairResponse per document, in document order. The scheduler's own HTTP errors (429
    queue full, 405 model not ready) pass through; any other failure becomes a 500 with its message."""
    try:
        return await service.process_request(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) or type(e).__name__)


@router.post("/rerank")
async def rerank(
    rerank_request: RerankRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Score each document against the query; results sorted by relevance, best first."""
    texts = rerank_request.document_texts()
    pairs = await _score_documents(service, rerank_request)
    order = sorted(range(len(texts)), key=lambda i: -pairs[i].logit)
    if rerank_request.top_n:
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
        "model": _served_model(rerank_request.model),
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
    # ScoreRequest already checked the model name and the document count.
    request = RerankRequest(
        model=score_request.model, query=score_request.text_1, documents=texts
    )
    pairs = await _score_documents(service, request)
    tokens = sum(p.total_tokens for p in pairs)
    return {
        "id": f"score-{uuid.uuid4().hex}",
        "object": "list",
        "model": _served_model(score_request.model),
        "data": [
            {"index": i, "object": "score", "score": relevance_score(p.logit)}
            for i, p in enumerate(pairs)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }
