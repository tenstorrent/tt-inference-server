# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import Optional, Union

from domain.base_request import BaseRequest
from pydantic import BaseModel, Field, PrivateAttr, field_validator


def served_model_names() -> set[str]:
    """Names a client may pass as ``model``: the running runner's ModelNames and their Hugging Face ids.
    Checked at validation (a 422) so a wrong name never reaches the worker, where it would be a 500
    that counts towards a worker restart."""
    from config.constants import (
        MODEL_RUNNER_TO_MODEL_NAMES_MAP,
        ModelRunners,
        SupportedModels,
    )
    from config.settings import get_settings

    try:
        names = MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(
            ModelRunners(get_settings().model_runner), set()
        )
    except ValueError:
        return set()
    served = set()
    for name in names:
        served.add(name.value)
        hf = getattr(SupportedModels, name.name, None)
        if hf is not None:
            served.add(hf.value)
    return served


def _check_model(model: str) -> str:
    served = served_model_names()
    if served and model not in served:
        raise ValueError(
            f"model {model!r} is not served here; use one of {sorted(served)}"
        )
    return model


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseRequest):
    """POST /v1/rerank (vLLM / Cohere shape): score each document against the query."""

    model: str
    query: str
    documents: list[Union[str, RerankDocument]] = Field(min_length=1)
    top_n: Optional[int] = Field(default=None, ge=1)
    return_documents: bool = False
    # Set by RerankService.pre_process: one segment (a document text) per (query, document) pair.
    _segments: Optional[list[str]] = PrivateAttr(default=None)

    _validate_model = field_validator("model")(_check_model)

    def document_texts(self) -> list[str]:
        return [d if isinstance(d, str) else d.text for d in self.documents]


class ScoreRequest(BaseModel):
    """POST /v1/score (vLLM shape): text_1 is the query, text_2 one document or a non-empty list of them."""

    model: str
    text_1: str
    text_2: Union[str, list[str]]

    _validate_model = field_validator("model")(_check_model)

    @field_validator("text_2")
    @classmethod
    def _non_empty(cls, v):
        if isinstance(v, list) and not v:
            raise ValueError("text_2 must not be an empty list")
        return v


class RerankPairRequest(BaseRequest):
    """One (query, document) pair: the unit the scheduler batches and the runner scores."""

    model: str
    query: str
    document: str
