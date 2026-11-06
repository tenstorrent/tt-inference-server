# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import Optional, Union

from domain.base_request import BaseRequest
from pydantic import BaseModel, Field, PrivateAttr, field_validator

# Most documents per request. A request becomes one scheduler task per document, so this keeps one
# request well inside the task queue (max_queue_size, 5000 by default); Cohere's rerank API has the
# same limit.
MAX_RERANK_DOCUMENTS = 1000


def _served_models() -> list[tuple[str, Optional[str]]]:
    """(short name, Hugging Face id or None) for each ModelNames the running runner serves."""
    from config.constants import (
        INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP,
        ModelRunners,
        SupportedModels,
    )
    from config.settings import get_settings

    try:
        names = INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(
            ModelRunners(get_settings().model_runner), set()
        )
    except ValueError:
        return []
    return [
        (n.value, getattr(getattr(SupportedModels, n.name, None), "value", None))
        for n in sorted(names, key=lambda n: n.value)
    ]


def served_model_names() -> set[str]:
    """Names a client may pass as ``model``: the running runner's ModelNames and their Hugging Face ids.
    Checked at validation (a 422) so a wrong name never reaches the worker, where it would be a 500
    that counts towards a worker restart."""
    return {name for pair in _served_models() for name in pair if name is not None}


def served_model_id() -> Optional[str]:
    """The name a response reports when the request left out ``model``: the Hugging Face id (not
    settings.model_weights_path, which can be a mounted local directory)."""
    served = _served_models()
    if not served:
        return None
    short, hf = served[0]
    return hf or short


def _check_model(model: Optional[str]) -> Optional[str]:
    """None (vLLM makes ``model`` optional) means the served model; a name must be one of the served names."""
    if model is None:
        return None
    served = served_model_names()
    if served and model not in served:
        raise ValueError(
            f"model {model!r} is not served here; use one of {sorted(served)}"
        )
    return model


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseRequest):
    """POST /v1/rerank (vLLM / Cohere shape): score each document against the query.
    ``top_n`` 0 or omitted returns every document (vLLM's default is 0)."""

    model: Optional[str] = None
    query: str
    documents: list[Union[str, RerankDocument]] = Field(
        min_length=1, max_length=MAX_RERANK_DOCUMENTS
    )
    top_n: Optional[int] = Field(default=None, ge=0)
    return_documents: bool = False
    # Set by RerankService.pre_process: one segment (a document text) per (query, document) pair.
    _segments: Optional[list[str]] = PrivateAttr(default=None)

    _validate_model = field_validator("model")(_check_model)

    def document_texts(self) -> list[str]:
        return [d if isinstance(d, str) else d.text for d in self.documents]


class ScoreRequest(BaseModel):
    """POST /v1/score (vLLM shape): text_1 is the query, text_2 one document or a list of
    1 to MAX_RERANK_DOCUMENTS documents."""

    model: Optional[str] = None
    text_1: str
    text_2: Union[str, list[str]]

    _validate_model = field_validator("model")(_check_model)

    @field_validator("text_2")
    @classmethod
    def _document_count(cls, v):
        if isinstance(v, list) and not 1 <= len(v) <= MAX_RERANK_DOCUMENTS:
            raise ValueError(
                f"text_2 must hold 1 to {MAX_RERANK_DOCUMENTS} documents, got {len(v)}"
            )
        return v


class RerankPairRequest(BaseRequest):
    """One (query, document) pair: the unit the scheduler batches and the runner scores."""

    model: Optional[str] = None
    query: str
    document: str
