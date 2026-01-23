# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Union

from domain.base_request import BaseRequest
from pydantic import BaseModel


class StreamOptions(BaseModel):
    """Stream options for OpenAI-compatible streaming responses."""

    include_usage: bool | None = True
    continuous_usage_stats: bool | None = False


class CompletionRequest(BaseRequest):
    """
    OpenAI-compatible completion request.
    Based on OpenAI API specification: https://platform.openai.com/docs/api-reference/completions/create
    """

    # Model identifier
    model: str | None = None

    prompt: str

    # Response configuration
    echo: bool | None = False
    max_tokens: int | None = 16
    n: int = 1
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    suffix: str | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    # Stopping criteria
    stop: Union[str, list[str], None] = []

    # Reproducibility
    seed: int | None = None

    # sampling-params
    temperature: float | None = None
    top_p: float | None = None

    # Logging and debugging
    logprobs: int | None = None
    # User identifier (for monitoring/abuse prevention)
    user: str | None = None

    # completion-sampling-params
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float = 1.0
    stop_token_ids: list[int] | None = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    allowed_token_ids: list[int] | None = None
    prompt_logprobs: int | None = None
