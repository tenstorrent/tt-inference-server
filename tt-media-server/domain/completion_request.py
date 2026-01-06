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
    suffix: str | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = 0.0
    presence_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None

    # Logging and debugging
    logprobs: int | None = None

    # Stopping criteria
    stop: Union[str, list[str], None] = []

    # Reproducibility
    seed: int | None = None

    # User identifier (for monitoring/abuse prevention)
    user: str | None = None
