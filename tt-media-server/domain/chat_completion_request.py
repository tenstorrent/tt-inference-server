# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Union

from domain.base_request import BaseRequest
from domain.completion_request import StreamOptions
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionRequest(BaseRequest):
    """
    OpenAI-compatible chat completion request.
    Based on OpenAI API specification: https://platform.openai.com/docs/api-reference/chat/create
    """

    # Model identifier
    model: str | None = None

    messages: list[ChatMessage]

    # Response configuration
    max_tokens: int | None = 16
    n: int = 1
    stream: bool | None = False
    stream_options: StreamOptions | None = None

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = 0.0
    presence_penalty: float | None = 0.0

    # Stopping criteria
    stop: Union[str, list[str], None] = []

    # Reproducibility
    seed: int | None = None

    # User identifier (for monitoring/abuse prevention)
    user: str | None = None
