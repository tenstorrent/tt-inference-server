# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Union

from pydantic import BaseModel


class ChatContentPart(BaseModel):
    """A single content part. Per OpenAI spec chat message content can be a
    string OR a list of these parts (multi-modal inputs). For text-only LLMs
    we consume only `type=="text"` parts; other types are accepted but ignored."""

    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    # Per OpenAI spec, content can be a string OR a list of content parts.
    # vllm bench / openai-python / lm-eval all send the list form by default.
    content: str | list[ChatContentPart]


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    See: https://platform.openai.com/docs/api-reference/chat/create
    """

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 2048
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = 0.0
    presence_penalty: float | None = 0.0
    stream: bool | None = False
    stop: Union[str, list[str], None] = []
    n: int = 1
    seed: int | None = None
    user: str | None = None

    adapter: str | None = None
