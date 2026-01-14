# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import TypeAlias
from pydantic import Field
from domain.base_request import BaseRequest
from config.vllm_settings import VLLMSettings


class TokenizeCompletionRequest(BaseRequest):
    model: str = VLLMSettings.model.value
    prompt: str

    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )
    return_token_strs: bool | None = Field(
        default=False,
        description=(
            "If true, also return the token strings corresponding to the token ids."
        ),
    )


"""
TODO:
Once we implement chat API, we need to implement a TokenizeChatRequest class.
"""


class TokenizeChatRequest(BaseRequest):
    def __init__(self):
        raise NotImplementedError("TokenizeChatRequest is not implemented yet")


TokenizeRequest: TypeAlias = TokenizeCompletionRequest | TokenizeChatRequest
