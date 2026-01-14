from typing import TypeAlias
from pydantic import Field
from domain.base_request import BaseRequest


class TokenizeCompletionRequest(BaseRequest):
    model: str | None = None
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
