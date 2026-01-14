from pydantic import BaseModel


class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: list[str] | None = None
