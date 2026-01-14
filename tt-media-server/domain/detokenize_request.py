from domain.base_request import BaseRequest


class DetokenizeRequest(BaseRequest):
    model: str | None = None
    tokens: list[int]
