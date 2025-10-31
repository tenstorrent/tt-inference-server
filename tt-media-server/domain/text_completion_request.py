from domain.base_request import BaseRequest


class TextCompletionRequest(BaseRequest):
    text: str