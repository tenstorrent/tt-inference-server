from pydantic import BaseModel


class DetokenizeResponse(BaseModel):
    prompt: str
