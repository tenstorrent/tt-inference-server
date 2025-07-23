from pydantic import BaseModel, Field, PrivateAttr

# from domain.output_format import OutputFormat

class ImageGenerateRequest(BaseModel):
    prompt: str
    # output_format: OutputFormat
    num_inference_step: int = Field(..., ge=1, le=50)
    _task_id: str = PrivateAttr()