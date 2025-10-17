
from pydantic import Field
from typing import Optional, Union, List, Tuple
from domain.base_image_generate_request import BaseImageGenerateRequest

class SDXLImageGenerateRequest(BaseImageGenerateRequest):
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    crop_coords_top_left: Optional[Tuple[int, float]] = Field(default=(0, 0))
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    timesteps: Optional[List[Union[int, float]]] = None
    sigmas: Optional[List[Union[int, float]]] = None
