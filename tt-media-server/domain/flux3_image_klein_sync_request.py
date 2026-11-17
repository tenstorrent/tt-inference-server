# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The image API's requests (ImageGenerateRequest and its subclasses) as FLUX.3
Image klein's synchronous /v1/images routes take them."""

from typing import Optional

from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import BaseImageRequest, ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
from pydantic import BaseModel, Field, field_validator


class _KleinImageFields(BaseModel):
    """The image API's fields as Klein takes them, over BaseImageRequest's."""

    width: Optional[int] = Field(default=None, ge=256, le=4096)
    height: Optional[int] = Field(default=None, ge=256, le=4096)
    # Klein's own: 16 steps, no guidance or negative prompt. Ignored.
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None

    # Replaces BaseImageRequest's check by name: steps are ignored.
    @field_validator("num_inference_steps")
    @classmethod
    def validate_num_inference_steps(cls, v):
        return v

    # One render per request keeps a call within seconds; four in a row would
    # hold the connection for half a minute.
    @field_validator("number_of_images", check_fields=False)
    @classmethod
    def _one_image(cls, v):
        if v not in (None, 1):
            raise ValueError(
                "FLUX.3 Image klein makes 1 image per request; send one request "
                "per image"
            )
        return v


class KleinImageGenerateRequest(_KleinImageFields, ImageGenerateRequest):
    pass


def _no_mask(v):
    if v:
        raise ValueError(
            "FLUX.3 Image klein does not inpaint with a mask; send the image "
            "to /v1/images/image-to-image without one"
        )
    return v


class KleinImageToImageRequest(_KleinImageFields, ImageToImageRequest):
    # Not in ImageToImageRequest, so it would pass unread.
    mask: Optional[str] = None

    _no_mask = field_validator("mask")(_no_mask)


class KleinImageEditRequest(_KleinImageFields, ImageEditRequest):
    _no_mask = field_validator("mask")(_no_mask)


class KleinImageRequest(_KleinImageFields, BaseImageRequest):
    """What the routes take together, as GET /v1/models lists it: with
    `image` (and `strength`, ignored) on /v1/images/image-to-image. Without
    ImageGenerateRequest's other fields, which Klein ignores: Console offers
    a LoRA control for a schema with lora_path and lora_scale."""

    image: Optional[str] = None
    # Klein has no strength. Console offers image to image only for a schema
    # whose strength has a numeric default: ImageToImageRequest's.
    strength: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
