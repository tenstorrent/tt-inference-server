# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import List, Optional

from domain.image_generate_request import BaseImageRequest
from pydantic import Field, field_validator

# The generation API's FLUX.2 klein 4B schema (docs/api/generation/models/
# black-forest-labs/flux.2-klein-4b.yaml): 0 to 4 reference images, 1K and 2K
# canvases. tt-metal's pipeline takes any side that is a multiple of 16.
FLUX2_KLEIN_MAX_REFERENCES = 4
FLUX2_KLEIN_MAX_SIDE = 2048
# The distilled model's own sampling: 4 steps, no CFG.
FLUX2_KLEIN_DEFAULT_STEPS = 4


class Flux2KleinImageRequest(BaseImageRequest):
    """Text to image with up to four reference images (editing with one or more).

    ``images`` are base64 or data-URI images. ``width`` / ``height`` default to
    the first reference's size, else 1024 x 1024, as the pipeline does.
    """

    images: List[str] = Field(
        default_factory=list, max_length=FLUX2_KLEIN_MAX_REFERENCES
    )
    num_inference_steps: Optional[int] = Field(
        default=FLUX2_KLEIN_DEFAULT_STEPS, ge=4, le=50
    )
    width: Optional[int] = Field(
        default=None, ge=256, le=FLUX2_KLEIN_MAX_SIDE, multiple_of=16
    )
    height: Optional[int] = Field(
        default=None, ge=256, le=FLUX2_KLEIN_MAX_SIDE, multiple_of=16
    )

    @field_validator("images")
    @classmethod
    def _no_urls(cls, v):
        # /v1/images/generations does not download media: a URL would reach the
        # worker's decoder and fail there (a 500 that counts towards a restart).
        for image in v:
            if image[:8].lower().startswith(("http://", "https://")):
                raise ValueError("images must be base64 or data URIs, not URLs")
        return v
