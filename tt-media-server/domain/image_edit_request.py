# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from config.settings import get_settings
from domain.image_to_image_request import ImageToImageRequest
from pydantic import Field, field_validator, model_validator

# Runners whose edit path requires a mask (mask-based inpainting). Other runners
# (e.g. FLUX.1-Kontext, which edits by instruction only) accept a missing mask.
_MASK_REQUIRED_RUNNERS = {"tt-sdxl-edit"}
_QWEN_IMAGE_EDIT_RUNNER = "tt-qwen-image-edit"


class ImageEditRequest(ImageToImageRequest):
    # Optional so the shared /edits endpoint serves both mask-based edits (SDXL)
    # and instruction-only edits with no mask (FLUX.1-Kontext). validate_default
    # so the per-runner check below also fires when mask is omitted entirely.
    mask: Optional[str] = Field(default=None, validate_default=True)

    @field_validator("mask")
    @classmethod
    def _require_mask_when_runner_needs_it(cls, v):
        # Reject a missing mask at validation time (clean 422) for runners that
        # require one, instead of letting it fail later in the runner's mask
        # preprocessing (500). Kontext and other runners still allow None.
        if v is None and get_settings().model_runner in _MASK_REQUIRED_RUNNERS:
            raise ValueError(f"mask is required for {get_settings().model_runner}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _qwen_image_edit_default_cfg(cls, data):
        # The inherited guidance_scale default (5.0) is SDXL's. Fill in the
        # Qwen-Image-Edit reference true-CFG scale here, as a field value, so it
        # survives model_dump() into the per-image segment requests.
        if (
            isinstance(data, dict)
            and data.get("guidance_scale") is None
            and get_settings().model_runner == _QWEN_IMAGE_EDIT_RUNNER
        ):
            from tt_model_runners.qwen_image_edit_policy import (
                QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE,
            )

            data = {**data, "guidance_scale": QWEN_IMAGE_EDIT_DEFAULT_TRUE_CFG_SCALE}
        return data

    @model_validator(mode="after")
    def _qwen_image_edit_square_canvas(self):
        # Clean 422 for a canvas the pipeline cannot render, instead of a 500
        # from the worker.
        if get_settings().model_runner == _QWEN_IMAGE_EDIT_RUNNER:
            from tt_model_runners.qwen_image_edit_policy import qwen_image_edit_side

            qwen_image_edit_side(self.width, self.height)
        return self
