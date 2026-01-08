# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List, Union

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    """Single image classification prediction."""

    object: str = Field(description="Predicted class label")
    confidence_level: float = Field(
        ge=0.0, le=100.0, description="Confidence percentage (0-100)"
    )


class ImageSearchResponse(BaseModel):
    """Response for image search API endpoint."""

    image_data: Union[List[List[ImagePrediction]], List[str]]
    status: str = "success"
