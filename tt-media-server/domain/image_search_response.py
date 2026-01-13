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


class ImageClassificationResult(BaseModel):
    """Image classification result with top prediction and full output."""

    top1_class_label: str = Field(description="Top predicted class label")
    top1_class_probability: str = Field(description="Top prediction probability string")
    output: Union[dict, str] = Field(
        description="Full prediction output (dict for JSON, comma-separated string for verbose)"
    )


class ImageSearchResponse(BaseModel):
    """Response for image search API endpoint."""

    image_data: Union[
        List[ImageClassificationResult], List[List[ImagePrediction]], List[str]
    ]
    status: str = "success"
