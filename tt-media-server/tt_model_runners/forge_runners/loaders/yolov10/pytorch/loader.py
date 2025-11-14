# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv10 model loader implementation
"""
import torch
from torchvision import transforms
from datasets import load_dataset
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from torch.hub import load_state_dict_from_url
from ultralytics.nn.tasks import DetectionModel
from ...tools.utils import print_compiled_model_results, get_label_for_index, yolo_postprocess, yolo_prediction

import requests
import yaml
from ultralytics.nn.modules.head import Detect

class ModelVariant(StrEnum):
    """Available YOLOv10 model variants."""

    YOLOV10X = "yolov10x"
    YOLOV10N = "yolov10n"


class ModelLoader(ForgeModel):
    """YOLOv10 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV10X: ModelConfig(
            pretrained_model_name="yolov10x",
        ),
        ModelVariant.YOLOV10N: ModelConfig(
            pretrained_model_name="yolov10n",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV10X

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.coco_yaml = None  # Lazy load COCO YAML on first use

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [ModelVariant.YOLOV10X]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="yolov10",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv10 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv10 model instance.
        """
        # Get the model name from the instance's variant config
        variant = self._variant_config.pretrained_model_name
        weights = load_state_dict_from_url(
            f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{variant}.pt",
            map_location="cpu",
        )
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())

        # https://github.com/tenstorrent/tt-xla/issues/1692
        model.end2end = False
        model.model[-1].end2end = False

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model


    def load_inputs(self, image, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv10 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        batch_tensor = image_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor


    def post_process(self, co_out):
        """Post-process YOLOv10 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv10 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolo_postprocess(co_out)

        
    def output_to_prediction(self, output):
        """Convert model output tensor to human-readable predictions dictionary."""
        return yolo_prediction(output)