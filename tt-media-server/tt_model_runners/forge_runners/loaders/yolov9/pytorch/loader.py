# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv9 model loader implementation
"""
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from torch.hub import load_state_dict_from_url
from ultralytics.nn.tasks import DetectionModel
from torchvision import transforms
from datasets import load_dataset
from ...tools.utils import yolo_postprocess, yolo_prediction


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="yolov9",
            variant=variant_name,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    """YOLOv9 model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_variant = "yolov9c"

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv9 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv9 model instance.
        """

        variant = self.model_variant
        weights = load_state_dict_from_url(
            f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{variant}.pt",
            map_location="cpu",
        )
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model


    def load_inputs(self, image, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv9 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        # Load sample image and preprocess
        if not image:
            dataset = load_dataset("huggingface/cats-image", split="test[:1]")
            image = dataset[0]["image"]
            
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_process(self, co_out):
        """Post-process YOLOv9 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv9 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolo_postprocess(co_out)


    def output_to_prediction(self, output):
        """Convert model output tensor to human-readable predictions dictionary."""
        return yolo_prediction(output)
    