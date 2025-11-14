# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv4 model loader implementation
"""
import torch
from datasets import load_dataset
from torchvision import transforms
from typing import Optional
import os
from ...tools.utils import get_file, yolo_prediction

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
from .src.yolov4 import Yolov4
from .src.post_processing import (
    gen_yolov4_boxes_confs,
    get_region_boxes,
    post_processing,
    plot_boxes_cv2,
)


class ModelVariant(StrEnum):
    """Available YOLOv4 model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """YOLOv4 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="",  # Not used
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="yolov4",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv4 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv4 model instance.
        """
        weights_pth = get_file("test_files/pytorch/yolov4/yolov4.pth")

        # Load weights checkpoint
        state_dict = torch.load(weights_pth, map_location="cpu")

        model = Yolov4()

        # Align keys and load weights
        new_state_dict = dict(zip(model.state_dict().keys(), state_dict.values()))
        model.load_state_dict(new_state_dict)
        model.eval()

        # Apply dtype override if needed
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model


    def load_inputs(self, image, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv4 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        if not image:
            # Load dataset
            dataset = load_dataset(
                "huggingface/cats-image", split="test"
            )  # cats-image is a dataset of 1000 images of cats

            # Get first image from dataset
            image = dataset[0]["image"]

        # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
            ]
        )

        img_tensor = [transform(image).unsqueeze(0)]  # Add batch dimension
        batch_tensor = torch.cat(img_tensor, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_processing(self, co_out):
        y1, y2, y3 = gen_yolov4_boxes_confs(co_out)
        output = get_region_boxes([y1, y2, y3])
        results = post_processing(0.3, 0.4, output)
        coco_names_path = get_file(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        )
        with open(coco_names_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        # Print detected boxes info
        print("Detected boxes:")
        for box in results[0]:
            if len(box) >= 6:
                *coords, score, class_id = box[:6]  # in case there are more than 6
                x1, y1, x2, y2 = coords
                class_name = class_names[int(class_id)]
                print(
                    f"Class: {class_name}, Score: {score:.2f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                )

        # Load dataset
        dataset = load_dataset("huggingface/cats-image", split="test").with_format(
            "np"
        )  # get the image as an numpy array

        img_cv = dataset[0]["image"]
        output_dir = "yolov4_predictions"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"yolov4_predicted.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plot_boxes_cv2(img_cv, results[0], output_path, class_names)

        return output_path

    def output_to_prediction(self, output):
        """Convert model output tensor to human-readable predictions dictionary."""
        return yolo_prediction(output)