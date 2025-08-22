# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Resnet model loader implementation for question answering
"""
import json
import os
from tabulate import tabulate
import torch

from transformers import ResNetForImageClassification
from torchvision import transforms
from torchvision.transforms import functional as F

from config.constants import SupportedModels

from .config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from .base import ForgeModel


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
            model="resnet",
            variant=variant_name,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # TODO channge / cleanup this
        use_1k_labels = True  # Use 1k labels by default

        current_dir = os.path.dirname(__file__)

        if use_1k_labels:            
            imagenet_class_index_path = os.path.join(current_dir, "imagenet_class_index.json")
        else:
            imagenet_class_index_path = os.path.join(current_dir, "imagenet_class_list.txt")

        self.class_labels = self.load_class_labels(imagenet_class_index_path)

        # Configuration parameters
        self.model_name = SupportedModels.MICROSOFT_RESNET_50.value
        self.input_shape = (3, 224, 224)

    def load_model(self, dtype_override=None):
        """Load a Resnet model from Hugging Face."""
        model = ResNetForImageClassification.from_pretrained(
            self.model_name, return_dict=False
        )

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, input_image, dtype_override=None):
        """Generate sample inputs for Resnet models."""

        self.logger.info(
            f"Generating sample inputs for Resnet model with shape {self.input_shape}"
        )

        # Create a random input tensor with the correct shape, using default dtype
        #inputs = torch.rand(1, *self.input_shape)
        inputs = self.image_to_tensor(
            image=input_image,
            input_shape=self.input_shape,
        )
        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        return self.print_compiled_model_results(compiled_model_out)


    def image_to_tensor(self, image, input_shape=(3, 224, 224)):
        """Convert an image to a tensor suitable for ResNet models.
        
        Args:
            image_path: Path to image file or PIL Image object
            input_shape: Model's expected input shape (channels, height, width)
        
        Returns:
            torch.Tensor: Normalized image tensor with batch dimension
        """
        
        # Define preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms and add batch dimension
        tensor = preprocess(image).unsqueeze(0)
        
        self.logger.info(f"Converting image to tensor with shape {input_shape} done")

        return tensor

    def load_class_labels(self, file_path):
        """Load class labels from a JSON or TXT file."""
        if file_path.endswith(".json"):
            with open(file_path, "r") as f:
                class_idx = json.load(f)
            return [class_idx[str(i)][1] for i in range(len(class_idx))]
        elif file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                return [line.strip() for line in f if line.strip()]


    def print_compiled_model_results(self,compiled_model_out, use_1k_labels: bool = True):

        # Get top-1 class index and probability
        compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
            compiled_model_out[0].softmax(dim=1) * 100, k=1
        )
        compiled_model_top1_class_idx = compiled_model_top1_class_indices[0, 0].item()
        compiled_model_top1_class_prob = compiled_model_top1_probabilities[0, 0].item()

        # Get class label
        compiled_model_top1_class_label = self.class_labels[compiled_model_top1_class_idx]

        table = [
            ["Metric", "Compiled Model"],
            ["Top 1 Predicted Class Label", compiled_model_top1_class_label],
            ["Top 1 Predicted Class Probability", compiled_model_top1_class_prob],
        ]
        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        # Prepare results
        return {
            "top1_class_label": compiled_model_top1_class_label,
            "top1_class_probability": compiled_model_top1_class_prob
        }
