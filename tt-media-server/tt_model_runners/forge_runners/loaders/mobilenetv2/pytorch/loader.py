# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV2 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    MobileNetV2ForSemanticSegmentation,
)

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
from ...tools.utils import print_compiled_model_results, get_label_for_index
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


@dataclass
class MobileNetV2Config(ModelConfig):
    """Configuration specific to MobileNetV2 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available MobileNetV2 model variants."""

    # TORCH_HUB variants
    MOBILENET_V2_TORCH_HUB = "mobilenet_v2"

    # HuggingFace variants
    MOBILENET_V2_035_96_HF = "google/mobilenet_v2_0.35_96"
    MOBILENET_V2_075_160_HF = "google/mobilenet_v2_0.75_160"
    MOBILENET_V2_100_224_HF = "google/mobilenet_v2_1.0_224"
    DEEPLABV3_MOBILENET_V2_HF = "google/deeplabv3_mobilenet_v2_1.0_513"

    # TIMM variants
    MOBILENET_V2_100_TIMM = "mobilenetv2_100"

    # TORCHVISION variants
    MOBILENET_V2_TORCHVISION = "mobilenet_v2_torchvision"


class ModelLoader(ForgeModel):
    """MobileNetV2 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # TORCH_HUB variants
        ModelVariant.MOBILENET_V2_TORCH_HUB: MobileNetV2Config(
            pretrained_model_name="mobilenet_v2",
            source=ModelSource.TORCH_HUB,
        ),
        # HuggingFace variants
        ModelVariant.MOBILENET_V2_035_96_HF: MobileNetV2Config(
            pretrained_model_name="google/mobilenet_v2_0.35_96",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.MOBILENET_V2_075_160_HF: MobileNetV2Config(
            pretrained_model_name="google/mobilenet_v2_0.75_160",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.MOBILENET_V2_100_224_HF: MobileNetV2Config(
            pretrained_model_name="google/mobilenet_v2_1.0_224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.DEEPLABV3_MOBILENET_V2_HF: MobileNetV2Config(
            pretrained_model_name="google/deeplabv3_mobilenet_v2_1.0_513",
            source=ModelSource.HUGGING_FACE,
        ),
        # TIMM variants
        ModelVariant.MOBILENET_V2_100_TIMM: MobileNetV2Config(
            pretrained_model_name="mobilenetv2_100",
            source=ModelSource.TIMM,
        ),
        # TORCHVISION variants
        ModelVariant.MOBILENET_V2_TORCHVISION: MobileNetV2Config(
            pretrained_model_name="mobilenet_v2",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOBILENET_V2_TORCH_HUB

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        if variant in [ModelVariant.MOBILENET_V2_TORCH_HUB]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="mobilenetv2",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained MobileNetV2 model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MobileNetV2 model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCH_HUB:
            # Load model using torch hub
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )

        elif source == ModelSource.HUGGING_FACE:
            # Load model using HuggingFace transformers
            if "deeplabv3" in model_name.lower():
                # For DeepLabV3 models, use semantic segmentation model
                model = MobileNetV2ForSemanticSegmentation.from_pretrained(model_name)
            else:
                # For standard classification models
                model = AutoModelForImageClassification.from_pretrained(model_name)

        elif source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)

        elif source == ModelSource.TORCHVISION:
            # Load model using torchvision
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        model.eval()

        # Cache model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model


    def image_to_input(self, image):
        """Convert a PIL Image to a model input tensor."""
        return self.load_inputs(image)

        
    def load_inputs(self, image, dtype_override=None, batch_size=1):
        """Prepare sample input for MobileNetV2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for MobileNetV2.
        """
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:

            model_name = self._variant_config.pretrained_model_name
            preprocessor = AutoImageProcessor.from_pretrained(model_name)
            # image_file = get_file(
            #     "http://images.cocodataset.org/val2017/000000039769.jpg"
            # )
            # image = Image.open(str(image_file))
            input_dict = preprocessor(images=image, return_tensors="pt")
            inputs = input_dict.pixel_values

        elif source == ModelSource.TIMM:

            # image_file = get_file(
            #     "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
            # )
            # image = Image.open(image_file).convert("RGB")

            # Use cached model if available, otherwise load it
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override)

            # Preprocess image using model's data config
            data_config = resolve_data_config({}, model=model_for_config)
            timm_transforms = create_transform(**data_config)
            inputs = timm_transforms(image).unsqueeze(0)

        else:

            # image_file = get_file("test_images/mobilenetv2.jpg")
            # image = Image.open(image_file).convert("RGB")

            # Preprocess image
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs


    def output_to_prediction(self, output):
        """Convert model output tensor to human-readable predictions dictionary."""
        
        compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
            output[0].softmax(dim=0) * 100, k=1
        )
        compiled_model_top1_class_idx = compiled_model_top1_class_indices[0].item()
        compiled_model_top1_class_prob = compiled_model_top1_probabilities[0].item()
        predicted_label = get_label_for_index(compiled_model_top1_class_idx)

        print(f"Predicted class index: {compiled_model_top1_class_idx}, label: {predicted_label}, confidence: {compiled_model_top1_class_prob:.4f}%")
            
        return [{
            "top1_class_label": predicted_label,
            "top1_class_probability": f"{compiled_model_top1_class_prob:.4f}%"
        }]
