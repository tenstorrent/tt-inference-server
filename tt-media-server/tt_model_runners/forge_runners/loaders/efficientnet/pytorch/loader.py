# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet model loader implementation
"""

import torch
from typing import Optional
from dataclasses import dataclass
from PIL import Image

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
from loguru import logger
import torchvision.models as models
from torchvision.models._api import WeightsEnum
from ...tools.utils import get_file, print_compiled_model_results, get_state_dict, get_label_for_index
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


@dataclass
class EfficientNetConfig(ModelConfig):
    """Configuration specific to EfficientNet models"""

    model_function: str = None
    weights_class: str = None
    use_1k_labels: bool = False
    source: ModelSource = ModelSource.TORCHVISION


class ModelVariant(StrEnum):
    """Available EfficientNet model variants."""

    # Torchvision variants
    B0 = "efficientnet_b0"
    B1 = "efficientnet_b1"
    B2 = "efficientnet_b2"
    B3 = "efficientnet_b3"
    B4 = "efficientnet_b4"
    B5 = "efficientnet_b5"
    B6 = "efficientnet_b6"
    B7 = "efficientnet_b7"

    # TIMM variants (values are identifiers for reporting; pretrained model names live in config)
    TIMM_EFFICIENTNET_B0 = "timm_efficientnet_b0"
    TIMM_EFFICIENTNET_B4 = "timm_efficientnet_b4"
    HF_TIMM_EFFICIENTNET_B0_RA_IN1K = "hf_hub_timm_efficientnet_b0_ra_in1k"
    HF_TIMM_EFFICIENTNET_B4_RA2_IN1K = "hf_hub_timm_efficientnet_b4_ra2_in1k"
    HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K = "hf_hub_timm_efficientnet_b5_in12k_ft_in1k"
    HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K = "hf_hub_timm_tf_efficientnet_b0_aa_in1k"
    HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K = "hf_hub_timm_efficientnetv2_rw_s_ra2_in1k"
    HF_TIMM_TF_EFFICIENTNETV2_S_IN21K = "hf_hub_timm_tf_efficientnetv2_s_in21k"


class ModelLoader(ForgeModel):
    """EfficientNet model loader implementation."""

    # Static dataclass instances for each variant
    B0_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b0",
        model_function="efficientnet_b0",
        weights_class="EfficientNet_B0_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B1_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b1",
        model_function="efficientnet_b1",
        weights_class="EfficientNet_B1_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B2_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b2",
        model_function="efficientnet_b2",
        weights_class="EfficientNet_B2_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B3_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b3",
        model_function="efficientnet_b3",
        weights_class="EfficientNet_B3_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B4_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b4",
        model_function="efficientnet_b4",
        weights_class="EfficientNet_B4_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B5_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b5",
        model_function="efficientnet_b5",
        weights_class="EfficientNet_B5_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B6_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b6",
        model_function="efficientnet_b6",
        weights_class="EfficientNet_B6_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B7_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b7",
        model_function="efficientnet_b7",
        weights_class="EfficientNet_B7_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    # TIMM config instances
    TIMM_EFFICIENTNET_B0_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b0",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    TIMM_EFFICIENTNET_B4_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b4",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNET_B0_RA_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnet_b0.ra_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNET_B4_RA2_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnet_b4.ra2_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnet_b5.in12k_ft_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/tf_efficientnet_b0.aa_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnetv2_rw_s.ra2_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_TF_EFFICIENTNETV2_S_IN21K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/tf_efficientnetv2_s.in21k",
        source=ModelSource.TIMM,
        use_1k_labels=False,
    )

    # Dictionary using the static dataclass instances (for compatibility with existing tests)
    _VARIANTS = {
        # Torchvision variants
        ModelVariant.B0: B0_CONFIG,
        ModelVariant.B1: B1_CONFIG,
        ModelVariant.B2: B2_CONFIG,
        ModelVariant.B3: B3_CONFIG,
        ModelVariant.B4: B4_CONFIG,
        ModelVariant.B5: B5_CONFIG,
        ModelVariant.B6: B6_CONFIG,
        ModelVariant.B7: B7_CONFIG,
        # TIMM variants
        ModelVariant.TIMM_EFFICIENTNET_B0: TIMM_EFFICIENTNET_B0_CONFIG,
        ModelVariant.TIMM_EFFICIENTNET_B4: TIMM_EFFICIENTNET_B4_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNET_B0_RA_IN1K: HF_TIMM_EFFICIENTNET_B0_RA_IN1K_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNET_B4_RA2_IN1K: HF_TIMM_EFFICIENTNET_B4_RA2_IN1K_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K: HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K_CONFIG,
        ModelVariant.HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K: HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K: HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K_CONFIG,
        ModelVariant.HF_TIMM_TF_EFFICIENTNETV2_S_IN21K: HF_TIMM_TF_EFFICIENTNETV2_S_IN21K_CONFIG,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.B0

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
        source = cls._VARIANTS[variant].source
        return ModelInfo(
            model="efficientnet",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.B0
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the EfficientNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The EfficientNet model instance.
        """
        source = self._variant_config.source

        if source == ModelSource.TORCHVISION:
            # Setup state dict function
            WeightsEnum.get_state_dict = get_state_dict

            # Get model function and weights class from dataclass config
            model_fn = getattr(models, self._variant_config.model_function)
            weights_class = getattr(models, self._variant_config.weights_class)

            # Load model with appropriate weights
            model = model_fn(weights=weights_class.IMAGENET1K_V1)
        else:
            # Load using timm
            model_name = self._variant_config.pretrained_model_name
            model = timm.create_model(model_name, pretrained=True)

        model.eval()
        # Cache model for use in load_inputs (to avoid reloading for timm transforms)
        self._cached_model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, image, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the EfficientNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for EfficientNet.
        """
        # Determine which image to use based on variant
        use_1k_labels = self._variant_config.use_1k_labels
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            # Use cached model if available, otherwise load it
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override)
            try:
                if image is not None:
                    img = image
                else:
                    if use_1k_labels:
                        file_path = get_file(
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
                        )
                        img = Image.open(file_path).convert("RGB")
                    else:
                        file_path = get_file(
                            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
                        )
                        img = Image.open(file_path).convert("RGB")
                config = resolve_data_config({}, model=model_for_config)
                timm_transforms = create_transform(**config)
                inputs = timm_transforms(img).unsqueeze(0)
            except:
                logger.warning(
                    "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
                )
                inputs = torch.rand(1, 3, 224, 224)
        else:
            if not image:
                image_file = get_file(
                    "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
                )
                image = Image.open(image_file)

            # Use standard torchvision preprocessing
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

    def print_cls_results(self, compiled_model_out):
        """Print classification results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        # Determine label set based on variant
        use_1k_labels = self._variant_config.use_1k_labels
        print_compiled_model_results(compiled_model_out, use_1k_labels=use_1k_labels)


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
    