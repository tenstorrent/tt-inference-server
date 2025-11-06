# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VovNet model loader implementation
"""
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from pytorchcv.model_provider import get_model as ptcv_get_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from .src.utils import download_model, preprocess_steps, preprocess_timm_model
from .src.src_vovnet_stigma import (
    vovnet39 as stigma_vovnet39,
    vovnet57 as stigma_vovnet57,
)
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from ...tools.utils import print_compiled_model_results, get_label_for_index
from dataclasses import dataclass
from loguru import logger


@dataclass
class VovNetConfig(ModelConfig):
    source: ModelSource


class ModelVariant(StrEnum):
    """Available VovNet model variants."""

    # OSMR (pytorchcv) image classification variants
    VOVNET27S = "vovnet27s"
    VOVNET39 = "vovnet39"
    VOVNET57 = "vovnet57"

    # TorchHub variant
    VOVNET39_TORCHHUB = "vovnet39_th"
    VOVNET57_TORCHHUB = "vovnet57_th"

    # TIMM image classification variants (subset)
    TIMM_VOVNET19B_DW = "ese_vovnet19b_dw"
    TIMM_VOVNET39B = "ese_vovnet39b"
    TIMM_VOVNET99B = "ese_vovnet99b"
    TIMM_VOVNET19B_DW_RAIN1K = "ese_vovnet19b_dw.ra_in1k"


class ModelLoader(ForgeModel):
    """VovNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # OSMR variants
        ModelVariant.VOVNET27S: VovNetConfig(
            pretrained_model_name="vovnet27s", source=ModelSource.OSMR
        ),
        ModelVariant.VOVNET39: VovNetConfig(
            pretrained_model_name="vovnet39", source=ModelSource.OSMR
        ),
        ModelVariant.VOVNET57: VovNetConfig(
            pretrained_model_name="vovnet57", source=ModelSource.OSMR
        ),
        # TorchHub
        ModelVariant.VOVNET39_TORCHHUB: VovNetConfig(
            pretrained_model_name="vovnet39", source=ModelSource.TORCH_HUB
        ),
        ModelVariant.VOVNET57_TORCHHUB: VovNetConfig(
            pretrained_model_name="vovnet57", source=ModelSource.TORCH_HUB
        ),
        # TIMM variants
        ModelVariant.TIMM_VOVNET19B_DW: VovNetConfig(
            pretrained_model_name="ese_vovnet19b_dw", source=ModelSource.TIMM
        ),
        ModelVariant.TIMM_VOVNET39B: VovNetConfig(
            pretrained_model_name="ese_vovnet39b", source=ModelSource.TIMM
        ),
        ModelVariant.TIMM_VOVNET99B: VovNetConfig(
            pretrained_model_name="ese_vovnet99b", source=ModelSource.TIMM
        ),
        ModelVariant.TIMM_VOVNET19B_DW_RAIN1K: VovNetConfig(
            pretrained_model_name="ese_vovnet19b_dw.ra_in1k", source=ModelSource.TIMM
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VOVNET27S

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.input_shape = (3, 224, 224)
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
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

        if variant in [ModelVariant.TIMM_VOVNET19B_DW_RAIN1K]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="vovnet",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load a VovNet model based on the configured source for this variant."""
        cfg = self._variant_config
        model_name = cfg.pretrained_model_name
        source = cfg.source

        if source == ModelSource.OSMR:
            model = ptcv_get_model(model_name, pretrained=True)
        elif source == ModelSource.TIMM:
            model, _ = download_model(preprocess_timm_model, model_name)
        elif source == ModelSource.TORCH_HUB:
            model_fn = stigma_vovnet39 if "39" in model_name else stigma_vovnet57
            model, _ = download_model(preprocess_steps, model_fn)
        else:
            raise ValueError(f"Unsupported source: {source}")

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        # Cache for TIMM preprocessing resolution
        self._cached_model = model
        return model


    def load_inputs(self, image, dtype_override=None, batch_size=1):
        """Generate sample inputs for VovNet models.

        Uses TIMM-specific preprocessing when the source is TIMM, otherwise
        applies standard ImageNet preprocessing.
        """
        # file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        # input_image = Image.open(file_path).convert("RGB")

        if self._variant_config.source == ModelSource.TIMM:
            model_for_config = (
                self._cached_model if self._cached_model is not None else None
            )
            if model_for_config is None:
                # Ensure model is available to resolve preprocess
                model_for_config = self.load_model(dtype_override)
            try:
                data_config = resolve_data_config({}, model=model_for_config)
                data_transforms = create_transform(**data_config)
                inputs = data_transforms(image).unsqueeze(0)
            except Exception as e:
                logger.warning(
                    f"Failed to resolve TIMM data config: {e}. Falling back to standard ImageNet preprocessing."
                )
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
        elif self._variant_config.source == ModelSource.TORCH_HUB:
            model_fn = (
                stigma_vovnet39
                if "39" in self._variant_config.pretrained_model_name
                else stigma_vovnet57
            )
            _, inputs = download_model(preprocess_steps, model_fn)
        else:
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

        # Replicate for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

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