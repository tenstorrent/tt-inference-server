# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration classes for ForgeModel implementations
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class StrEnum(Enum):
    """Enum with string representation matching its value"""

    def __str__(self) -> str:
        return self.value


class ModelGroup(StrEnum):
    """Model groups for categorization and reporting"""

    GENERALITY = "generality"
    RED = "red"
    PRIORITY = "priority"


class ModelTask(StrEnum):
    """
    Classification of tasks models can perform.

    Implemented based on
    https://huggingface.co/docs/transformers/en/tasks/sequence_classification.
    """

    NLP_TEXT_CLS = "nlp_text_cls"
    NLP_TOKEN_CLS = "nlp_token_cls"
    NLP_EMBED_GEN = "nlp_embed_gen"
    NLP_QA = "nlp_qa"
    NLP_CAUSAL_LM = "nlp_causal_lm"
    NLP_MASKED_LM = "nlp_masked_lm"
    NLP_TRANSLATION = "nlp_translation"
    NLP_SUMMARIZATION = "nlp_summarization"
    NLP_MULTI_CHOICE = "nlp_multi_choice"
    AUDIO_CLS = "audio_cls"
    AUDIO_ASR = "audio_asr"
    CV_IMAGE_CLS = "cv_image_cls"
    CV_IMAGE_SEG = "cv_image_seg"
    CV_VIDEO_CLS = "cv_video_cls"
    CV_OBJECT_DET = "cv_object_det"
    CV_ZS_OBJECT_DET = "cv_zs_object_det"
    CV_ZS_IMAGE_CLS = "cv_zs_image_cls"
    CV_DEPTH_EST = "cv_depth_est"
    CV_IMG_TO_IMG = "cv_img_to_img"
    CV_IMAGE_FE = "cv_image_fe"
    CV_MASK_GEN = "cv_mask_gen"
    CV_KEYPOINT_DET = "cv_keypoint_det"
    CV_KNOW_DISTILL = "cv_know_distill"
    CV_PANOPTIC_SEG = "cv_panoptic_seg"
    REALTIME_MAP_CONSTRUCTION = "realtime_map_construction"
    MM_IMAGE_CAPT = "mm_image_capt"
    MM_DOC_QA = "mm_doc_qa"
    MM_VISUAL_QA = "mm_visual_qa"
    MM_TTS = "mm_tts"
    MM_IMAGE_TTT = "mm_image_ttt"
    MM_VIDEO_TTT = "mm_video_ttt"
    MM_MASKED_LM = "mm_masked_lm"
    MM_CAUSAL_LM = "mm_causal_lm"
    MM_ACTION_PREDICTION = "mm_action_prediction"
    MM_CONDITIONAL_GENERATION = "mm_conditional_generation"
    CONDITIONAL_GENERATION = "conditional_generation"
    ATOMIC_ML = "atomic_ml"


class ModelSource(StrEnum):
    """Where the model was sourced from"""

    HUGGING_FACE = "huggingface"
    TORCH_HUB = "torch_hub"
    CUSTOM = "custom"
    TORCHVISION = "torchvision"
    TIMM = "timm"
    TORCH_XRAY_VISION = "torchxrayvision"
    OSMR = "osmr"
    GITHUB = "github"


class Framework(StrEnum):
    """Framework the model is implemented in"""

    JAX = "jax"
    TORCH = "pytorch"
    NUMPY = "numpy"
    ONNX = "onnx"


class Parallelism(StrEnum):
    """Multi-device parallelism strategy the model is using."""

    SINGLE_DEVICE = "single_device"
    DATA_PARALLEL = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"


@dataclass(frozen=True)
class ModelInfo:
    """
    Dashboard/reporting metadata about a model.
    Used for categorization and metrics tracking.
    """

    model: str
    variant: StrEnum
    group: ModelGroup
    task: ModelTask
    source: ModelSource
    framework: Framework

    def __post_init__(self):
        if not isinstance(self.variant, StrEnum):
            # TODO - Change to raise TypeError once all models updated.
            print(
                f"Warning: ModelInfo.variant should be a StrEnum, not {type(self.variant).__name__}"
            )

    @property
    def name(self) -> str:
        """Generate a standardized model identifier"""
        return f"{self.framework}_{self.model}_{self.variant}_{self.task}_{self.source}"

    def to_report_dict(self) -> dict:
        """Represents self as dict suitable for pytest reporting pipeline."""
        return {
            "task": str(self.task),
            "source": str(self.source),
            "framework": str(self.framework),
            "model_arch": self.model,
            "variant_name": str(self.variant),
        }


@dataclass
class ModelConfig:
    """
    Base configuration for model variants.
    Contains common configuration parameters that apply across model types.
    """

    pretrained_model_name: str

    # Common configuration fields shared across models

    def __post_init__(self):
        """Validate required fields after initialization"""
        pass


@dataclass
class LLMModelConfig(ModelConfig):
    """Configuration specific to language models"""

    max_length: Optional[int] = None
    attention_mechanism: Optional[str] = None
    sliding_window: Optional[int] = None

    # Additional LLM-specific configuration
