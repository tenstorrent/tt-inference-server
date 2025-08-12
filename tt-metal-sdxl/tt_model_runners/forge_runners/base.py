# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Base class for model loaders.

This module provides the ForgeModel base class with common functionality
for loading models, inputs, etc.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Type, Any

from .config import ModelConfig, ModelInfo, StrEnum
import torch


class ForgeModel(ABC):
    """Base class for all TT-Forge model loaders."""

    # This is intended to be overridden by subclasses to define available model variants
    # Format: {ModelVariant: ModelConfig(...), ...}
    _VARIANTS: Dict[
        StrEnum, ModelConfig
    ] = {}  # Empty by default for models without variants
    DEFAULT_VARIANT = None

    def __init__(self, variant=None):
        """Initialize a ForgeModel instance.

        Args:
            variant: Optional StrEnum value specifying which variant to use.
                    If None, the default variant will be used.
        """
        # Validate and store the variant for this instance
        self._variant = self._validate_variant(variant)

        # Cache the variant configuration for efficiency
        self._variant_config = self.get_variant_config(variant)

    @classmethod
    def query_available_variants(cls) -> Dict[StrEnum, ModelConfig]:
        """Returns a dictionary of available model variants and their configs.

        Returns:
            Dict[StrEnum, ModelConfig]:
                Mapping of variant enum members to their ModelConfig,
                or an empty dict if the model doesn't support variants.

        """
        if not cls._VARIANTS:
            return {}
        return cls._VARIANTS

    @classmethod
    def _validate_variant(cls, variant: Optional[StrEnum] = None) -> Optional[StrEnum]:
        """Validates and returns the variant to use.

        Args:
            variant: Optional StrEnum specifying which variant to validate.

        Returns:
            StrEnum or None: Validated variant, or None for models without variants.

        Raises:
            TypeError: If caller passed something other than StrEnum/Variants
            ValueError: If the specified variant doesn't exist.

        """

        # If model doesn't support variants, return None
        if not cls._VARIANTS:
            return None

        # Use default if none specified
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Enforce user provided correct variant type (same as default variant)
        if not isinstance(variant, type(cls.DEFAULT_VARIANT)):
            raise TypeError(
                f"variant must be a {type(cls.DEFAULT_VARIANT).__name__}, not {type(variant).__name__}"
            )

        # Validate the variant exists
        if variant not in cls._VARIANTS:
            valid_variants = [v.value for v in cls._VARIANTS.keys()]
            raise ValueError(
                f"Invalid variant '{variant}'. Available variants: {valid_variants}"
            )

        return variant

    @classmethod
    def get_variant_config(
        cls, variant: Optional[StrEnum] = None
    ) -> Optional[ModelConfig]:
        """Get configuration for a specific variant after validation.

        Args:
            variant: Optional StrEnum specifying which variant to get config for.

        Returns:
            ModelConfig or None: Variant configuration object, or None for models without variants.
        """
        variant = cls._validate_variant(variant)
        if variant is None:
            return None

        return cls._VARIANTS[variant]

    @classmethod
    def get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional StrEnum specifying which variant to get info for.
                     If None, DEFAULT_VARIANT is used.


        Returns:
            ModelInfo: Information about the model and variant
        """
        variant_enum = cls._validate_variant(variant)
        return cls._get_model_info(variant_enum)

    @classmethod
    @abstractmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional StrEnum specifying which variant to get info for.

        Returns:
            ModelInfo: Information about the model and variant
        """
        raise NotImplementedError("Subclasses must implement _get_model_info")

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and return the model instance using this instance's variant.

        Args:
            **kwargs: Additional model-specific arguments.

        Returns:
            torch.nn.Module: The model instance
        """
        pass

    @abstractmethod
    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the model using this instance's variant.

        Args:
            **kwargs: Additional input-specific arguments.

        Returns:
            Any: Sample inputs that can be fed to the model
        """
        pass

    @classmethod
    def decode_output(cls, **kwargs):
        """Decode model outputs into a human-readable format if applicable.

        This method is optional - only text-based models typically need it.
        Default implementation returns the outputs unchanged.

        Args:
            **kwargs: Model-specific arguments like outputs, inputs, etc.

        Returns:
            Any: Decoded outputs or raw outputs if decoding is not implemented
        """
        # Default implementation just returns outputs if present in kwargs
        return kwargs.get("outputs", None)