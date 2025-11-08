# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Base model wrapper interface for TT-Comfy Bridge.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    All model wrappers should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.loaded = False
        self.logger = logger
    
    @abstractmethod
    async def load_model(self):
        """
        Load the model onto the device.
        
        This method should handle model initialization, weight loading,
        and device setup. It may take several minutes for large models.
        """
        pass
    
    @abstractmethod
    async def encode_prompts(self, prompt: str, negative_prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Encode text prompts to embeddings.
        
        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing encoded prompt data
        """
        pass
    
    @abstractmethod
    async def run_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full inference pipeline.
        
        Args:
            request_data: Dictionary with inference parameters
            
        Returns:
            Dictionary containing inference results (typically images as base64)
        """
        pass
    
    async def cleanup(self):
        """
        Cleanup model resources.
        
        Override this method to implement custom cleanup logic.
        """
        self.loaded = False
        self.logger.info(f"Cleaned up model on device {self.device_id}")

