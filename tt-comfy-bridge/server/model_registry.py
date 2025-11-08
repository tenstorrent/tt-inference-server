# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Model registry for managing loaded models.
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing loaded model instances.
    
    Maintains a dictionary of model_id -> model_wrapper mappings.
    """
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._loading = set()  # Track models currently being loaded
        
    def register(self, model_id: str, model_wrapper: Any):
        """
        Register a loaded model.
        
        Args:
            model_id: Unique identifier for the model
            model_wrapper: The model wrapper instance
        """
        if model_id in self._models:
            logger.warning(f"Overwriting existing model: {model_id}")
            
        self._models[model_id] = model_wrapper
        self._loading.discard(model_id)
        logger.info(f"Registered model: {model_id}")
    
    def unregister(self, model_id: str):
        """
        Unregister and cleanup a model.
        
        Args:
            model_id: Model identifier to unregister
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Unregistered model: {model_id}")
        else:
            logger.warning(f"Attempted to unregister non-existent model: {model_id}")
    
    def get(self, model_id: str) -> Optional[Any]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model wrapper instance or None if not found
        """
        return self._models.get(model_id)
    
    def exists(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return model_id in self._models
    
    def is_loading(self, model_id: str) -> bool:
        """Check if a model is currently being loaded."""
        return model_id in self._loading
    
    def mark_loading(self, model_id: str):
        """Mark a model as currently loading."""
        self._loading.add(model_id)
    
    def list_models(self) -> list:
        """Get list of registered model IDs."""
        return list(self._models.keys())
    
    def clear(self):
        """Clear all registered models."""
        self._models.clear()
        self._loading.clear()
        logger.info("Cleared all models from registry")

