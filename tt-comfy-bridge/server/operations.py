# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Operation handlers for TT-Comfy Bridge.
"""

import logging
import asyncio
from typing import Dict, Any

from protocol.messages import OperationType, Response, MessageStatus
from server.model_registry import ModelRegistry
from server.tensor_bridge import TensorBridge

logger = logging.getLogger(__name__)


class OperationHandler:
    """
    Handles operations requested by ComfyUI clients.
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.tensor_bridge = TensorBridge()
        
        # Map operation types to handler methods
        self._handlers = {
            OperationType.INIT_MODEL.value: self.handle_init_model,
            OperationType.UNLOAD_MODEL.value: self.handle_unload_model,
            OperationType.ENCODE_PROMPT.value: self.handle_encode_prompt,
            OperationType.FULL_INFERENCE.value: self.handle_full_inference,
            OperationType.ENCODE_IMAGE.value: self.handle_encode_image,
            OperationType.IMG2IMG_INFERENCE.value: self.handle_img2img_inference,
            OperationType.PING.value: self.handle_ping,
            OperationType.SHUTDOWN.value: self.handle_shutdown,
        }
    
    async def handle_operation(self, operation: str, data: Dict[str, Any], request_id: str = None) -> Response:
        """
        Route an operation to its handler.
        
        Args:
            operation: Operation type string
            data: Operation data
            request_id: Optional request ID for tracking
            
        Returns:
            Response object
        """
        handler = self._handlers.get(operation)
        
        if not handler:
            logger.error(f"Unknown operation: {operation}")
            return Response.error(f"Unknown operation: {operation}", request_id=request_id)
        
        try:
            logger.debug(f"Handling operation: {operation}")
            result = await handler(data)
            return Response.success(result, request_id=request_id)
            
        except Exception as e:
            logger.error(f"Error handling {operation}: {e}", exc_info=True)
            return Response.error(str(e), request_id=request_id)
    
    async def handle_init_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize and load a model.
        
        Expected data:
            - model_type: "sdxl", "sd35", "sd14"
            - config: dict with model configuration
            - device_id: optional device ID
        """
        model_type = data.get("model_type")
        config = data.get("config", {})
        device_id = data.get("device_id", "0")
        
        if not model_type:
            raise ValueError("model_type is required")
        
        model_id = f"{model_type}_{device_id}"
        
        # Check if already loaded
        if self.registry.exists(model_id):
            logger.info(f"Model {model_id} already loaded")
            return {"model_id": model_id, "status": "already_loaded"}
        
        # Check if currently loading
        if self.registry.is_loading(model_id):
            logger.info(f"Model {model_id} is currently loading")
            return {"model_id": model_id, "status": "loading"}
        
        # Mark as loading
        self.registry.mark_loading(model_id)
        
        try:
            # Import and instantiate the appropriate wrapper
            if model_type == "sdxl":
                from models.sdxl_wrapper import SDXLModelWrapper
                wrapper = SDXLModelWrapper(device_id=device_id)
            elif model_type == "sd35":
                from models.sd35_wrapper import SD35ModelWrapper
                wrapper = SD35ModelWrapper(device_id=device_id)
            elif model_type == "sd14":
                from models.sd14_wrapper import SD14ModelWrapper
                wrapper = SD14ModelWrapper(device_id=device_id)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load model (this is async and may take several minutes)
            logger.info(f"Loading model {model_id}...")
            await wrapper.load_model()
            
            # Register the loaded model
            self.registry.register(model_id, wrapper)
            
            logger.info(f"Model {model_id} loaded successfully")
            return {"model_id": model_id, "status": "loaded"}
            
        except Exception as e:
            self.registry.mark_loading(model_id)  # Remove from loading set
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def handle_unload_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Unload a model from memory."""
        model_id = data.get("model_id")
        
        if not model_id:
            raise ValueError("model_id is required")
        
        if not self.registry.exists(model_id):
            logger.warning(f"Cannot unload non-existent model: {model_id}")
            return {"model_id": model_id, "status": "not_found"}
        
        # Get wrapper and cleanup
        wrapper = self.registry.get(model_id)
        if hasattr(wrapper, 'cleanup'):
            await wrapper.cleanup()
        
        self.registry.unregister(model_id)
        
        logger.info(f"Unloaded model: {model_id}")
        return {"model_id": model_id, "status": "unloaded"}
    
    async def handle_encode_prompt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode text prompts.
        
        Expected data:
            - model_id: Model to use
            - prompt: Positive prompt
            - negative_prompt: Negative prompt
            - prompt_2: Optional second prompt (for SDXL)
            - negative_prompt_2: Optional second negative prompt
        """
        model_id = data.get("model_id")
        
        if not model_id:
            raise ValueError("model_id is required")
        
        wrapper = self.registry.get(model_id)
        if not wrapper:
            raise ValueError(f"Model not found: {model_id}")
        
        # Call wrapper's encode method
        result = await wrapper.encode_prompts(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            prompt_2=data.get("prompt_2"),
            negative_prompt_2=data.get("negative_prompt_2")
        )
        
        return result
    
    async def handle_full_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full text-to-image inference.
        
        Expected data:
            - model_id: Model to use
            - prompt: Text prompt
            - negative_prompt: Negative prompt
            - num_inference_steps: Number of denoising steps
            - guidance_scale: CFG scale
            - seed: Random seed
            - ... other model-specific parameters
        """
        model_id = data.get("model_id")
        
        if not model_id:
            raise ValueError("model_id is required")
        
        wrapper = self.registry.get(model_id)
        if not wrapper:
            raise ValueError(f"Model not found: {model_id}")
        
        # Run inference
        logger.info(f"Running inference on {model_id}")
        result = await wrapper.run_inference(data)
        
        logger.info(f"Inference completed on {model_id}")
        return result
    
    async def handle_encode_image(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode an image to latents for img2img.
        
        Expected data:
            - model_id: Model to use
            - image: Base64-encoded image
        """
        model_id = data.get("model_id")
        
        if not model_id:
            raise ValueError("model_id is required")
        
        wrapper = self.registry.get(model_id)
        if not wrapper:
            raise ValueError(f"Model not found: {model_id}")
        
        # Encode image (if wrapper supports it)
        if hasattr(wrapper, 'encode_image'):
            result = await wrapper.encode_image(data.get("image"))
            return result
        else:
            raise ValueError(f"Model {model_id} does not support image encoding")
    
    async def handle_img2img_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run image-to-image inference.
        
        Expected data:
            - model_id: Model to use
            - prompt: Text prompt
            - negative_prompt: Negative prompt
            - image: Base64-encoded input image
            - strength: How much to transform (0.0-1.0)
            - ... other parameters
        """
        model_id = data.get("model_id")
        
        if not model_id:
            raise ValueError("model_id is required")
        
        wrapper = self.registry.get(model_id)
        if not wrapper:
            raise ValueError(f"Model not found: {model_id}")
        
        # Run img2img inference
        if hasattr(wrapper, 'run_img2img_inference'):
            logger.info(f"Running img2img inference on {model_id}")
            result = await wrapper.run_img2img_inference(data)
            logger.info(f"Img2img inference completed on {model_id}")
            return result
        else:
            raise ValueError(f"Model {model_id} does not support img2img")
    
    async def handle_ping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping for health check."""
        return {"status": "pong", "models_loaded": self.registry.list_models()}
    
    async def handle_shutdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request."""
        logger.info("Shutdown requested")
        # Cleanup all models
        for model_id in self.registry.list_models():
            await self.handle_unload_model({"model_id": model_id})
        
        # Cleanup tensor bridge
        self.tensor_bridge.cleanup_all()
        
        return {"status": "shutting_down"}

