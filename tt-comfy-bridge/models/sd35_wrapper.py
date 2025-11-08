# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Stable Diffusion 3.5 model wrapper for TT-Comfy Bridge.

Wraps the existing TTSD35Runner from tt-media-server.
"""

import sys
import asyncio
import base64
from io import BytesIO
from typing import Dict, Any
import logging

# Add paths
sys.path.insert(0, '/home/tt-admin/tt-inference-server/tt-media-server')
sys.path.insert(0, '/home/tt-admin/tt-metal')

from models.base_model import BaseModelWrapper

# Import after path setup
from tt_model_runners.dit_runners import TTSD35Runner
from domain.image_generate_request import ImageGenerateRequest

logger = logging.getLogger(__name__)


class SD35ModelWrapper(BaseModelWrapper):
    """
    Wrapper for Stable Diffusion 3.5 model using TTSD35Runner.
    
    This wrapper adapts the existing tt-media-server SD3.5 runner
    for use with the ComfyUI bridge.
    """
    
    def __init__(self, device_id: str = "0"):
        super().__init__(device_id)
        self.runner = None
    
    async def load_model(self):
        """Load SD3.5 model onto Tenstorrent device."""
        logger.info(f"Loading SD3.5 model on device {self.device_id}...")
        
        try:
            # Initialize runner
            self.runner = TTSD35Runner(device_id=self.device_id)
            
            # Load model (this initializes the DiT pipeline)
            # The runner's load_model handles device initialization and warmup
            device = self.runner.get_device()
            await self.runner.load_model(device=device)
            
            self.loaded = True
            logger.info(f"SD3.5 model loaded successfully on device {self.device_id}")
            
        except Exception as e:
            logger.error(f"Failed to load SD3.5 model: {e}", exc_info=True)
            self.loaded = False
            raise
    
    async def encode_prompts(
        self, 
        prompt: str, 
        negative_prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode prompts for SD3.5.
        
        Note: SD3.5 uses a DiT architecture with different text encoding
        than SDXL. The encoding is handled internally by the pipeline.
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            
        Returns:
            Dictionary with encoding status
        """
        if not self.loaded or not self.runner:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Encoding prompts for SD3.5")
        
        return {
            "status": "encoded",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
    
    async def run_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SD3.5 text-to-image inference.
        
        Args:
            request_data: Dictionary with inference parameters:
                - prompt: Text prompt
                - negative_prompt: Negative prompt (optional)
                - num_inference_steps: Number of denoising steps (default: 28)
                - seed: Random seed (default: 0)
                
        Returns:
            Dictionary with generated images as base64 strings
        """
        if not self.loaded or not self.runner:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Running SD3.5 inference on device {self.device_id}")
        
        # Build request object
        request = ImageGenerateRequest(
            prompt=request_data.get("prompt", ""),
            negative_prompt=request_data.get("negative_prompt", ""),
            num_inference_steps=request_data.get("num_inference_steps", 28),
            seed=request_data.get("seed", 0),
            number_of_images=1
        )
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            self.runner.run_inference,
            [request]
        )
        
        # SD3.5 runner returns a single PIL image
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info(f"SD3.5 inference completed")
        
        return {
            "images": [img_base64],
            "num_images": 1
        }
    
    async def cleanup(self):
        """Cleanup SD3.5 model resources."""
        if self.runner and hasattr(self.runner, 'close_device'):
            try:
                device = self.runner.get_device()
                self.runner.close_device(device)
            except Exception as e:
                logger.warning(f"Error closing device: {e}")
        
        self.runner = None
        await super().cleanup()

