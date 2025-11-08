# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
SDXL model wrapper for TT-Comfy Bridge.

Wraps the existing TTSDXLGenerateRunnerTrace from tt-media-server.
"""

import sys
import os
import asyncio
import base64
from io import BytesIO
from typing import Dict, Any, Optional
import logging

# Add tt-media-server to path
sys.path.insert(0, '/home/tt-admin/tt-inference-server/tt-media-server')
sys.path.insert(0, '/home/tt-admin/tt-metal')

from models.base_model import BaseModelWrapper

# Import after path setup
from tt_model_runners.sdxl_generate_runner_trace import TTSDXLGenerateRunnerTrace
from domain.image_generate_request import ImageGenerateRequest

logger = logging.getLogger(__name__)


class SDXLModelWrapper(BaseModelWrapper):
    """
    Wrapper for SDXL model using TTSDXLGenerateRunnerTrace.
    
    This wrapper adapts the existing tt-media-server SDXL runner
    for use with the ComfyUI bridge.
    """
    
    def __init__(self, device_id: str = "0"):
        super().__init__(device_id)
        self.runner = None
        self.device = None
    
    async def load_model(self):
        """Load SDXL model onto Tenstorrent device."""
        logger.info(f"Loading SDXL model on device {self.device_id}...")
        
        try:
            # Initialize runner
            self.runner = TTSDXLGenerateRunnerTrace(device_id=self.device_id)
            
            # Load model (this calls the async load_model from the runner)
            # The runner will initialize the device, load weights, and warmup
            await self.runner.load_model(device=None)
            
            self.loaded = True
            logger.info(f"SDXL model loaded successfully on device {self.device_id}")
            
        except Exception as e:
            logger.error(f"Failed to load SDXL model: {e}", exc_info=True)
            self.loaded = False
            raise
    
    async def encode_prompts(
        self, 
        prompt: str, 
        negative_prompt: str,
        prompt_2: Optional[str] = None,
        negative_prompt_2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encode SDXL prompts to embeddings.
        
        Args:
            prompt: Primary positive prompt
            negative_prompt: Primary negative prompt
            prompt_2: Secondary positive prompt (SDXL dual encoders)
            negative_prompt_2: Secondary negative prompt
            
        Returns:
            Dictionary with encoded prompt tensors
        """
        if not self.loaded or not self.runner:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Encoding prompts for SDXL")
        
        # Use the runner's text encoding
        # Note: This returns torch tensors that we'd need to serialize
        # For now, return metadata indicating prompts are encoded
        all_prompt_embeds_torch, torch_add_text_embeds = self.runner.tt_sdxl.encode_prompts(
            prompts=[prompt],
            negative_prompt=negative_prompt,
            prompts_2=[prompt_2] if prompt_2 else None,
            negative_prompt_2=negative_prompt_2
        )
        
        return {
            "status": "encoded",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            # In a full implementation, we'd serialize tensors to shared memory here
        }
    
    async def run_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SDXL text-to-image inference.
        
        Args:
            request_data: Dictionary with inference parameters:
                - prompt: Text prompt
                - negative_prompt: Negative prompt (optional)
                - prompt_2: Secondary prompt (optional)
                - negative_prompt_2: Secondary negative prompt (optional)
                - num_inference_steps: Number of denoising steps (default: 50)
                - guidance_scale: CFG scale (default: 5.0)
                - guidance_rescale: Guidance rescale (default: 0.0)
                - seed: Random seed (optional)
                - timesteps: Custom timesteps (optional)
                - sigmas: Custom sigmas (optional)
                - crop_coords_top_left: Crop coordinates (default: (0, 0))
                
        Returns:
            Dictionary with generated images as base64 strings
        """
        if not self.loaded or not self.runner:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Running SDXL inference on device {self.device_id}")
        
        # Build request object
        request = ImageGenerateRequest(
            prompt=request_data.get("prompt", ""),
            negative_prompt=request_data.get("negative_prompt", ""),
            prompt_2=request_data.get("prompt_2"),
            negative_prompt_2=request_data.get("negative_prompt_2"),
            num_inference_steps=request_data.get("num_inference_steps", 50),
            guidance_scale=request_data.get("guidance_scale", 5.0),
            guidance_rescale=request_data.get("guidance_rescale", 0.0),
            seed=request_data.get("seed", 0),
            timesteps=request_data.get("timesteps"),
            sigmas=request_data.get("sigmas"),
            crop_coords_top_left=request_data.get("crop_coords_top_left", (0, 0)),
            number_of_images=1
        )
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            self.runner.run_inference,
            [request]
        )
        
        # Convert PIL images to base64
        result_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result_images.append(img_base64)
        
        logger.info(f"SDXL inference completed, generated {len(result_images)} image(s)")
        
        return {
            "images": result_images,
            "num_images": len(result_images)
        }
    
    async def cleanup(self):
        """Cleanup SDXL model resources."""
        if self.runner and hasattr(self.runner, 'close_device'):
            try:
                self.runner.close_device(self.runner.get_device())
            except Exception as e:
                logger.warning(f"Error closing device: {e}")
        
        self.runner = None
        await super().cleanup()

