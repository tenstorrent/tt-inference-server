# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Stable Diffusion 1.4 model wrapper for TT-Comfy Bridge.

Wraps the proven Wormhole SD 1.4 implementation from tt-metal.
"""

import sys
import asyncio
import base64
from io import BytesIO
from typing import Dict, Any
import logging

# Add paths
sys.path.insert(0, '/home/tt-admin/tt-metal')

from models.base_model import BaseModelWrapper

logger = logging.getLogger(__name__)


class SD14ModelWrapper(BaseModelWrapper):
    """
    Wrapper for Stable Diffusion 1.4 model using tt-metal Wormhole implementation.
    
    This wrapper adapts the proven SD 1.4 implementation from
    models/demos/wormhole/stable_diffusion/ for ComfyUI bridge.
    """
    
    def __init__(self, device_id: str = "0"):
        super().__init__(device_id)
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.device = None
    
    async def load_model(self):
        """Load SD1.4 model onto Tenstorrent device."""
        logger.info(f"Loading SD1.4 model on device {self.device_id}...")
        
        try:
            # Import after sys.path is set
            import ttnn
            from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
                UNet2DConditionModel as UNet2D
            )
            from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae import Vae
            from models.demos.wormhole.stable_diffusion.sd_helper_funcs import (
                get_reference_clip_text_encoder,
                get_reference_clip_tokenizer,
            )
            from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
            from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE, SD_TRACE_REGION_SIZE
            
            # Initialize device
            device_params = {
                'l1_small_size': SD_L1_SMALL_SIZE,
                'trace_region_size': SD_TRACE_REGION_SIZE
            }
            self.device = ttnn.open_device(device_id=int(self.device_id), **device_params)
            
            # Load text encoder and tokenizer (CPU)
            self.text_encoder = get_reference_clip_text_encoder()
            self.tokenizer = get_reference_clip_tokenizer()
            
            # Load UNet on device
            logger.info("Loading UNet...")
            self.unet = UNet2D(device=self.device)
            
            # Load VAE on device
            logger.info("Loading VAE...")
            self.vae = Vae(device=self.device)
            
            # Initialize scheduler
            self.scheduler = TtPNDMScheduler()
            
            self.loaded = True
            logger.info(f"SD1.4 model loaded successfully on device {self.device_id}")
            
        except Exception as e:
            logger.error(f"Failed to load SD1.4 model: {e}", exc_info=True)
            self.loaded = False
            # Cleanup on failure
            if self.device:
                import ttnn
                ttnn.close_device(self.device)
                self.device = None
            raise
    
    async def encode_prompts(
        self, 
        prompt: str, 
        negative_prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode prompts for SD1.4.
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            
        Returns:
            Dictionary with encoding status
        """
        if not self.loaded or not self.text_encoder:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.debug(f"Encoding prompts for SD1.4")
        
        # Tokenize prompts
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        # Handle negative prompt
        uncond_input = self.tokenizer(
            negative_prompt if negative_prompt else "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        
        return {
            "status": "encoded",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            # In full implementation, would cache these embeddings
        }
    
    async def run_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SD1.4 text-to-image inference.
        
        Args:
            request_data: Dictionary with inference parameters:
                - prompt: Text prompt
                - negative_prompt: Negative prompt (optional)
                - num_inference_steps: Number of denoising steps (default: 50)
                - guidance_scale: CFG scale (default: 7.5)
                - seed: Random seed (default: 0)
                - height: Image height (default: 512)
                - width: Image width (default: 512)
                
        Returns:
            Dictionary with generated images as base64 strings
        """
        if not self.loaded or not self.unet:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Running SD1.4 inference on device {self.device_id}")
        
        import torch
        import numpy as np
        from PIL import Image
        
        # Get parameters
        prompt = request_data.get("prompt", "")
        negative_prompt = request_data.get("negative_prompt", "")
        num_inference_steps = request_data.get("num_inference_steps", 50)
        guidance_scale = request_data.get("guidance_scale", 7.5)
        seed = request_data.get("seed", 0)
        height = request_data.get("height", 512)
        width = request_data.get("width", 512)
        
        # Set seed
        generator = torch.Generator().manual_seed(seed)
        
        # Encode prompts
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids)[0]
        
        uncond_input = self.tokenizer(
            negative_prompt if negative_prompt else "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
        
        # Concatenate for CFG
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Initialize latents
        latents_shape = (1, 4, height // 8, width // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = self.unet(latent_model_input, t, text_embeddings)
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents
        image = self.vae.decode(latents)
        
        # Post-process image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info(f"SD1.4 inference completed")
        
        return {
            "images": [img_base64],
            "num_images": 1
        }
    
    async def cleanup(self):
        """Cleanup SD1.4 model resources."""
        if self.device:
            try:
                import ttnn
                ttnn.close_device(self.device)
            except Exception as e:
                logger.warning(f"Error closing device: {e}")
        
        self.device = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        
        await super().cleanup()

