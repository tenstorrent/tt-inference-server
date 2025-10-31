# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1" # Set this before importing torch_xla

import base64
from io import BytesIO
import time
from typing import List

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from domain.image_search_request import ImageSearchRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from PIL import Image

from .loaders.tools.utils import output_to_tensor

xla_backend = "tt"
runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
use_optimizer = os.getenv("USE_OPTIMIZER", "true").lower() == "true"


class ForgeRunner(BaseDeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.device_id = device_id
        self.logger.info(f"ForgeRunner initialized for device {self.device_id}")


    def close_device(self) -> bool:
        self.logger.info("Closing device...")
        time.sleep(5)  # Use time.sleep() instead of await asyncio.sleep()
        return True


    async def load_model(self, device=None) -> bool:
        model_config = self.loader._variant_config
        self.logger.info(f"Loading { model_config.pretrained_model_name } model on device {self.device_id} using tt-xla ...")

        if runs_on_cpu:
            # Use cpu
            self.device = torch.device('cpu')
            self.model = self.loader.load_model()
            self.compiled_model = self.model.to(self.device)
        else:
            # Use TT device
            xr.set_device_type("TT")
            self.device = xm.xla_device()
            self.model = self.loader.load_model()
        
            self.logger.info(f"## Compiling model ##")
            if use_optimizer:                
                torch_xla.set_custom_compile_options({
                    "enable_optimizer": True,
                    "enable_fusing_conv2d_with_multiply_pattern": True,
                })
                self.model.compile(backend="tt")
                self.compiled_model = self.model.to(self.device)
            else:
                self.compiled_model = torch.compile(
                    self.model,
                    backend=xla_backend
                ).to(self.device)
        
        self.logger.info(f"## Load inputs ##")
        inputs = self.loader.image_to_input(Image.new(
            mode="RGB", 
            size=(224, 224), 
            color=(255,255,255))
        ).to(self.device)
        
        self.logger.info(f"## Run inference ##")
        
        with torch.no_grad():
            output = self.compiled_model(inputs)
            output = output_to_tensor(output)
            predictions = self.loader.output_to_prediction(output)
            
        return True


    def get_device(self, device_id: int = None): 
        self.logger.info(f"Getting device {device_id or self.device_id}")
        return {"device_id": device_id or "MockDevice"}


    def run_inference(self, image_search_requests: List[ImageSearchRequest], num_inference_steps: int = 50):
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))
        
        if not image_search_requests:
            raise ValueError("Empty requests list provided")
        
        if len(image_search_requests) > 1:
            self.logger.warning(f"Batch processing not fully implemented. Processing only first of {len(image_search_requests)} requests")
        
        # Get the first request
        request = image_search_requests[0]
        
        # Get PIL image from the request (which contains base64 image data in prompt field)
        pil_image = self.base64_to_pil_image(request.prompt, target_mode="RGB")
        
        # Run inference on Tenstorrent device
        inputs = self.loader.image_to_input(pil_image).to(self.device)
        
        # # Debug with random inputs
        # inputs = torch.rand(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            output = self.compiled_model(inputs)
            output = output_to_tensor(output)
            return self.loader.output_to_prediction(output)


    def base64_to_pil_image(self, base64_string, target_mode="RGB"):
        """
        Convert base64 encoded image to PIL Image with specified format
        
        Args:
            base64_string: Base64 encoded image string
            target_size: Tuple of (width, height) for resizing
            target_mode: PIL Image mode (e.g., "RGB", "RGBA", "L")
        
        Returns:
            PIL Image object
        """
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create PIL Image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to target mode if different
        if image.mode != target_mode:
            image = image.convert(target_mode)
        
        return image
