# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
import time

from config.settings import settings
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger
from PIL import Image

import forge
from .loader import ModelLoader

class ForgeRunner(DeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.device_id = device_id
        self.logger = TTLogger()
        self.logger.info(f"ForgeRunner initialized for device {self.device_id}")
        self.loader = ModelLoader()
        self.compiled_model = None

    def close_device(self) -> bool:
        self.logger.info("Closing device...")
        time.sleep(5)  # Use time.sleep() instead of await asyncio.sleep()
        return True

    async def load_model(self, device=None) -> bool:
        model = self.loader.load_model()
        inputs = self.loader.load_inputs(Image.new(mode="RGB", size=(324, 324), color=(255,255,255)))

        self.logger.info(f"Loading model on device {self.device_id} with inputs shape {inputs.shape}")

        # Compile the model using Forge
        self.compiled_model = forge.compile(model, sample_inputs=[inputs])

        # Run inference on Tenstorrent device
        output = self.compiled_model(inputs)
        self.loader.print_cls_results(output)
        return True

    def get_device(self, device_id: int = None): 
        self.logger.info(f"Getting device {device_id or self.device_id}")
        return {"device_id": device_id or "MockDevice"}

    def get_devices(self):
        self.logger.info("Getting all devices")
        return (self.get_device() ,[self.get_device() for _ in range(settings.mock_devices_count)])

    def run_inference(self, image, num_inference_steps: int = 50):
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))
        
        pil_image = self.base64_to_pil_image(image[0], target_size=(324, 324), target_mode="RGB")
        
        # Run inference on Tenstorrent device
        inputs = self.loader.load_inputs(pil_image)
        self.logger.info(f"Running inference with inputs shape {inputs.shape} on device {self.device_id}")
        output = self.compiled_model(inputs)

        return [self.loader.print_cls_results(output)]

    def base64_to_pil_image(self, base64_string, target_size=(324, 324), target_mode="RGB"):
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
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
