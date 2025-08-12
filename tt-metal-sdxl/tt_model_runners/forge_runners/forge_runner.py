# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time

from config.settings import settings
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger

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
        inputs = self.loader.load_inputs()

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

    def runInference(self, prompt: str, num_inference_steps: int = 50):
        self.logger.info(f"Running inference for prompt: {prompt} with {num_inference_steps} steps")
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))
        
        # Run inference on Tenstorrent device
        inputs = self.loader.load_inputs()
        output = self.compiled_model(inputs)
        self.loader.print_cls_results(output)
    
        return f"Mock inference result for prompt: {prompt} on device: {self.device_id}"
