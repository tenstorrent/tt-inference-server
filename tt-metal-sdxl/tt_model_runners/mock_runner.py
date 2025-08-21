# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time  # Add this import
from config.settings import settings
from tests.scripts.common import get_updated_device_params
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger

class MockRunner(DeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.device_id = device_id
        self.logger = TTLogger()
        self.logger.info(f"MockRunner initialized for device {self.device_id}")

    def close_device(self) -> bool:
        self.logger.info("Closing device...")
        time.sleep(5)  # Use time.sleep() instead of await asyncio.sleep()
        return True

    async def load_model(self) -> bool:
        self.logger.info("Loading model...")
        time.sleep(10)  # Use time.sleep() instead of await asyncio.sleep()
        self.logger.info(f"Model loaded successfully on device {self.device_id}")
        time.sleep(10)  # Use time.sleep() instead of await asyncio.sleep()
        self.logger.info(f"Model warmup completed on device {self.device_id}")
        return True

    def get_device(self, device_id: int): 
        self.logger.info(f"Getting device {device_id or self.device_id}")
        return {"device_id": device_id or "MockDevice"}

    def get_devices(self):
        self.logger.info("Getting all devices")
        return (self.get_device() ,[self.get_device() for _ in range(settings.mock_devices_count)])

    def run_inference(self, prompt: str, num_inference_steps: int = 50):
        self.logger.info(f"Running inference for prompt: {prompt} with {num_inference_steps} steps")
        time.sleep(2 * num_inference_steps + 10)  # Use time.sleep() instead of await asyncio.sleep()
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))
        return None