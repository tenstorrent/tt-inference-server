# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time  # Add this import

from tt_model_runners.base_device_runner import BaseDeviceRunner


class MockRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.logger.info(f"MockRunner initialized for device {self.device_id}")

    def close_device(self) -> bool:
        self.logger.info("Closing device...")
        time.sleep(5)  # Use time.sleep() instead of await asyncio.sleep()
        return True

    async def warmup(self) -> bool:
        self.logger.info("Loading model...")
        time.sleep(10)  # Use time.sleep() instead of await asyncio.sleep()
        self.logger.info(f"Model loaded successfully on device {self.device_id}")
        time.sleep(10)  # Use time.sleep() instead of await asyncio.sleep()
        self.logger.info(f"Model warmup completed on device {self.device_id}")
        return True

    def set_device(self, device_id: int):
        self.logger.info(f"Getting device {device_id or self.device_id}")
        return {"device_id": device_id or "MockDevice"}

    def run(self, prompt: str):
        self.logger.info(f"Running inference for prompt: {prompt}")
        time.sleep(20)
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))
        return None
