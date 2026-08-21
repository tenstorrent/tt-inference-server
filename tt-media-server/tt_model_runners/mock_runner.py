# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import time

from config.constants import ModelServices
from config.settings import settings
from domain.text_to_speech_response import TextToSpeechResponse
from tt_model_runners.base_device_runner import BaseDeviceRunner


class MockRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
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

    def set_device(self, device_id: int = None):
        self.logger.info(f"Getting device {device_id or self.device_id}")
        return {"device_id": device_id or "MockDevice"}

    def run(self, requests):
        self.logger.info(f"Running mock inference for {len(requests)} request(s)")
        time.sleep(2)
        self.logger.info("Starting ttnn inference... on device: " + str(self.device_id))
        # For TTS deployments, return a mock result per request so the full
        # result-queue -> service -> HTTP response path can be exercised without a
        # device. Other services keep the pre-existing deterministic
        # "No responses generated" error (a TTS-shaped object would silently
        # degrade their post-processing instead of failing loudly).
        if settings.model_service != ModelServices.TEXT_TO_SPEECH.value:
            return None
        return [
            TextToSpeechResponse(
                audio=base64.b64encode(b"RIFF-mock-wav-bytes").decode("utf-8"),
                duration=0.1,
                sample_rate=16000,
                format="wav",
            )
            for _ in requests
        ]
