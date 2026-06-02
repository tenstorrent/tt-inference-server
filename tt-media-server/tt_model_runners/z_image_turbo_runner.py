# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import os
import time

import ttnn
from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time

DEFAULT_STEPS = 9

WARMUP_TIMEOUT_SECONDS = 6000


class ZImageTurboRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None

    def set_device(self):
        pass

    def close_device(self):
        if self.pipeline is not None and self.pipeline.mesh_device is not None:
            try:
                self.logger.info(f"Device {self.device_id}: Closing mesh device...")
                ttnn.close_mesh_device(self.pipeline.mesh_device)
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed mesh device"
                )
            except Exception as e:
                self.logger.error(
                    f"Device {self.device_id}: Failed to close device: {e}"
                )
                raise RuntimeError(
                    f"Device {self.device_id}: Device cleanup failed: {str(e)}"
                ) from e

    @log_execution_time(
        "Z-Image-Turbo warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading Z-Image-Turbo ...")

        def load_and_warmup():
            from models.demos.z_image_turbo.tt.z_image_turbo import ZImageTurbo

            self.pipeline = ZImageTurbo()
            self.pipeline.warmup()

        await asyncio.wait_for(
            asyncio.to_thread(load_and_warmup),
            timeout=WARMUP_TIMEOUT_SECONDS,
        )

        self.logger.info(f"Device {self.device_id}: Z-Image-Turbo warmup complete")
        return True

    @log_execution_time(
        "Z-Image-Turbo inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[ImageGenerateRequest]):
        request = requests[0]
        seed = int(request.seed or 0)

        t_start = time.time()
        image = self.pipeline.forward(
            prompt=request.prompt,
            steps=DEFAULT_STEPS,
            seed=seed,
        )
        elapsed = time.time() - t_start

        self.logger.info(
            f"Device {self.device_id}: Generated in {elapsed:.2f}s  seed={seed}"
        )
        return [image]
