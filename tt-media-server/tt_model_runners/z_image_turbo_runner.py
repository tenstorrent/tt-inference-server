# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import os
import time

import ttnn
from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from telemetry.image_metrics import record_image_run, resolution_of_images
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

        t_start = time.perf_counter()
        image = self.pipeline.forward(
            prompt=request.prompt,
            steps=DEFAULT_STEPS,
            seed=seed,
        )
        elapsed = time.perf_counter() - t_start

        self.logger.info(
            f"Device {self.device_id}: Generated in {elapsed:.2f}s  seed={seed}"
        )
        self._record_image_stage_metrics([image], elapsed)
        return [image]

    def _record_image_stage_metrics(self, images, elapsed):
        """Export what this runner can see of the image stages.

        forward() is a single opaque call with no stage boundaries, so only the
        engine total and the fixed step count are knowable. The denoise / VAE /
        conditioning splits stay absent rather than guessed.
        """
        try:
            resolution, _, _ = resolution_of_images(images)
            record_image_run(
                model_type=self.settings.model_runner,
                device_id=self.device_id,
                resolution=resolution,
                sampler="unknown",
                batch=1,
                engine_seconds=elapsed,
                step_count=DEFAULT_STEPS,
            )
        except Exception as exc:
            self.logger.warning(f"Failed to record image stage metrics: {exc}")
