# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from domain.training_request import TrainingRequest
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time


class LoraTrainerRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time(
        "Setting up Lora training",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Setting up Lora training...")
        # TODO: implement Lora training setup logic her
        self.logger.info(f"Device {self.device_id}: Lora training setup completed")

        return True

    @log_execution_time(
        "Lora training",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[TrainingRequest]) -> list:
        self.logger.debug(f"Device {self.device_id}: Starting training...")
        # TODO: implement lora training logic here
        result = None
        self.logger.debug(f"Device {self.device_id}: Training completed")
        return result
