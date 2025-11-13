# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from config.settings import get_settings
from config.constants import SupportedModels, ModelRunners
from abc import abstractmethod
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import StableDiffusion3Pipeline
from models.experimental.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from domain.image_generate_request import ImageGenerateRequest

dit_runner_log_map={
    ModelRunners.TT_SD3_5.value: "SD35",
    ModelRunners.TT_FLUX_1_DEV.value: "FLUX.1-dev",
    ModelRunners.TT_FLUX_1_SCHNELL.value: "FLUX.1-schnell"
}

class TTDiTRunner(BaseDeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None

    @abstractmethod
    def create_pipeline(self):
        """Create a pipeline for the model"""

    @abstractmethod
    def get_pipeline_device_params(self):
        """Get the device parameters for the pipeline"""

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} warmup", TelemetryEvent.DEVICE_WARMUP, os.environ.get("TT_VISIBLE_DEVICES"))
    async def load_model(self)->bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        distribute_block = lambda: setattr(self,"pipeline", self.create_pipeline())

        # 12 minutes to distribute the model on device
        weights_distribution_timeout = 720
        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id}: ttnn.distribute block timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Exception during model loading: {e}")
            raise

        self.logger.info(f"Device {self.device_id}: Model loaded successfully")

        # we use model construct to create the request without validation
        self.run_inference([ImageGenerateRequest.model_construct(
                prompt="Sunrise on a beach",
                negative_prompt="",
                num_inference_steps=1
            )])

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference", TelemetryEvent.MODEL_INFERENCE, os.environ.get("TT_VISIBLE_DEVICES"))
    def run_inference(self, requests: list[ImageGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        prompt = requests[0].prompt
        negative_prompt = requests[0].negative_prompt
        seed = int(requests[0].seed or 0)
        num_inference_steps = requests[0].num_inference_steps or self.settings.num_inference_steps
        image = self.pipeline.run_single_prompt(prompt=prompt,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,seed=seed)
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return image

class TTSD35Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        return StableDiffusion3Pipeline.create_pipeline(
            mesh_device=self.ttnn_device,
            model_checkpoint_path=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value
        )

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 25000000}

#TODO: Merge dev and schnell
class TTFlux1DevRunner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        return Flux1Pipeline.create_pipeline(
            checkpoint_name=SupportedModels.FLUX_1_DEV.value,
            mesh_device=self.ttnn_device,
        )

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 34000000}

class TTFlux1SchnellRunner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        return Flux1Pipeline.create_pipeline(
            checkpoint_name=SupportedModels.FLUX_1_SCHNELL.value,
            mesh_device=self.ttnn_device,
        )

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 34000000}
