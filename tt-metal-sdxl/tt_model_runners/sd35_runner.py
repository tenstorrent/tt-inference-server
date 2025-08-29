# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from config.settings import get_settings
from tests.scripts.common import get_updated_device_params
from tt_model_runners.base_device_runner import DeviceRunner
from utils.helpers import log_execution_time
from utils.logger import TTLogger
import ttnn
from models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import create_pipeline
from domain.image_generate_request import ImageGenerateRequest

class TTSD35Runner(DeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.settings = get_settings()
        self.pipeline = None
        self.logger = TTLogger()
        self.mesh_shape = ttnn.MeshShape(*self.settings.device_mesh_shape)
        self.mesh_device = None

    def get_device(self):
        return self._mesh_device() #using all device

    def _mesh_device(self):
        if self.mesh_device is  None:
            device_params = {"l1_small_size": 32768, "trace_region_size": 25000000}
            updated_device_params = get_updated_device_params(device_params)
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            self.mesh_device = ttnn.open_mesh_device(mesh_shape=self.mesh_shape, **updated_device_params)

            self.logger.info(f"multidevice with {self.mesh_device.get_num_devices()} devices is created")
        return self.mesh_device

    def get_devices(self):
        device = self._mesh_device()
        return [device]

    def close_device(self, device) -> bool:
        if device is None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
        else:
            ttnn.close_mesh_device(device)
        return True

    @log_execution_time("SD35 warmpup")
    async def load_model(self, device)->bool:
        self.logger.info("Loading model...")
        if (device is None):
            self.mesh_device = self._mesh_device()
        else:
            self.mesh_device = device

        distribute_block = lambda: setattr(self,"pipeline",create_pipeline(mesh_device=self.mesh_device))

        # 6 minutes to distribute the model on device
        weights_distribution_timeout = 360
        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"ttnn.distribute block timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Exception during model loading: {e}")
            raise

        self.logger.info("Model loaded successfully")

        # we use model construct to create the request without validation
        self.run_inference([ImageGenerateRequest.model_construct(
                prompt="Sunrise on a beach",
                negative_prompt="",
                num_inference_steps=1
            )])

        self.logger.info("Model warmup completed")

        return True

    @log_execution_time("SD35 inference")
    def run_inference(self, requests: list[ImageGenerateRequest]):
        prompt = requests[0].prompt
        negative_prompt = requests[0].negative_prompt
        seed = int(requests[0].seed or 0)
        num_inference_steps = requests[0].num_inference_steps or self.settings.num_inference_steps
        image = self.pipeline.run_single_prompt(prompt=prompt,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,seed=seed)

        return image