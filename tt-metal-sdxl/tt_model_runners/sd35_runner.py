# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from pathlib import Path
from typing import List

from sklearn import utils
from config.settings import settings
from tests.scripts.common import get_updated_device_params
from tt_model_runners.base_device_runner import DeviceRunner
from tt_model_runners.sd35_utils.sd_35_pipeline import TtStableDiffusion3Pipeline
from utils.logger import TTLogger
import ttnn
import torch
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.utility_functions import profiler

class   TTSD35Runner(DeviceRunner):
    device = None
    batch_size = 0
    tt_unet = None
    tt_scheduler = None
    ttnn_prompt_embeds = None
    ttnn_time_ids = None
    ttnn_text_embeds = None
    ttnn_timesteps = []
    extra_step_kwargs = None
    scaling_factor = None
    tt_vae = None
    pipeline = None
    latents = None

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()

    def _set_fabric(self,fabric_config):
        # If fabric_config is not None, set it to fabric_config
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        # for now use all availalbe devices
        return self._mesh_device()

    def _mesh_device(self):
        device_params = {'l1_small_size': 57344}
        device_ids = ttnn.get_device_ids()

        param = len(device_ids)  # Default to using all available devices

        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if num_devices_requested > len(device_ids):
                print("Requested more devices than available. Test not applicable for machine")
            mesh_shape = ttnn.MeshShape(*grid_dims)
            assert num_devices_requested <= len(device_ids), "Requested more devices than available."
        else:
            num_devices_requested = min(param, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)


        updated_device_params = get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def get_devices(self) -> List[ttnn.MeshDevice]:
        device = self._mesh_device()
        device_shape = settings.device_mesh_shape
        return device.create_submeshes(ttnn.MeshShape(*device_shape))

    def close_device(self) -> bool:
        for submesh in self.mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(self.mesh_device)

    async def load_model(self, device)->bool:
        self.logger.info("Loading model...")
        if (device is None):
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device

        # TODO chang this
        model_version = "large"
        guidance_scale = 5.0

        if guidance_scale > 1:
            guidance_cond = 2
        else:
            guidance_cond = 1

        # 1. Load components
        # TODO check how to point to a model file
        self.pipeline = TtStableDiffusion3Pipeline(
            checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_version}",
            device=self.ttnn_device,
            enable_t5_text_encoder=self.ttnn_device.get_num_devices() >= 4,
            vae_cpu_fallback=True,
            guidance_cond=guidance_cond,
        )

        self.pipeline.prepare(
            width=1024,
            height=1024,
            guidance_scale=guidance_scale,
            prompt_sequence_length=333,
            spatial_sequence_length=4096,
        )

        self.logger.info("Model weights downloaded successfully")

        self.logger.info("Model loaded successfully")

        self.runInference("Sunrise on a beach", 20)

        self.logger.info("Model warmup completed")

        return True

    def runInference(self, prompt: str, num_inference_steps: int = 50):
        prompts = [prompt]

        torch.manual_seed(0)

        if isinstance(prompts, str):
            prompts = [prompts]
        
        negative_prompt = "bad quality, low resolution, blurry, dark, noisy, bad lighting, bad composition"

        images = self.pipeline(
            prompt_1=[prompts[0]],
            prompt_2=[prompt[0]],
            prompt_3=[prompt[0]],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
        )

        return images
