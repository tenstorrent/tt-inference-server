# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from typing import List
from config.settings import get_settings
from tests.scripts.common import get_updated_device_params
from tt_model_runners.base_device_runner import DeviceRunner
from utils.helpers import log_execution_time
from utils.logger import TTLogger
import ttnn
import torch
from diffusers import DiffusionPipeline
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE
)
from domain.image_generate_request import ImageGenerateRequest
from models.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig

class TTSDXLRunnerTrace(DeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.tt_sdxl: TtSDXLPipeline = None
        self.settings = get_settings()
        self.batch_size = 0
        self.pipeline = None
        self.logger = TTLogger()

    def _set_fabric(self, fabric_config):
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
        device_params = {'l1_small_size': SDXL_L1_SMALL_SIZE, 'trace_region_size': self.settings.trace_region_size or SDXL_TRACE_REGION_SIZE}
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

        self.logger.info(f"Device {self.device_id} multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def close_device(self, device) -> bool:
        if device is None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
        else:
            ttnn.close_mesh_device(device)
        return True

    @log_execution_time("SDXL warmpup")
    async def load_model(self, device)->bool:
        self.logger.info("Device {self.device_id} Loading model...")
        if (device is None):
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device
        
        self.batch_size = self.ttnn_device.get_num_devices()

        # 1. Load components
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.settings.model_weights_path or "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        
        self.logger.info(f"Device {self.device_id} Model weights downloaded successfully")

        def distribute_block():
            self.tt_sdxl = TtSDXLPipeline(
                ttnn_device=self.ttnn_device,
                torch_pipeline=self.pipeline,
                pipeline_config=TtSDXLPipelineConfig(
                    encoders_on_device=True,
                    num_inference_steps=self.settings.num_inference_steps,
                    guidance_scale=5.0,
                ),
            )


        # 6 minutes to distribute the model on device
        weights_distribution_timeout = 720

        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id} ttnn.distribute block timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id} Exception during model loading: {e}")
            raise

        self.logger.info(f"Device {self.device_id} Model loaded successfully")

        # we use model construct to create the request without validation
        def warmup_inference_block():
            self.run_inference([ImageGenerateRequest.model_construct(
                    prompt="Sunrise on a beach",
                    negative_prompt="low resolution",
                    num_inference_steps=1,
                    guidance_scale=5.0,
                    number_of_images=1
                )])

        warmup_inference_timeout = 1000

        try:
            await asyncio.wait_for(asyncio.to_thread(warmup_inference_block), timeout=warmup_inference_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id} warmup inference timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id} Exception during warmup inference: {e}")
            raise

        self.logger.info(f"Device {self.device_id} Model warmup completed")

        return True

    @log_execution_time("SDXL inference")
    def run_inference(self, requests: list[ImageGenerateRequest]):
        prompts = [request.prompt for request in requests]
        # TODO include negative prompts handling
        negative_prompt = requests[0].negative_prompt if requests[0].negative_prompt else None
        if isinstance(prompts, str):
            prompts = [prompts]

        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        prompts = prompts + [""] * needed_padding

        if (requests[0].seed is not None):
            torch.manual_seed(requests[0].seed)

        if (requests[0].num_inference_steps is not None):
            self.tt_sdxl.set_num_inference_steps(requests[0].num_inference_steps)
        
        if (requests[0].guidance_scale is not None):
            self.tt_sdxl.set_guidance_scale(requests[0].guidance_scale)

        self.logger.debug(f"Device {self.device_id} Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        (
            prompt_embeds_torch,
            negative_prompt_embeds_torch,
            pooled_prompt_embeds_torch,
            negative_pooled_prompt_embeds_torch,
        ) = self.tt_sdxl.encode_prompts(prompts)

        self.logger.info(f"Device {self.device_id} Generating input tensors...")

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_sdxl.generate_input_tensors(
            prompt_embeds_torch,
            negative_prompt_embeds_torch,
            pooled_prompt_embeds_torch,
            negative_pooled_prompt_embeds_torch,
        )
        
        self.logger.debug(f"Device {self.device_id} Preparing input tensors...") 
        
        self.tt_sdxl.prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[0],
                tt_add_text_embeds[0][0],
                tt_add_text_embeds[0][1],
            ]
        )

        self.logger.debug(f"Device {self.device_id} Compiling image processing...")

        self.tt_sdxl.compile_image_processing()

        profiler.clear()

        images = []
        self.logger.info(f"Device {self.device_id} Starting ttnn inference...")
        for iter in range(len(prompts) // self.batch_size):
            self.logger.info(
                f"Device {self.device_id} Running inference for prompts {iter * self.batch_size + 1}-{iter * self.batch_size + self.batch_size}/{len(prompts)}"
            )

            self.tt_sdxl.prepare_input_tensors(
                [
                    tt_latents,
                    *tt_prompt_embeds[iter],
                    tt_add_text_embeds[iter][0],
                    tt_add_text_embeds[iter][1],
                ]
            )
            imgs = self.tt_sdxl.generate_images()
            
            self.logger.info(
                f"Device {self.device_id} Prepare input tensors for {self.batch_size} prompts completed in {profiler.times['prepare_input_tensors'][-1]:.2f} seconds"
            )
            self.logger.info(f"Device {self.device_id} Image gen for {self.batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
            self.logger.info(
                f"Device {self.device_id} Denoising loop for {self.batch_size} promts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
            )
            self.logger.info(
                f"Device {self.device_id} On device VAE decoding completed in {profiler.times['vae_decode'][-1]:.2f} seconds"
            )
            self.logger.info(f"Device {self.device_id} Output tensor read completed in {profiler.times['read_output_tensor'][-1]:.2f} seconds")

            for idx, img in enumerate(imgs):
                if iter == len(prompts) // self.batch_size - 1 and idx >= self.batch_size - needed_padding:
                    break
                img = img.unsqueeze(0)
                img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
                images.append(img)

        return images