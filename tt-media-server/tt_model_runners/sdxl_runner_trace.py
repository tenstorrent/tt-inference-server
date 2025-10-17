# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from config.settings import get_settings
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from utils.logger import TTLogger
import ttnn
import torch
from diffusers import DiffusionPipeline
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG
)
from domain.sdxl_image_generate_request import SDXLImageGenerateRequest
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig

class TTSDXLRunnerTrace(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.tt_sdxl: TtSDXLPipeline = None
        self.settings = get_settings()
        self.logger = TTLogger()
        # setup is tensor parallel if device mesh shape first param starts with 2
        self.is_tensor_parallel = self.settings.device_mesh_shape[0] > 1
        if (self.is_tensor_parallel):
            self.logger.info(f"Device {self.device_id}: Tensor parallel mode enabled with mesh shape {self.settings.device_mesh_shape}")
        self.batch_size = 0
        self.pipeline = None

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
        if self.is_tensor_parallel:
            device_params["fabric_config"] = SDXL_FABRIC_CONFIG

        mesh_shape = ttnn.MeshShape(self.settings.device_mesh_shape)

        updated_device_params = self.get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"Device {self.device_id}: multidevice with {mesh_device.get_num_devices()} devices is created")
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
        self.logger.info(f"Device {self.device_id}: Loading model...")
        if device is None:
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device
        
        self.batch_size = self.settings.max_batch_size

        # 1. Load components
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.settings.model_weights_path or "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        
        self.logger.info(f"Device {self.device_id}: Model weights downloaded successfully")

        def distribute_block():
            self.tt_sdxl = TtSDXLPipeline(
                ttnn_device=self.ttnn_device,
                torch_pipeline=self.pipeline,
                pipeline_config=TtSDXLPipelineConfig(
                    encoders_on_device=True,
                    is_galaxy=self.settings.is_galaxy,
                    num_inference_steps=self.settings.num_inference_steps,
                    guidance_scale=5.0,
                    use_cfg_parallel=self.is_tensor_parallel,
                ),
            )


        # 6 minutes to distribute the model on device
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
        def warmup_inference_block():
            self.run_inference([SDXLImageGenerateRequest.model_construct(
                    prompt="Sunrise on a beach",
                    prompt_2="Mountains in the background",
                    negative_prompt="low resolution",
                    negative_prompt_2="blurry",
                    num_inference_steps=1,
                    timesteps=None,
                    sigmas=None,
                    guidance_scale=5.0,
                    guidance_rescale=0.7,
                    number_of_images=1,
                    crop_coords_top_left=(0, 0),
                )])

        warmup_inference_timeout = 1000

        try:
            await asyncio.wait_for(asyncio.to_thread(warmup_inference_block), timeout=warmup_inference_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id}: warmup inference timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Exception during warmup inference: {e}")
            raise

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time("SDXL inference")
    def run_inference(self, requests: list[SDXLImageGenerateRequest]):
        prompts = [request.prompt for request in requests]
        negative_prompt = requests[0].negative_prompt if requests[0].negative_prompt else None
        if isinstance(prompts, str):
            prompts = [prompts]

        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        prompts = prompts + [""] * needed_padding

        prompts_2 = [request.prompt_2 if request.prompt_2 is not None else "" for request in requests]
        negative_prompt_2 = requests[0].negative_prompt_2 if requests[0].negative_prompt_2 else None
        if isinstance(prompts_2, str):
            prompts_2 = [prompts_2]

        needed_padding = (self.batch_size - len(prompts_2) % self.batch_size) % self.batch_size
        prompts_2 = prompts_2 + [""] * needed_padding

        if requests[0].num_inference_steps is not None:
            self.tt_sdxl.set_num_inference_steps(requests[0].num_inference_steps)
        
        if requests[0].guidance_scale is not None:
            self.tt_sdxl.set_guidance_scale(requests[0].guidance_scale)

        if requests[0].guidance_rescale is not None:
            self.tt_sdxl.set_guidance_rescale(requests[0].guidance_rescale)

        if requests[0].crop_coords_top_left is not None:
            self.tt_sdxl.set_crop_coords_top_left(requests[0].crop_coords_top_left)

        self.logger.debug(f"Device {self.device_id}: Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = self.tt_sdxl.encode_prompts(prompts, negative_prompt, prompts_2, negative_prompt_2)

        self.logger.info(f"Device {self.device_id}: Generating input tensors...")

        if requests[0].timesteps is not None and requests[0].sigmas is not None:
            raise ValueError("Cannot pass both timesteps and sigmas. Choose one.")

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_sdxl.generate_input_tensors(
            all_prompt_embeds_torch = all_prompt_embeds_torch,
            torch_add_text_embeds = torch_add_text_embeds,
            start_latent_seed = requests[0].seed,
            timesteps = requests[0].timesteps,
            sigmas = requests[0].sigmas
        )
        
        self.logger.debug(f"Device {self.device_id}: Preparing input tensors...") 
        
        self.tt_sdxl.prepare_input_tensors(
            [
                tt_latents,
                tt_prompt_embeds[0],
                tt_add_text_embeds[0],
            ]
        )

        self.logger.debug(f"Device {self.device_id}: Compiling image processing...")

        self.tt_sdxl.compile_image_processing()

        profiler.clear()

        images = []
        self.logger.info(f"Device {self.device_id}: Starting ttnn inference...")
        for iter in range(len(prompts) // self.batch_size):
            self.logger.info(
                f"Device {self.device_id}: Running inference for prompts {iter * self.batch_size + 1}-{iter * self.batch_size + self.batch_size}/{len(prompts)}"
            )

            self.tt_sdxl.prepare_input_tensors(
                [
                    tt_latents,
                    tt_prompt_embeds[iter],
                    tt_add_text_embeds[iter],
                ]
            )
            imgs = self.tt_sdxl.generate_images()
            
            self.logger.info(
                f"Device {self.device_id}: Prepare input tensors for {self.batch_size} prompts completed in {profiler.times['prepare_input_tensors'][-1]:.2f} seconds"
            )
            self.logger.info(f"Device {self.device_id}: Image gen for {self.batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
            self.logger.info(
                f"Device {self.device_id}: Denoising loop for {self.batch_size} promts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
            )
            self.logger.info(
                f"Device {self.device_id}: On device VAE decoding completed in {profiler.times['vae_decode'][-1]:.2f} seconds"
            )
            self.logger.info(f"Device {self.device_id}: Output tensor read completed in {profiler.times['read_output_tensor'][-1]:.2f} seconds")

            for idx, img in enumerate(imgs):
                if iter == len(prompts) // self.batch_size - 1 and idx >= self.batch_size - needed_padding:
                    break
                img = img.unsqueeze(0)
                img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
                images.append(img)

        return images