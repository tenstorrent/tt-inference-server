# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from config.constants import SupportedModels
from config.settings import get_settings
from domain.image_to_image_request import ImageToImageRequest
import numpy as np
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from utils.image_manager import ImageManager
from utils.logger import TTLogger
import ttnn
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG
)
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import TtSDXLImg2ImgPipeline, TtSDXLImg2ImgPipelineConfig

class TTSDXLImg2ImgRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.image_manager = ImageManager("img")
        self.tt_sdxl: TtSDXLImg2ImgPipeline = None
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
        # for now use all available devices
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

    @log_execution_time("SDXL warmup")
    async def load_model(self, device)->bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")
        if device is None:
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device
        
        self.batch_size = self.settings.max_batch_size

        # 1. Load components
        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.settings.model_weights_path or SupportedModels.STABLE_DIFFUSION_XL_BASE.value,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        
        self.logger.info(f"Device {self.device_id}: Model weights downloaded successfully")

        def distribute_block():
            self.tt_sdxl = TtSDXLImg2ImgPipeline(
                ttnn_device=self.ttnn_device,
                torch_pipeline=self.pipeline,
                pipeline_config=TtSDXLImg2ImgPipelineConfig(
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
            self.run_inference([ImageToImageRequest.model_construct(
                    prompt="Sunrise on a beach",
                    image="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",
                    negative_prompt="low resolution",
                    num_inference_steps=2,
                    guidance_scale=5.0,
                    number_of_images=1,
                    strength=0.99,
                    aesthetic_score=6.0,
                    negative_aesthetic_score=2.5,
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

    @log_execution_time("SDXL Image 2 Image inference")
    def run_inference(self, requests: list[ImageToImageRequest]):
        prompts = [request.prompt for request in requests]
        negative_prompt = requests[0].negative_prompt if requests[0].negative_prompt else None
        if isinstance(prompts, str):
            prompts = [prompts]

        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        prompts = prompts + [""] * needed_padding

        if requests[0].num_inference_steps is not None:
            self.tt_sdxl.set_num_inference_steps(requests[0].num_inference_steps)
        
        if requests[0].guidance_scale is not None:
            self.tt_sdxl.set_guidance_scale(requests[0].guidance_scale)

        self.logger.debug(f"Device {self.device_id}: Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        image = requests[0].image
        image = self.image_manager.base64_to_pil_image(image, target_size=(1024, 1024), target_mode="RGB")
        image_tensor = [
            self.tt_sdxl.torch_pipeline.image_processor.preprocess(
                image, height=1024, width=1024, crops_coords=None, resize_mode="default"
            ).to(dtype=torch.float32)
        ]
        images = torch.cat(image_tensor, dim=0)

        if requests[0].strength is not None:
            self.tt_sdxl.set_strength(requests[0].strength)

        if requests[0].aesthetic_score is not None:
            self.tt_sdxl.set_aesthetic_score(requests[0].aesthetic_score)

        if requests[0].negative_aesthetic_score is not None:
            self.tt_sdxl.set_negative_aesthetic_score(requests[0].negative_aesthetic_score)


        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = self.tt_sdxl.encode_prompts(prompts, negative_prompt)

        self.logger.info(f"Device {self.device_id}: Generating input tensors...")

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_sdxl.generate_input_tensors(
            torch_image=images,
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
            start_latent_seed = requests[0].seed,
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
                f"Device {self.device_id}: Denoising loop for {self.batch_size} prompts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
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
