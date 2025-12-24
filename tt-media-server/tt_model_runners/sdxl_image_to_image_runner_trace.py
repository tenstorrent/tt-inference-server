# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

import torch
from config.constants import SupportedModels
from diffusers import StableDiffusionXLImg2ImgPipeline
from domain.image_to_image_request import ImageToImageRequest
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLImg2ImgPipeline,
    TtSDXLImg2ImgPipelineConfig,
)
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_sdxl_runner import BaseSDXLRunner
from utils.decorators import log_execution_time
from utils.image_manager import ImageManager


class TTSDXLImageToImageRunner(BaseSDXLRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.image_manager = ImageManager("img")
        self.image_size = (1024, 1024)
        self.image_mode = "RGB"

    def _load_pipeline(self):
        self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.settings.model_weights_path
            or SupportedModels.STABLE_DIFFUSION_XL_IMG2IMG.value,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

    def _distribute_block(self):
        self.tt_sdxl = TtSDXLImg2ImgPipeline(
            ttnn_device=self.ttnn_device,
            torch_pipeline=self.pipeline,
            pipeline_config=TtSDXLImg2ImgPipelineConfig(
                encoders_on_device=True,
                is_galaxy=self.settings.is_galaxy,
                num_inference_steps=2,
                guidance_scale=5.0,
                use_cfg_parallel=self.is_tensor_parallel,
            ),
        )

    def _warmup_inference_block(self):
        self.run_inference(
            [
                ImageToImageRequest.model_construct(
                    prompt="Sunrise on a beach",
                    image="R0lGODdhAQABAPAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==",  # 1x1 transparent pixel
                    negative_prompt="low resolution",
                    num_inference_steps=2,
                    guidance_scale=5.0,
                    number_of_images=1,
                    strength=0.99,
                    aesthetic_score=6.0,
                    negative_aesthetic_score=2.5,
                )
            ]
        )

    def _preprocess_image(self, base64_image: str) -> torch.Tensor:
        try:
            pil_image = self.image_manager.base64_to_pil_image(
                base64_image, target_size=self.image_size, target_mode=self.image_mode
            )

            image_tensor = self.tt_sdxl.torch_pipeline.image_processor.preprocess(
                pil_image,
                height=self.image_size[1],
                width=self.image_size[0],
                crops_coords=None,
                resize_mode="default",
            ).to(dtype=torch.float32)

            return torch.cat([image_tensor], dim=0)

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to preprocess image: {e}"
            )
            raise

    def _apply_image_to_image_request_settings(
        self, request: ImageToImageRequest
    ) -> None:
        if request.strength is not None:
            self.tt_sdxl.set_strength(request.strength)

        """ TODO: Reintroduce these fields when https://github.com/tenstorrent/tt-metal/issues/31032 is resolved
        if request.aesthetic_score is not None:
            self.tt_sdxl.set_aesthetic_score(request.aesthetic_score)

        if request.negative_aesthetic_score is not None:
            self.tt_sdxl.set_negative_aesthetic_score(request.negative_aesthetic_score)
        """

    def _prepare_input_tensors_for_iteration(self, tensors):
        tt_image_latents, tt_prompt_embeds, tt_add_text_embeds = tensors
        self.tt_sdxl.prepare_input_tensors(
            [
                tt_image_latents,
                tt_prompt_embeds[0],
                tt_add_text_embeds[0],
            ]
        )

    @log_execution_time(
        "SDXL image-to-image inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run_inference(self, requests: list[ImageToImageRequest]):
        prompts, negative_prompts, prompts_2, negative_prompt_2, needed_padding = (
            self._process_prompts(requests)
        )

        self._apply_request_settings(requests[0])
        self._apply_image_to_image_request_settings(requests[0])

        self.logger.debug(f"Device {self.device_id}: Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = self.tt_sdxl.encode_prompts(
            prompts, negative_prompts, prompts_2, negative_prompt_2
        )

        images = [self._preprocess_image(request.image) for request in requests]
        images = torch.cat(images, dim=0)

        self.logger.info(f"Device {self.device_id}: Generating input tensors...")

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = (
            self.tt_sdxl.generate_input_tensors(
                torch_image=images,
                all_prompt_embeds_torch=all_prompt_embeds_torch,
                torch_add_text_embeds=torch_add_text_embeds,
                start_latent_seed=requests[0].seed,
                timesteps=requests[0].timesteps,
                sigmas=requests[0].sigmas,
            )
        )

        self.logger.debug(f"Device {self.device_id}: Preparing input tensors...")

        tensors = (
            tt_latents,
            tt_prompt_embeds,
            tt_add_text_embeds,
        )
        self._prepare_input_tensors_for_iteration(tensors)

        self.logger.debug(f"Device {self.device_id}: Compiling image processing...")

        self.tt_sdxl.compile_image_processing()

        return self._ttnn_inference(tensors, prompts, needed_padding)

    def is_request_batchable(self, request, batch=None):
        return super().is_request_batchable(request, batch) and (
            batch is None or request.strength == batch[0].strength
        )
