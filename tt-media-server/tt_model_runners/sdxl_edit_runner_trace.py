# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import SupportedModels
from diffusers import DiffusionPipeline
from domain.image_edit_request import EditImageRequest
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_inpainting_pipeline import TtSDXLInpaintingPipeline, TtSDXLInpaintingPipelineConfig
import torch
from tt_model_runners.sdxl_image_to_image_runner_trace import TTSDXLImageToImageRunner
from utils.helpers import log_execution_time

class TTSDXLEditRunner(TTSDXLImageToImageRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.settings.model_weights_path or SupportedModels.STABLE_DIFFUSION_XL_INPAINTING.value,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )


    def _distribute_block(self):
        self.tt_sdxl = TtSDXLInpaintingPipeline(
            ttnn_device=self.ttnn_device,
            torch_pipeline=self.pipeline,
            pipeline_config=TtSDXLInpaintingPipelineConfig(
                encoders_on_device=True,
                is_galaxy=self.settings.is_galaxy,
                num_inference_steps=self.settings.num_inference_steps,
                guidance_scale=5.0,
                use_cfg_parallel=self.is_tensor_parallel,
            ),        
        )


    def _warmup_inference_block(self):
        self.run_inference([EditImageRequest.model_construct(
            prompt="Sunrise on a beach",
            image="R0lGODdhAQABAPAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==",  # 1x1 transparent pixel
            negative_prompt="low resolution",
            num_inference_steps=2,
            guidance_scale=5.0,
            number_of_images=1,
            strength=0.99,
            aesthetic_score=6.0,
            negative_aesthetic_score=2.5,
        )])


    @log_execution_time("SDXL edit inference")
    def run_inference(self, requests: list[EditImageRequest]):
        prompts, negative_prompt, prompts_2, negative_prompt_2, needed_padding = self._process_prompts(requests)

        self._apply_request_settings(requests[0])

        self._apply_image_to_image_request_settings(requests[0])

        self.logger.debug(f"Device {self.device_id}: Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = self.tt_sdxl.encode_prompts(prompts, negative_prompt, prompts_2, negative_prompt_2)

        image = self._preprocess_image(requests[0].image)

        self.logger.info(f"Device {self.device_id}: Generating input tensors...")

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_sdxl.generate_input_tensors(
            torch_image=image,
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
            start_latent_seed=requests[0].seed,
            timesteps=requests[0].timesteps,
            sigmas=requests[0].sigmas

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

        return self._ttnn_inference(tt_latents, tt_prompt_embeds, tt_add_text_embeds, prompts, needed_padding)