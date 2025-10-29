# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import SupportedModels
from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_sdxl_runner import BaseSDXLRunner
from utils.helpers import log_execution_time
import torch
from diffusers import DiffusionPipeline
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig

class TTSDXLGenerateRunnerTrace(BaseSDXLRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.settings.model_weights_path or SupportedModels.STABLE_DIFFUSION_XL_BASE.value,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

    def _distribute_block(self):
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

    def _warmup_inference_block(self):
        self.run_inference([ImageGenerateRequest.model_construct(
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

    @log_execution_time("SDXL generate inference")
    def run_inference(self, requests: list[ImageGenerateRequest]):
        prompts, negative_prompt, prompts_2, negative_prompt_2, needed_padding = self._process_prompts(requests)

        self._apply_request_settings(requests[0])
        
        self.logger.debug(f"Device {self.device_id}: Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        ) = self.tt_sdxl.encode_prompts(prompts, negative_prompt, prompts_2, negative_prompt_2)

        self.logger.info(f"Device {self.device_id}: Generating input tensors...")

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_sdxl.generate_input_tensors(
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
    