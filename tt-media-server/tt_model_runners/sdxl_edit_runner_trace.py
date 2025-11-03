# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import SupportedModels
from diffusers import DiffusionPipeline
from domain.image_edit_request import ImageEditRequest
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
        dummy_data = "R0lGODdhAQABAPAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw=="
        self.run_inference([ImageEditRequest.model_construct(
            prompt="Sunrise on a beach",
            image=dummy_data,
            mask=dummy_data,
            negative_prompt="low resolution",
            num_inference_steps=2,
            guidance_scale=5.0,
            number_of_images=1,
            strength=0.99,
            aesthetic_score=6.0,
            negative_aesthetic_score=2.5,
        )])
    
    def _preprocess_mask(self, base64_mask: str) -> torch.Tensor:
        try:
            pil_mask = self.image_manager.base64_to_pil_image(
                base64_mask, 
                target_size=self.image_size, 
                target_mode=self.image_mode
            )

            mask_tensor = self.tt_sdxl.torch_pipeline.mask_processor.preprocess(
                pil_mask, 
                height=self.image_size[1], 
                width=self.image_size[0], 
                crops_coords=None, 
                resize_mode="default"
            )

            return torch.cat([mask_tensor], dim=0)
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to preprocess mask: {e}")
            raise

    def _process_image_and_mask(self, requests: list[ImageEditRequest]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._preprocess_image(requests[0].image)
        mask = self._preprocess_mask(requests[0].mask)

        masked_image = [i * (m < 0.5) for i, m in zip(image, mask)]
        masked_image = torch.stack(masked_image, dim=0)

        return image, mask, masked_image

    def _prepare_input_tensors_for_iteration(self, tensors, iter: int):
        tt_image_latents, tt_masked_image_latents, tt_mask, tt_prompt_embeds, tt_add_text_embeds = tensors
        self.tt_sdxl.prepare_input_tensors([
            tt_image_latents,
            tt_masked_image_latents,
            tt_mask,
            tt_prompt_embeds[iter],
            tt_add_text_embeds[iter],
        ])


    @log_execution_time("SDXL edit inference")
    def run_inference(self, requests: list[ImageEditRequest]):
        prompts, negative_prompt, prompts_2, negative_prompt_2, needed_padding = self._process_prompts(requests)

        self._apply_request_settings(requests[0])
        self._apply_image_to_image_request_settings(requests[0])

        self.logger.debug(f"Device {self.device_id}: Starting text encoding...")
        self.tt_sdxl.compile_text_encoding()

        all_prompt_embeds_torch, torch_add_text_embeds = self.tt_sdxl.encode_prompts(
            prompts, negative_prompt, prompts_2, negative_prompt_2
        )

        image, mask, masked_image = self._process_image_and_mask(requests)

        self.logger.info(f"Device {self.device_id}: Generating input tensors...")

        (
            tt_image_latents,
            tt_masked_image_latents,
            tt_mask,
            tt_prompt_embeds,
            tt_add_text_embeds,
        ) = self.tt_sdxl.generate_input_tensors(
            torch_image=image,
            torch_masked_image=masked_image,
            torch_mask=mask,
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
            start_latent_seed=requests[0].seed,
            timesteps=requests[0].timesteps,
            sigmas=requests[0].sigmas
        )
        
        self.logger.debug(f"Device {self.device_id}: Preparing input tensors...") 

        tensors = (
            tt_image_latents,
            tt_masked_image_latents,
            tt_mask,
            tt_prompt_embeds,
            tt_add_text_embeds,
        )
        self._prepare_input_tensors_for_iteration(tensors, 0)

        self.logger.debug(f"Device {self.device_id}: Compiling image processing...")

        self.tt_sdxl.compile_image_processing()

        profiler.clear()

        return self._ttnn_inference(tensors, prompts, needed_padding)