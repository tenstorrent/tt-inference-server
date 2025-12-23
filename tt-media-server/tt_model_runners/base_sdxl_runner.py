# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from abc import abstractmethod
import ttnn

from domain.image_generate_request import ImageGenerateRequest
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_FABRIC_CONFIG,
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import (
    TtSDXLPipeline,
)
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.helpers import log_execution_time


class BaseSDXLRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.tt_sdxl: TtSDXLPipeline = None
        self.batch_size = 0
        self.pipeline = None

    def get_pipeline_device_params(self):
        device_params = {
            "l1_small_size": SDXL_L1_SMALL_SIZE,
            "trace_region_size": self.settings.trace_region_size
            or SDXL_TRACE_REGION_SIZE,
        }
        if self.is_tensor_parallel:
            device_params["fabric_tensix_config"] = SDXL_FABRIC_CONFIG
            from loguru import logger
            logger.warning(f"I {device_params=}")
        return device_params

    def _configure_fabric(self, updated_device_params):
        fabric_config = updated_device_params.pop("fabric_tensix_config", None)
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)
        return None

    @log_execution_time(
        "SDXL warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")
        self.batch_size = self.settings.max_batch_size

        # 1. Load components
        self._load_pipeline()

        self.logger.info(
            f"Device {self.device_id}: Model weights downloaded successfully"
        )

        # 6 minutes to distribute the model on device
        weights_distribution_timeout = 720

        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._distribute_block),
                timeout=weights_distribution_timeout,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                f"Device {self.device_id}: ttnn.distribute block timed out after {weights_distribution_timeout} seconds"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Exception during model loading: {e}"
            )
            raise

        self.logger.info(f"Device {self.device_id}: Model loaded successfully")

        warmup_inference_timeout = 1000

        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._warmup_inference_block),
                timeout=warmup_inference_timeout,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                f"Device {self.device_id}: warmup inference timed out after {warmup_inference_timeout} seconds"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Exception during warmup inference: {e}"
            )
            raise

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @abstractmethod
    def run_inference(self, requests: list[ImageGenerateRequest]):
        pass

    @abstractmethod
    def _load_pipeline(self):
        pass

    @abstractmethod
    def _distribute_block(self):
        pass

    @abstractmethod
    def _warmup_inference_block(self):
        pass

    @abstractmethod
    def _prepare_input_tensors_for_iteration(self, iter: int):
        pass

    def _process_prompts(
        self, request: ImageGenerateRequest
    ) -> tuple[list[str], str, int]:
        prompts = request.prompt
        negative_prompt = request.negative_prompt
        if isinstance(prompts, str):
            prompts = [prompts]

        needed_padding = (
            self.batch_size - len(prompts) % self.batch_size
        ) % self.batch_size
        prompts = prompts + [""] * needed_padding

        prompts_2 = request.prompt_2
        negative_prompt_2 = request.negative_prompt_2
        if prompts_2 is not None:
            prompts_2 = request.prompt_2
            if isinstance(prompts_2, str):
                prompts_2 = [prompts_2]

            needed_padding = (
                self.batch_size - len(prompts_2) % self.batch_size
            ) % self.batch_size
            prompts_2 = prompts_2 + [""] * needed_padding

        return prompts, negative_prompt, prompts_2, negative_prompt_2, needed_padding

    def _apply_request_settings(self, request: ImageGenerateRequest) -> None:
        if request.num_inference_steps is not None:
            self.tt_sdxl.set_num_inference_steps(request.num_inference_steps)

        if request.guidance_scale is not None:
            self.tt_sdxl.set_guidance_scale(request.guidance_scale)

        if request.guidance_rescale is not None:
            self.tt_sdxl.set_guidance_rescale(request.guidance_rescale)

        if request.crop_coords_top_left is not None:
            self.tt_sdxl.set_crop_coords_top_left(request.crop_coords_top_left)

        if request.timesteps is not None and request.sigmas is not None:
            raise ValueError("Cannot pass both timesteps and sigmas. Choose one.")

    def _ttnn_inference(self, tensors, prompts, needed_padding):
        images = []
        self.logger.info(f"Device {self.device_id}: Starting ttnn inference...")
        for iter in range(len(prompts) // self.batch_size):
            self.logger.info(
                f"Device {self.device_id}: Running inference for prompts {iter * self.batch_size + 1}-{iter * self.batch_size + self.batch_size}/{len(prompts)}"
            )

            self._prepare_input_tensors_for_iteration(tensors, iter)

            imgs = self.tt_sdxl.generate_images()

            for idx, img in enumerate(imgs):
                if (
                    iter == len(prompts) // self.batch_size - 1
                    and idx >= self.batch_size - needed_padding
                ):
                    break
                img = img.unsqueeze(0)
                img = self.pipeline.image_processor.postprocess(img, output_type="pil")[
                    0
                ]
                images.append(img)

        return images
