# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from abc import abstractmethod

import ttnn
from domain.image_generate_request import ImageGenerateRequest
from models.demos.stable_diffusion_xl_base.tests.test_common import (
    SDXL_FABRIC_CONFIG,
    SDXL_L1_SMALL_SIZE,
)
from models.demos.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import (
    TtSDXLPipeline,
)
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.logger import log_exception_chain
from utils.lora_utils import prepare_prompt_with_lora, resolve_lora_path


class BaseSDXLRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.tt_sdxl: TtSDXLPipeline = None
        self.batch_size = 0
        self.pipeline = None
        self._current_lora_path: str | None = None
        self._current_lora_scale: float | None = None

    def get_pipeline_device_params(self):
        device_params = {
            "l1_small_size": SDXL_L1_SMALL_SIZE,
        }
        if self.is_tensor_parallel:
            device_params["fabric_config"] = SDXL_FABRIC_CONFIG
        return device_params

    def _configure_fabric(self, updated_device_params):
        try:
            fabric_config = updated_device_params.pop("fabric_config", None)
            if fabric_config:
                ttnn.set_fabric_config(fabric_config)
            return None
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Fabric configuration failed",
                e,
            )
            raise RuntimeError(f"Fabric configuration failed: {str(e)}") from e

    def load_weights(self):
        try:
            self._load_pipeline()
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Exception during pipeline load",
                e,
            )
            raise
        return True

    @log_execution_time(
        "SDXL warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")
        self.batch_size = self.settings.max_batch_size

        # 1. Load components
        try:
            self._load_pipeline()
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Exception during pipeline load",
                e,
            )
            raise

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
            log_exception_chain(
                self.logger,
                self.device_id,
                "Exception during model loading",
                e,
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
            log_exception_chain(
                self.logger,
                self.device_id,
                "Exception during warmup inference",
                e,
            )
            raise

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @abstractmethod
    def run(self, requests: list[ImageGenerateRequest]):
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
    def _prepare_input_tensors_for_iteration(self, tensors):
        pass

    def _process_prompts(
        self, requests: list[ImageGenerateRequest]
    ) -> tuple[list[str], str, int]:
        batch_size = len(requests)
        needed_padding = self.max_batch_size - batch_size

        prompts = [request.prompt for request in requests] + [""] * needed_padding
        negative_prompts = [request.negative_prompt for request in requests] + [
            ""
        ] * needed_padding
        if negative_prompts == [None]:
            negative_prompts = None

        prompts_2 = requests[0].prompt_2
        if prompts_2 is not None and isinstance(requests[0].prompt_2, str):
            prompts_2 = [requests[0].prompt_2]
        if prompts_2 is not None:
            prompts_2 = prompts_2 + [""] * needed_padding

        negative_prompt_2 = requests[0].negative_prompt_2

        return prompts, negative_prompts, prompts_2, negative_prompt_2, needed_padding

    def _inject_lora_triggers(
        self, prompts: list[str], lora_path: str | None
    ) -> list[str]:
        """Inject LoRA trigger words into non-empty prompts."""
        if not lora_path:
            return prompts
        return [prepare_prompt_with_lora(p, lora_path) for p in prompts]

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

    def _ensure_lora_state(self, request: ImageGenerateRequest) -> None:
        requested_path = request.lora_path
        requested_scale = request.lora_scale

        needs_change = requested_path != self._current_lora_path or (
            requested_path is not None and requested_scale != self._current_lora_scale
        )
        if not needs_change:
            return

        if self._current_lora_path is not None:
            self.logger.info(
                f"Device {self.device_id}: Unloading LoRA adapter: {self._current_lora_path}"
            )
            self.tt_sdxl.unload_lora_weights()
            self._current_lora_path = None
            self._current_lora_scale = None

        if requested_path is not None:
            try:
                local_path = resolve_lora_path(requested_path)
                self.logger.info(
                    f"Device {self.device_id}: Loading LoRA adapter: {requested_path} (scale={requested_scale})"
                )
                self.tt_sdxl.load_lora_weights(local_path)
                self.tt_sdxl.fuse_lora(requested_scale)
                self._current_lora_path = requested_path
                self._current_lora_scale = requested_scale
            except Exception as e:
                self._current_lora_path = None
                self._current_lora_scale = None
                raise RuntimeError(
                    f"Failed to load LoRA adapter '{requested_path}': {e}"
                ) from e

    def _ttnn_inference(self, tensors, prompts, needed_padding):
        images = []
        self.logger.info(f"Device {self.device_id}: Starting ttnn inference...")
        self._prepare_input_tensors_for_iteration(tensors)

        imgs = self.tt_sdxl.generate_images()

        for idx, img in enumerate(imgs):
            if idx >= self.batch_size - needed_padding:
                break

            img = img.unsqueeze(0)
            img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
            images.append(img)

        return images

    def is_request_batchable(self, request, batch=None):
        if len(batch or []) >= self.max_batch_size:
            return False

        if batch is None:
            return True

        first_request = batch[0]
        return (
            request.num_inference_steps == first_request.num_inference_steps
            and request.guidance_scale == first_request.guidance_scale
            and request.guidance_rescale == first_request.guidance_rescale
            and request.crop_coords_top_left == first_request.crop_coords_top_left
            and request.timesteps == first_request.timesteps
            and request.sigmas == first_request.sigmas
            and request.prompt_2 == first_request.prompt_2
            and request.negative_prompt_2 == first_request.negative_prompt_2
            and request.lora_path == first_request.lora_path
            and request.lora_scale == first_request.lora_scale
        )
