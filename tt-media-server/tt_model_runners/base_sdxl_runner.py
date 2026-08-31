# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import inspect
import os
import time
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
from telemetry.image_metrics import (
    SdxlSectionTimings,
    add_conditioning_seconds,
    record_image_run,
    resolution_of_images,
    sampler_name,
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
        # Measured around encode_prompts(), flushed in _ttnn_inference().
        self._conditioning_seconds: dict[str, float] = {}
        # Warmup calls run(); recording it would skew the stage metrics.
        self._warming_up = False

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

        weights_distribution_timeout = (
            self.settings.weights_distribution_timeout_seconds
        )

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

        self._warming_up = True
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
        finally:
            self._warming_up = False

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

    def _encode_prompts(self, *args, **kwargs):
        """Time tt_sdxl.encode_prompts().

        It drives both CLIP encoders in one call, so there is no per-encoder
        breakdown to report -- only ``encoder="all"`` until image encode is
        added by :meth:`_generate_input_tensors`.
        """
        self._conditioning_seconds = {}
        start = time.perf_counter()
        result = self.tt_sdxl.encode_prompts(*args, **kwargs)
        if not self._warming_up:
            add_conditioning_seconds(
                self._conditioning_seconds, "all", time.perf_counter() - start
            )
        return result

    def _generate_input_tensors(self, *args, **kwargs):
        """Time image conditioning on img2img / inpaint.

        txt2img has no ``torch_image`` and is not counted as conditioning.

        ``torch_image`` is the third *positional* parameter of the img2img
        pipeline's ``generate_input_tensors``, so the argument is resolved
        against the real signature rather than read out of ``kwargs`` -- a
        positional call would otherwise silently stop counting.
        """
        start = time.perf_counter()
        result = self.tt_sdxl.generate_input_tensors(*args, **kwargs)
        if not self._warming_up and self._has_image_conditioning(args, kwargs):
            add_conditioning_seconds(
                self._conditioning_seconds, "image", time.perf_counter() - start
            )
        return result

    def _has_image_conditioning(self, args, kwargs) -> bool:
        """True when ``generate_input_tensors`` was given a ``torch_image``."""
        if "torch_image" in kwargs:
            return kwargs["torch_image"] is not None
        if not args:
            return False
        try:
            bound = inspect.signature(self.tt_sdxl.generate_input_tensors).bind_partial(
                *args, **kwargs
            )
        except (TypeError, ValueError):
            return False
        return bound.arguments.get("torch_image") is not None

    def _ttnn_inference(self, tensors, prompts, needed_padding):
        images = []
        self.logger.info(f"Device {self.device_id}: Starting ttnn inference...")
        self._prepare_input_tensors_for_iteration(tensors)

        # generate_images() resets the step count when it returns, so read it now.
        num_steps = getattr(self.tt_sdxl, "num_inference_steps", None)
        with SdxlSectionTimings() as timings:
            imgs = self.tt_sdxl.generate_images()

        for idx, img in enumerate(imgs):
            if idx >= self.batch_size - needed_padding:
                break

            img = img.unsqueeze(0)
            img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
            images.append(img)

        self._record_image_stage_metrics(images, needed_padding, num_steps, timings)

        return images

    def _record_image_stage_metrics(self, images, needed_padding, num_steps, timings):
        """Export the stage metrics for one SDXL generation."""
        if self._warming_up:
            return
        try:
            conditioning = self._conditioning_seconds
            self._conditioning_seconds = {}

            resolution, image_count, pixels = resolution_of_images(images)
            batch = max(self.batch_size - needed_padding, len(images), 1)

            # image_gen covers denoising + VAE only; add conditioning back so
            # the engine total matches tt_dit's "total" span.
            engine_seconds = timings.engine_seconds
            if engine_seconds is not None and "all" in conditioning:
                engine_seconds += conditioning["all"]

            record_image_run(
                model_type=self.settings.model_runner,
                device_id=self.device_id,
                resolution=resolution,
                sampler=sampler_name(self.pipeline),
                batch=batch,
                engine_seconds=engine_seconds,
                denoise_seconds=timings.denoise_seconds,
                step_count=num_steps,
                vae_seconds=timings.vae_seconds,
                conditioning_seconds=conditioning,
                image_count=image_count,
                pixels_per_image=pixels,
            )
        except Exception as exc:
            # Never fail an inference that already succeeded.
            self.logger.warning(f"Failed to record image stage metrics: {exc}")

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
