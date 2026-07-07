# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import os
import uuid
from abc import abstractmethod
from pathlib import Path

import numpy as np
import ttnn
from config.constants import (
    WAN22_ANISORA_NUM_STEPS,
    WAN22_DISTILL_NUM_STEPS,
    WAN22_LIGHTNING_NUM_STEPS,
    WAN22_NUM_FRAMES,
    ModelRunners,
    ModelServices,
    SupportedModels,
    is_large_mesh,
    wan22_target_resolution,
)
from config.settings import get_settings
from domain.image_generate_request import ImageGenerateRequest
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import ImagePromptEntry, VideoI2VGenerateRequest
from huggingface_hub import hf_hub_download
from models.common.utility_functions import is_blackhole
from models.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from models.tt_dit.pipelines.flux1.pipeline_flux1_kontext import (
    Flux1KontextPipeline,
)
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    resolve_mesh_preset,
)
from models.tt_dit.pipelines.mochi.pipeline_mochi import MochiPipeline
from models.tt_dit.pipelines.motif.pipeline_motif import MotifPipeline
from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    QwenImagePipeline,
)
from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
)
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import (
    ImagePrompt,
    WanPipelineI2V,
)
from PIL import Image
from telemetry.image_metrics import ImageStageRecorder, sampler_name
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from tt_model_runners.minimax_h3_policy import (
    MINIMAX_H3_ASPECT_RATIOS,
    MINIMAX_H3_DEFAULT_ASPECT_RATIO,
    MINIMAX_H3_DEFAULT_DURATION_S,
    MINIMAX_H3_DURATIONS_S,
    MINIMAX_H3_NUM_INFERENCE_STEPS,
    minimax_h3_frames_are_aligned,
    minimax_h3_parse_aspect_ratio,
)
from utils.decorators import log_execution_time
from utils.image_manager import ImageManager
from utils.logger import log_exception_chain

dit_runner_log_map = {
    ModelRunners.TT_SD3_5.value: "SD35",
    ModelRunners.TT_FLUX_1_DEV.value: "FLUX.1-dev",
    ModelRunners.TT_FLUX_1_SCHNELL.value: "FLUX.1-schnell",
    ModelRunners.TT_FLUX_1_KONTEXT_DEV.value: "FLUX.1-Kontext-dev",
    ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW.value: "Motif-Image-6B-Preview",
    ModelRunners.TT_MOCHI_1.value: "Mochi1",
    ModelRunners.TT_WAN_2_2.value: "Wan22",
    ModelRunners.TT_WAN_2_2_T2V_PRODIA.value: "Wan22-T2V-Prodia",
    ModelRunners.TT_WAN_2_2_I2V.value: "Wan22-I2V",
    ModelRunners.TT_WAN_2_2_I2V_PRODIA.value: "Wan22-I2V-Prodia",
    ModelRunners.TT_WAN_2_2_I2V_ANISORA.value: "Wan22-I2V-AniSora",
    ModelRunners.TT_WAN_2_2_I2V_DISTILL.value: "Wan22-I2V-Distill",
    ModelRunners.TT_WAN_2_2_I2V_LORA.value: "Wan22-I2V-LoRA",
    ModelRunners.TT_WAN_2_2_I2V_LIGHTNING.value: "Wan22-I2V-Lightning",
    ModelRunners.TT_LTX_2_3_DISTILLED.value: "LTX-2.3-distilled",
    ModelRunners.TT_MINIMAX_H3_T2VA.value: "MiniMaxH3-T2VA",
    ModelRunners.TT_QWEN_IMAGE.value: "Qwen-Image",
    ModelRunners.TT_QWEN_IMAGE_2512.value: "Qwen-Image-2512",
    ModelRunners.SP_RUNNER.value: "SP-Runner",
}

DIT_WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS = 6000


class TTDiTRunner(BaseMetalDeviceRunner):
    # Set True by runners that require image conditioning (I2V). The SP inference
    # loop uses this to reject image-less requests with a clean per-request error
    # instead of letting a base VideoGenerateRequest reach the runner and crash
    # on the missing ``image_prompts`` field.
    requires_image_conditioning: bool = False

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None
        # Warmup calls run(); recording it would skew the stage metrics.
        self._warming_up = False

    def _configure_fabric(self, updated_device_params):
        try:
            fabric_config = updated_device_params.pop(
                "fabric_config", ttnn.FabricConfig.FABRIC_1D
            )
            fabric_tensix_config = updated_device_params.pop(
                "fabric_tensix_config", ttnn.FabricTensixConfig.DISABLED
            )
            reliability_mode = updated_device_params.pop(
                "reliability_mode", ttnn.FabricReliabilityMode.STRICT_INIT
            )
            fabric_router_config = updated_device_params.pop(
                "fabric_router_config", ttnn.FabricRouterConfig()
            )
            ttnn.set_fabric_config(
                fabric_config,
                reliability_mode,
                None,
                fabric_tensix_config,
                ttnn.FabricUDMMode.DISABLED,
                ttnn.FabricManagerMode.DEFAULT,
                fabric_router_config,
            )
            return fabric_config
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Fabric configuration failed",
                e,
            )
            raise RuntimeError(f"Fabric configuration failed: {str(e)}") from e

    @abstractmethod
    def create_pipeline(self):
        """Create a pipeline for the model"""

    @abstractmethod
    def get_pipeline_device_params(self):
        """Get the device parameters for the pipeline"""

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def load_weights(self):
        return True  # weights will be loaded upon pipeline creation

    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        def distribute_block():
            self.pipeline = self.create_pipeline()

        weights_distribution_timeout = max(
            self.settings.weights_distribution_timeout_seconds,
            DIT_WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS,
        )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(distribute_block),
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

        # we use model_construct to create the request without validation
        # (warmup uses 2 inference steps which is below the normal minimum)
        self._warming_up = True
        try:
            if self.settings.model_service == ModelServices.IMAGE.value:
                self.run(
                    [
                        ImageGenerateRequest.model_construct(
                            prompt="Sunrise on a beach",
                            negative_prompt="",
                            num_inference_steps=2,
                        )
                    ],
                )
            elif self.settings.model_service == ModelServices.VIDEO.value:
                self.run([self._build_warmup_video_request()])
        finally:
            self._warming_up = False

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def _build_warmup_video_request(self) -> VideoGenerateRequest:
        """
        Build the throwaway request used to trigger compile/trace on warmup.
        """
        return VideoGenerateRequest.model_construct(
            prompt="Sunrise on a beach",
            negative_prompt="",
            num_inference_steps=2,
        )

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[ImageGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        request = requests[0]
        # run_single_prompt is single-prompt by construction, hence batch=1.
        recorder = (
            None
            if self._warming_up
            else ImageStageRecorder(
                model_type=self.settings.model_runner,
                device_id=self.device_id,
                sampler=sampler_name(self.pipeline),
                batch=1,
            )
        )
        image = self.pipeline.run_single_prompt(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            seed=int(request.seed or 0),
            on_event=recorder,
        )
        if recorder is not None:
            recorder.flush(image)
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return image


class TTSD35Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        try:
            return StableDiffusion3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "SD3.5 pipeline creation failed",
                e,
            )
            raise

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 25000000}


# Runner for Flux.1 dev and schnell. Model weights from settings.model_weights_path determine the exact model variant.
class TTFlux1Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        try:
            return Flux1Pipeline.create_pipeline(
                checkpoint_name=self.settings.model_weights_path,
                mesh_device=self.ttnn_device,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Flux1 pipeline creation failed",
                e,
            )
            raise

    def get_pipeline_device_params(self):
        return {
            "l1_small_size": 32768,
            "trace_region_size": self.settings.trace_region_size,
        }


# FLUX.1-Kontext-dev: instruction-based image editing (reference image + prompt).
# Shares the FLUX transformer/VAE/encoders with TTFlux1Runner but takes an input
# image, so run() is overridden to feed it into the pipeline. No mask is used
# (unlike SDXL edit/inpainting) — requests arrive as ImageToImageRequest.
class TTFluxKontextRunner(TTDiTRunner):
    _KONTEXT_GUIDANCE_SCALE = 3.5  # BFL-recommended default for Kontext-dev

    def __init__(self, device_id: str):
        super().__init__(device_id)

    @staticmethod
    def _active_lora():
        """Read the active LoRA (path, scale) from the shared state file, if any.
        Written by the /v1/lora/apply endpoint; read here so a worker restart
        (deep_reset) rebuilds the pipeline with the LoRA fused in."""
        import json
        import os

        state = os.path.join(os.environ.get("LORA_DIR", "/loras"), "active_lora.json")
        try:
            if os.path.isfile(state):
                d = json.load(open(state))
                p = d.get("path")
                if p and os.path.isfile(p):
                    return p, float(d.get("scale", 1.0))
        except Exception:
            pass
        return None, 1.0

    def create_pipeline(self):
        try:
            lora_path, lora_scale = self._active_lora()
            if lora_path:
                self.logger.info(
                    f"Device {self.device_id}: building with LoRA {lora_path} (scale={lora_scale})"
                )
            return Flux1KontextPipeline.create_pipeline(
                checkpoint_name=self.settings.model_weights_path,
                mesh_device=self.ttnn_device,
                lora_path=lora_path,
                lora_scale=lora_scale,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Flux1-Kontext pipeline creation failed",
                e,
            )
            raise

    def get_pipeline_device_params(self):
        return {
            "l1_small_size": 32768,
            "trace_region_size": self.settings.trace_region_size,
        }

    def _decode_image(self, image_b64: str, width: int, height: int) -> Image.Image:
        # strip an optional data-URL prefix (e.g. "data:image/png;base64,")
        if "," in image_b64 and image_b64.strip().startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB").resize((width, height))

    def run(self, requests: list[ImageGenerateRequest]):
        request = requests[0]
        # Per-request resolution (falls back to the pipeline's configured size;
        # the pipeline snaps to the nearest Kontext preferred bucket).
        width = getattr(request, "width", None) or getattr(
            self.pipeline, "_width", 1024
        )
        height = getattr(request, "height", None) or getattr(
            self.pipeline, "_height", 1024
        )

        # Edit mode = a reference image is supplied; otherwise text-to-image.
        image_b64 = getattr(request, "image", None)
        image = self._decode_image(image_b64, width, height) if image_b64 else None
        mode = "edit" if image is not None else "generate"

        # NOTE: JP->EN translation (when translate=True) is applied in the API
        # layer (open_ai_api/image.py) before the request reaches the worker, so
        # request.prompt is already English here.
        prompt = request.prompt

        self.logger.info(
            f"Device {self.device_id}: Kontext {mode} inference "
            f"({width}x{height}, {request.num_inference_steps} steps)"
        )
        images = self.pipeline(
            image=image,
            width=width,
            height=height,
            prompts=[prompt],
            num_inference_steps=request.num_inference_steps,
            guidance_scale=self._KONTEXT_GUIDANCE_SCALE,  # Kontext default (ignore SDXL 5.0 default)
            seed=int(request.seed or 0),
            # Siblings trace for throughput; the Kontext path is validated untraced,
            # so start conservative.
            traced=False,
        )
        self.logger.debug(f"Device {self.device_id}: Kontext inference completed")
        return images


class TTMotifImage6BPreviewRunner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        try:
            return MotifPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=SupportedModels.MOTIF_IMAGE_6B_PREVIEW.value,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Motif pipeline creation failed",
                e,
            )
            raise

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 31000000}


# Runner for Qwen-Image and Qwen-Image-2512. Model weights from settings.model_weights_path determine the exact model variant.
class TTQwenImageRunner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        try:
            return QwenImagePipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Qwen-Image pipeline creation failed",
                e,
            )
            raise

    def get_pipeline_device_params(self):
        return {"trace_region_size": 47000000}


class TTMochi1Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def create_pipeline(self):
        try:
            return MochiPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=SupportedModels.MOCHI_1.value,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Mochi pipeline creation failed",
                e,
            )
            raise

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        request = requests[0]
        # MochiPipeline.__call__ takes prompts/negative_prompts lists since the
        # tt_dit pipeline refactor; num_frames/height/width/output_type moved to
        # MochiPipelineConfig defaults (168 frames, 480x848).
        frames = self.pipeline(
            prompts=[request.prompt],
            negative_prompts=[request.negative_prompt or ""],
            num_inference_steps=request.num_inference_steps,
            guidance_scale=3.5,
            seed=int(request.seed or 0),
        )
        # The pipeline returns PIL frames (or a tensor); the video exporter
        # needs a (batch, frames, H, W, C) uint8 array.
        if hasattr(frames, "cpu"):
            frames = frames.cpu().numpy()
        elif isinstance(frames, list):
            frames = np.stack(
                [np.stack([np.asarray(f) for f in video]) for video in frames]
            )
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    def get_pipeline_device_params(self):
        return {}


WAN22_BH_RING_MESH_SHAPES = frozenset({(1, 4)})

WAN22_GALAXY_BH_TRACE_REGION_BYTES = 125_000_000
WAN22_GALAXY_ROUTER_MAX_PAYLOAD_BYTES = 8192

# The LightX2V 4-step distill, with the fast-VAE-encode + on-device conditioning
# optimizations enabled, captures a larger trace than the shared 125MB default
# (the fully-optimized 4x32 traced run needs ~200MB; 125MB hits the TT_FATAL
# `trace_buffers_size <= trace_region_size` during warmup). Distill-only so the
# other Wan2.2 runners keep the shared default.
WAN22_DISTILL_BH_TRACE_REGION_BYTES = 200_000_000

# Fast-image-encode optimizations for the LightX2V distill pipeline. Enabling all
# three takes the traced 4x32 pipeline from ~6-7s to ~4s with no quality loss
# (validated via per-frame PCC + visual checks against the full-encode baseline):
#   - WAN_DISTILL_FAST_VAE_ENCODER: rebuild the VAE encoder at the real resolution
#     so it keys the swept conv3d blockings (encoder compute 1.44s -> 0.21s).
#   - WAN_DISTILL_ENCODER_T_OUT_1: cap conv3d T_out_block at 1, which removes the
#     temporal-blocking "duplicate subject" artifact the swept encoder otherwise
#     introduces in the 4-step distill. MUST accompany FAST_VAE_ENCODER.
#   - WAN_DISTILL_ONDEVICE_COND: assemble the (mostly-zero) conditioning video on
#     device instead of transferring it from host (prepare_latents 2.99s -> 0.38s).
# Set via setdefault so a deployment can still pin any flag to "0" to disable.
WAN_DISTILL_FAST_ENCODE_FLAGS = {
    "WAN_DISTILL_FAST_VAE_ENCODER": "1",
    "WAN_DISTILL_ENCODER_T_OUT_1": "1",
    "WAN_DISTILL_ONDEVICE_COND": "1",
}

# AniSora V3.2 reuses the same fast-image-encode optimizations (via the shared
# FastImageEncodeMixin) behind AniSora-scoped flags. All three enabled takes the
# traced 8-step 4x32 pipeline image-encode from ~7.5s to ~0.35s (total ~16s ->
# ~9.3s) with quality matching the full-encode baseline. ENCODER_T_OUT_1 MUST
# accompany FAST_VAE_ENCODER to avoid the temporal-blocking artifact.
WAN_ANISORA_FAST_ENCODE_FLAGS = {
    "WAN_ANISORA_FAST_VAE_ENCODER": "1",
    "WAN_ANISORA_ENCODER_T_OUT_1": "1",
    "WAN_ANISORA_ONDEVICE_COND": "1",
}

# AniSora runs real CFG (guidance 3.5 on both experts) and its fully-optimized
# 8-step trace needs the same ~200MB region as the distill (the shared 125MB
# default OOMs during warmup).
WAN22_ANISORA_BH_TRACE_REGION_BYTES = 200_000_000
WAN22_ANISORA_GUIDANCE_SCALE = 3.5
# AniSora / Lightning / Distill step counts live in config.constants
# (WAN22_*_NUM_STEPS) so telemetry can use the same values.

WAN22_LIGHTNING_BOUNDARY_RATIO = 0.875
WAN22_LIGHTNING_FLOW_SHIFT = 5.0


def _wan22_needs_ring_fabric(mesh_shape: tuple) -> bool:
    """Return True when Wan2.2 must advertise FABRIC_1D_RING for ``mesh_shape``."""
    if is_large_mesh(mesh_shape):
        return True
    return is_blackhole() and tuple(mesh_shape) in WAN22_BH_RING_MESH_SHAPES


def _wan22_galaxy_router_config():
    """Build the FabricRouterConfig used by Galaxy-class BH meshes."""
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = WAN22_GALAXY_ROUTER_MAX_PAYLOAD_BYTES
    return config


def _wan22_dit_device_params(mesh_shape: tuple) -> dict:
    """Resolve fabric / trace-region defaults shared by Wan2.2 T2V and I2V runners."""
    fabric_config = (
        ttnn.FabricConfig.FABRIC_1D_RING
        if _wan22_needs_ring_fabric(mesh_shape)
        else ttnn.FabricConfig.FABRIC_1D
    )
    device_params: dict = {"fabric_config": fabric_config}

    if is_blackhole():
        device_params["reliability_mode"] = ttnn.FabricReliabilityMode.RELAXED_INIT

    if is_large_mesh(mesh_shape) and is_blackhole():
        device_params["trace_region_size"] = WAN22_GALAXY_BH_TRACE_REGION_BYTES
        device_params["fabric_router_config"] = _wan22_galaxy_router_config()

    return device_params


def _wan22_pipeline_args(
    request,
    resolution=None,
    image_prompt=None,
):
    """Build the kwargs dict shared by Wan2.2 T2V and I2V ``__call__`` sites."""
    seed = int(request.seed) if request.seed is not None else 0
    pipeline_args = {
        "prompts": [request.prompt],
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "seed": seed,
        "traced": True,
    }
    if image_prompt is not None:
        pipeline_args["image_prompt"] = image_prompt
    if bool(request.negative_prompt):
        pipeline_args["negative_prompts"] = [request.negative_prompt]
    return pipeline_args


class TTWan22Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)

    def create_pipeline(self):
        try:
            return WanPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
                height=self.resolution.height,
                width=self.resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Wan pipeline creation failed",
                e,
            )
            raise

    def load_weights(self):
        return False

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        frames = self.pipeline(**_wan22_pipeline_args(requests[0], self.resolution))
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    def get_pipeline_device_params(self):
        return _wan22_dit_device_params(self.settings.device_mesh_shape)


def _prodia_wan_device_params(device_mesh_shape) -> dict:
    """Shared device params for Prodia Wan T2V/I2V distilled runners.

    The 4x8 LoudBox trace binary needs ~30.6MB; the default 30MB region
    rejects it and warmup OOMs. Both 4x8 (32 chips) and 4x32 (128 chips)
    Blackhole meshes get the bumped trace region.
    """
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
    mesh_size = device_mesh_shape[0] * device_mesh_shape[1]
    if mesh_size >= 32 and is_blackhole():
        device_params["trace_region_size"] = 150_000_000
        config = ttnn.FabricRouterConfig()
        config.max_packet_payload_size_bytes = 8192
        device_params["fabric_router_config"] = config
    return device_params


class TTWan22T2VProdiaRunner(TTDiTRunner):
    """Wan2.2 T2V runner using the Prodia distilled pipeline (no image prompt)."""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        # Export MP4 inside the device worker by default to avoid pickling the
        # raw frame array (~226MB at 720p×81 frames) over IPC.
        self.export_in_runner = True

    def load_weights(self):
        return False

    def get_pipeline_device_params(self):
        return _prodia_wan_device_params(self.settings.device_mesh_shape)

    def create_pipeline(self):
        try:
            from pipelines.pipeline import create_pipeline

            resolution = wan22_target_resolution(self.settings.device_mesh_shape)
            return create_pipeline(
                self.ttnn_device,
                weights_dir=self.settings.model_weights_path,
                height=resolution.height,
                width=resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Prodia T2V pipeline creation failed",
                e,
            )
            raise

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        request = requests[0]
        resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        frames = self.pipeline(
            prompt=request.prompt,
            height=resolution.height,
            width=resolution.width,
            num_frames=WAN22_NUM_FRAMES,
            seed=int(request.seed or 0),
            traced=True,
        )
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        if self.export_in_runner:
            from utils.video_manager import VideoManager

            return [VideoManager().export_to_mp4(frames)]
        return frames


class TTWan22I2VProdiaRunner(TTDiTRunner):
    """Wan2.2 I2V runner using the Prodia distilled pipeline.
    Single-image conditioning only — when the broadcast carries
    ``image_prompts`` with multiple entries, the prompt with the lowest
    ``frame_pos`` is selected and the rest are dropped (the distilled pipeline
    does not accept multi-frame conditioning).
    """

    requires_image_conditioning = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.image_manager = ImageManager("img")
        # Export MP4 inside the device worker by default to avoid pickling the
        # raw frame array (~226MB at 720p×81 frames) over IPC.
        self.export_in_runner = True

    def _build_warmup_video_request(self) -> VideoI2VGenerateRequest:
        """Synthetic 64x64 PIL warmup — same approach as TTWan22I2VRunner.

        The Prodia pipeline resizes to (height, width) before VAE encoding,
        so the input resolution is irrelevant; a small black frame exercises
        the same kernels as a real photo without paying the JPEG encode cost.
        """
        dummy = Image.new("RGB", (64, 64), color=0)
        buf = io.BytesIO()
        dummy.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return VideoI2VGenerateRequest.model_construct(
            prompt="Sunrise on a beach",
            negative_prompt="",
            num_inference_steps=2,
            image_prompts=[ImagePromptEntry(image=b64, frame_pos=0)],
        )

    def load_weights(self):
        return False

    def get_pipeline_device_params(self):
        return _prodia_wan_device_params(self.settings.device_mesh_shape)

    def create_pipeline(self):
        try:
            from pipelines.pipeline_i2v import (
                create_i2v_pipeline,
            )

            resolution = wan22_target_resolution(self.settings.device_mesh_shape)
            return create_i2v_pipeline(
                self.ttnn_device,
                weights_dir=self.settings.model_weights_path,
                height=resolution.height,
                width=resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Prodia I2V pipeline creation failed",
                e,
            )
            raise

    def _build_image_prompt(
        self, request: VideoI2VGenerateRequest, target_size: tuple[int, int]
    ) -> list:
        """Decode ``image_prompts`` into the (PIL, frame_pos) tuple list the
        Prodia pipeline expects for multi-frame conditioning.
        """
        return [
            (
                self.image_manager.base64_to_pil_image(
                    entry.image, target_size=target_size, target_mode="RGB"
                ),
                entry.frame_pos,
            )
            for entry in request.image_prompts
        ]

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoI2VGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        request = requests[0]
        resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        image_prompt = self._build_image_prompt(
            request, target_size=(resolution.width, resolution.height)
        )
        frames = self.pipeline(
            prompt=request.prompt,
            image=image_prompt,
            height=resolution.height,
            width=resolution.width,
            num_frames=WAN22_NUM_FRAMES,
            seed=int(request.seed or 0),
            traced=True,
        )
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        if self.export_in_runner:
            from utils.video_manager import VideoManager

            return [VideoManager().export_to_mp4(frames)]
        return frames


class TTWan22I2VRunner(TTDiTRunner):
    """
    Wan2.2 image-to-video runner.
    """

    requires_image_conditioning = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        self.image_manager = ImageManager()

    def create_pipeline(self):
        try:
            return WanPipelineI2V.create_pipeline(
                mesh_device=self.ttnn_device,
                height=self.resolution.height,
                width=self.resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Wan I2V pipeline creation failed",
                e,
            )
            raise

    def load_weights(self):
        return False

    def _build_image_prompt(self, request: VideoI2VGenerateRequest) -> list:
        """Decode base64 images into ``List[ImagePrompt]`` for the pipeline."""
        return [
            ImagePrompt(
                image=self.image_manager.base64_to_pil_image(entry.image),
                frame_pos=entry.frame_pos,
            )
            for entry in request.image_prompts
        ]

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoI2VGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        request = requests[0]
        pipeline_args = _wan22_pipeline_args(
            request,
            self.resolution,
            image_prompt=self._build_image_prompt(request),
        )
        frames = self.pipeline(**pipeline_args)
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    def get_pipeline_device_params(self):
        return _wan22_dit_device_params(self.settings.device_mesh_shape)

    def _build_warmup_video_request(self) -> VideoI2VGenerateRequest:
        """Warmup request with a synthetic 64x64 PIL so the VAE encoder has
        input to process.

        The I2V pipeline resizes the image to the target (height, width)
        before VAE encoding, so the input resolution is irrelevant — a
        small black frame exercises the same kernels as a real photo.
        """
        dummy = Image.new("RGB", (64, 64), color=0)
        buf = io.BytesIO()
        dummy.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return VideoI2VGenerateRequest.model_construct(
            prompt="Sunrise on a beach",
            negative_prompt="",
            num_inference_steps=2,
            image_prompts=[ImagePromptEntry(image=b64, frame_pos=0)],
        )


# ---------------------------------------------------------------------------
# Wan2.2 I2V experimental variants: AniSora, Distill (LightX2V), LoRA
# ---------------------------------------------------------------------------


def _wan22_i2v_warmup_request(prompt: str = "Sunrise on a beach"):
    """Shared warmup request builder for I2V experimental runners."""
    dummy = Image.new("RGB", (64, 64), color=0)
    buf = io.BytesIO()
    dummy.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return VideoI2VGenerateRequest.model_construct(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=2,
        image_prompts=[ImagePromptEntry(image=b64, frame_pos=0)],
    )


class TTWan22I2VAniSoraRunner(TTDiTRunner):
    """Wan2.2 I2V with AniSora V3.2 anime fine-tune weights."""

    requires_image_conditioning = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        self.image_manager = ImageManager()

    def create_pipeline(self):
        try:
            from models.tt_dit.experimental.pipelines.pipeline_anisora import (
                AniSoraPipeline,
            )

            # Enable the fast-image-encode path before the pipeline reads these
            # flags at build time. setdefault keeps any deployment-provided value.
            for flag, value in WAN_ANISORA_FAST_ENCODE_FLAGS.items():
                os.environ.setdefault(flag, value)
            self.logger.info(
                "AniSora fast-encode flags: "
                + ", ".join(
                    f"{flag}={os.environ.get(flag)}"
                    for flag in WAN_ANISORA_FAST_ENCODE_FLAGS
                )
            )

            return AniSoraPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                height=self.resolution.height,
                width=self.resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "AniSora I2V pipeline creation failed", e
            )
            raise

    def load_weights(self):
        return False

    def _build_image_prompt(self, request: VideoI2VGenerateRequest) -> list:
        return [
            ImagePrompt(
                image=self.image_manager.base64_to_pil_image(entry.image),
                frame_pos=entry.frame_pos,
            )
            for entry in request.image_prompts
        ]

    @log_execution_time(
        f"{dit_runner_log_map.get(get_settings().model_runner, 'AniSora')} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoI2VGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running AniSora inference")
        request = requests[0]
        pipeline_args = _wan22_pipeline_args(
            request,
            self.resolution,
            image_prompt=self._build_image_prompt(request),
        )
        # AniSora-specific: force 8 steps (ignore the client's num_inference_steps,
        # same as the distill forces 4) and use the model's real CFG (3.5 on both
        # experts) rather than the shared 4.0/3.0 default.
        pipeline_args["num_inference_steps"] = WAN22_ANISORA_NUM_STEPS
        pipeline_args["guidance_scale"] = WAN22_ANISORA_GUIDANCE_SCALE
        pipeline_args["guidance_scale_2"] = WAN22_ANISORA_GUIDANCE_SCALE
        frames = self.pipeline(**pipeline_args)
        self.logger.debug(f"Device {self.device_id}: AniSora inference completed")
        return frames

    def get_pipeline_device_params(self):
        # Start from the shared Wan2.2 fabric/trace defaults, then bump the trace
        # region for AniSora's fully-optimized 8-step trace (see constant above).
        device_params = _wan22_dit_device_params(self.settings.device_mesh_shape)
        if is_large_mesh(self.settings.device_mesh_shape) and is_blackhole():
            device_params["trace_region_size"] = WAN22_ANISORA_BH_TRACE_REGION_BYTES
        return device_params

    def _build_warmup_video_request(self) -> VideoI2VGenerateRequest:
        return _wan22_i2v_warmup_request("An anime girl smiling, soft lighting")


class TTWan22I2VDistillRunner(TTDiTRunner):
    """Wan2.2 I2V with LightX2V 4-step distilled weights.

    Distill bakes in classifier-free guidance, so ``guidance_scale`` is forced
    to 1.0 and ``num_inference_steps`` defaults to 4.
    """

    requires_image_conditioning = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        self.image_manager = ImageManager()

    def create_pipeline(self):
        try:
            from models.tt_dit.experimental.pipelines.pipeline_wan_distill import (
                WanDistillPipelineI2V,
            )

            # Enable the fast-image-encode path before the pipeline reads these
            # flags at build time. setdefault keeps any deployment-provided value.
            for flag, value in WAN_DISTILL_FAST_ENCODE_FLAGS.items():
                os.environ.setdefault(flag, value)
            self.logger.info(
                "Distill fast-encode flags: "
                + ", ".join(
                    f"{flag}={os.environ.get(flag)}"
                    for flag in WAN_DISTILL_FAST_ENCODE_FLAGS
                )
            )

            return WanDistillPipelineI2V.create_pipeline(
                mesh_device=self.ttnn_device,
                height=self.resolution.height,
                width=self.resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Distill I2V pipeline creation failed", e
            )
            raise

    def load_weights(self):
        return False

    def _build_image_prompt(self, request: VideoI2VGenerateRequest) -> list:
        return [
            ImagePrompt(
                image=self.image_manager.base64_to_pil_image(entry.image),
                frame_pos=entry.frame_pos,
            )
            for entry in request.image_prompts
        ]

    @log_execution_time(
        f"{dit_runner_log_map.get(get_settings().model_runner, 'Distill')} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoI2VGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running Distill inference")
        request = requests[0]
        seed = int(request.seed) if request.seed is not None else 0
        pipeline_args = {
            "prompts": [request.prompt],
            "num_inference_steps": WAN22_DISTILL_NUM_STEPS,
            "guidance_scale": 1.0,
            "guidance_scale_2": 1.0,
            "seed": seed,
            "traced": True,
            "image_prompt": self._build_image_prompt(request),
        }
        frames = self.pipeline(**pipeline_args)
        self.logger.debug(f"Device {self.device_id}: Distill inference completed")
        return frames

    def get_pipeline_device_params(self):
        # Start from the shared Wan2.2 fabric/trace defaults, then bump the trace
        # region for the distill's fully-optimized trace (see constant above).
        device_params = _wan22_dit_device_params(self.settings.device_mesh_shape)
        if is_large_mesh(self.settings.device_mesh_shape) and is_blackhole():
            device_params["trace_region_size"] = WAN22_DISTILL_BH_TRACE_REGION_BYTES
        return device_params

    def _build_warmup_video_request(self) -> VideoI2VGenerateRequest:
        return _wan22_i2v_warmup_request()


class TTWan22I2VLoRARunner(TTDiTRunner):
    """Wan2.2 I2V with LoRA adapter fusion (camera control, style, etc.).

    LoRA weights are resolved from ``LORA_HIGH_PATH`` / ``LORA_LOW_PATH``
    environment variables by the pipeline's ``__init__``.
    """

    requires_image_conditioning = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        self.image_manager = ImageManager()

    def create_pipeline(self):
        try:
            from models.tt_dit.experimental.pipelines.pipeline_wan_lora import (
                WanPipelineI2VLora,
            )

            lora_high = os.environ.get("LORA_HIGH_PATH")
            lora_low = os.environ.get("LORA_LOW_PATH")

            # create_pipeline can't forward LoRA stacks, so build the config and
            # construct directly.
            config = WanPipelineI2VLora.default_config(
                mesh_device=self.ttnn_device,
                height=self.resolution.height,
                width=self.resolution.width,
                num_frames=WAN22_NUM_FRAMES,
            )
            return WanPipelineI2VLora(
                device=self.ttnn_device,
                config=config,
                lora_high=lora_high,
                lora_low=lora_low,
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "LoRA I2V pipeline creation failed", e
            )
            raise

    def load_weights(self):
        return False

    def _build_image_prompt(self, request: VideoI2VGenerateRequest) -> list:
        return [
            ImagePrompt(
                image=self.image_manager.base64_to_pil_image(entry.image),
                frame_pos=entry.frame_pos,
            )
            for entry in request.image_prompts
        ]

    @log_execution_time(
        f"{dit_runner_log_map.get(get_settings().model_runner, 'LoRA')} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoI2VGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running LoRA inference")
        request = requests[0]
        pipeline_args = _wan22_pipeline_args(
            request,
            self.resolution,
            image_prompt=self._build_image_prompt(request),
        )
        frames = self.pipeline(**pipeline_args)
        self.logger.debug(f"Device {self.device_id}: LoRA inference completed")
        return frames

    def get_pipeline_device_params(self):
        return _wan22_dit_device_params(self.settings.device_mesh_shape)

    def _build_warmup_video_request(self) -> VideoI2VGenerateRequest:
        return _wan22_i2v_warmup_request("A golden retriever running on a sandy beach")


# ---------------------------------------------------------------------------
# LTX-2.3 distilled text->audio-video
# ---------------------------------------------------------------------------

# Proven-good 1080p ~6s AV generation shape for the (4, 8) Galaxy ring config
# (validated on-device). H/W must be %64 and (num_frames-1)%8 == 0.
LTX_NUM_FRAMES = 145
LTX_HEIGHT = 1088
LTX_WIDTH = 1920
LTX_FPS = 24
# (4, 8) BH Galaxy ring defaults (mirrors LTXPipeline.create_pipeline's own 4x8
# device_configs entry): dynamic_load off, Ring topology, 2 links.
LTX_DYNAMIC_LOAD = False
# Trace + reserve L1_SMALL for the traced two-stage decode / audio vocoder.
# Mirrors ``_ring_trace`` in models/tt_dit/tests/models/ltx/ltx_mesh_params.py:
# without l1_small_size the vocoder OOMs ("bank size is 0 B"); the two stage
# traces need the larger region.
LTX_TRACED = True
LTX_L1_SMALL_SIZE = 32768
LTX_TRACE_REGION_BYTES = 500_000_000

# Reuse the default TT_VIDEO_OUTPUT_DIR convention (VideoManager writes here too)
# so the served-file lifecycle in open_ai_api/video.py is identical to the frame
# runners: run() returns a filesystem path, VideoService.post_process passes a
# str straight through, and the API streams it back with FileResponse.
LTX_VIDEO_OUTPUT_DIR = Path(os.environ.get("TT_VIDEO_OUTPUT_DIR", "/tmp/videos"))


class TTLTX23DistilledRunner(TTDiTRunner):
    """LTX-2.3 distilled text->audio-video runner (Galaxy, ring topology).

    Unlike the Wan runners, the LTX distilled pipeline's ``generate()`` encodes
    the MP4 (h264 video + aac audio) itself and returns the on-disk path, so
    ``run()`` returns ``[path]`` rather than a raw frame array.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        # Set for the duration of warmup() so run() can tell the trace-capture
        # generation apart from a real request. See _discard_warmup_output.
        self._warming_up = False

    async def warmup(self) -> bool:
        self._warming_up = True
        try:
            return await super().warmup()
        finally:
            self._warming_up = False

    def _discard_warmup_output(self, path: str) -> None:
        """Delete the warmup generation's MP4.

        create_pipeline() compiles kernels untraced; the traces themselves are
        captured lazily on the first traced generate(), which is the run() that
        TTDiTRunner.warmup() issues. That gen exists only for its side effects --
        no job owns its output and VideoManager never reclaims it -- so without
        this, every server start would orphan a ~1080p MP4 in TT_VIDEO_OUTPUT_DIR.
        """
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as e:
            # Never fail warmup over cleanup; a leaked file is the lesser problem.
            self.logger.warning(
                f"Device {self.device_id}: could not remove warmup output {path}: {e}"
            )

    def create_pipeline(self):
        try:
            # Imported lazily (as the Prodia runners do) so the LTX pipeline and
            # its deps are not pulled in for every other runner in this module.
            from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import (
                LTXDistilledPipeline,
            )
            from models.tt_dit.utils.ltx import default_ltx_gemma

            return LTXDistilledPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
                gemma_path=default_ltx_gemma(),
                sp_axis=1,
                tp_axis=0,
                num_links=2,
                dynamic_load=LTX_DYNAMIC_LOAD,
                topology=ttnn.Topology.Ring,
                is_fsdp=False,
                traced=LTX_TRACED,
                num_frames=LTX_NUM_FRAMES,
                height=LTX_HEIGHT,
                width=LTX_WIDTH,
                image_conditioning=False,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "LTX-2.3 distilled pipeline creation failed",
                e,
            )
            raise

    def load_weights(self):
        return False

    @log_execution_time(
        f"{dit_runner_log_map.get(get_settings().model_runner, 'LTX-2.3-distilled')} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running LTX inference")
        request = requests[0]
        LTX_VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(LTX_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")
        # generate() writes the AV MP4 to output_path and returns that path.
        result_path = self.pipeline.generate(
            request.prompt,
            output_path=output_path,
            num_frames=LTX_NUM_FRAMES,
            height=LTX_HEIGHT,
            width=LTX_WIDTH,
            seed=int(request.seed or 0),
            fps=LTX_FPS,
        )
        self.logger.debug(f"Device {self.device_id}: LTX inference completed")
        if self._warming_up:
            self._discard_warmup_output(result_path)
        return [result_path]

    def get_pipeline_device_params(self):
        # Start from the shared Wan2.2 ring/fabric defaults (FABRIC_1D_RING +
        # RELAXED_INIT + 8k router payload for the BH Galaxy mesh), then apply the
        # LTX-specific L1_SMALL reservation and the larger traced-decode region.
        device_params = _wan22_dit_device_params(self.settings.device_mesh_shape)
        device_params["l1_small_size"] = LTX_L1_SMALL_SIZE
        if LTX_TRACED:
            device_params["trace_region_size"] = LTX_TRACE_REGION_BYTES
        return device_params


class TTWan22I2VLightningRunner(TTDiTRunner):
    """Wan2.2 I2V with lightx2v/Wan2.2-Lightning distilled LoRA weights."""

    requires_image_conditioning = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)
        self.image_manager = ImageManager()

    def create_pipeline(self):
        try:
            from models.tt_dit.experimental.pipelines.pipeline_wan_lora import (
                WanPipelineI2VLora,
            )

            lora_high = os.environ.get("LORA_HIGH_PATH") or hf_hub_download(
                repo_id="lightx2v/Wan2.2-Lightning",
                filename="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
            )
            lora_low = os.environ.get("LORA_LOW_PATH") or hf_hub_download(
                repo_id="lightx2v/Wan2.2-Lightning",
                filename="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
            )
            self.logger.info(
                f"Device {self.device_id}: Lightning adapters "
                f"high={lora_high}, low={lora_low}"
            )

            config = WanPipelineI2VLora.default_config(
                mesh_device=self.ttnn_device,
                height=self.resolution.height,
                width=self.resolution.width,
                num_frames=WAN22_NUM_FRAMES,
                cfg_enabled=False,
                config_overrides={"boundary_ratio": WAN22_LIGHTNING_BOUNDARY_RATIO},
            )
            return WanPipelineI2VLora(
                device=self.ttnn_device,
                config=config,
                lora_high=lora_high,
                lora_low=lora_low,
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Lightning I2V pipeline creation failed", e
            )
            raise

    def load_weights(self):
        return False

    def _build_image_prompt(self, request: VideoI2VGenerateRequest) -> list:
        return [
            ImagePrompt(
                image=self.image_manager.base64_to_pil_image(entry.image),
                frame_pos=entry.frame_pos,
            )
            for entry in request.image_prompts
        ]

    @log_execution_time(
        f"{dit_runner_log_map.get(get_settings().model_runner, 'Lightning')} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoI2VGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running Lightning inference")
        request = requests[0]
        seed = int(request.seed) if request.seed is not None else 0
        pipeline_args = {
            "prompts": [request.prompt],
            "num_inference_steps": WAN22_LIGHTNING_NUM_STEPS,
            "guidance_scale": 1.0,
            "guidance_scale_2": 1.0,
            "seed": seed,
            "traced": True,
            "image_prompt": self._build_image_prompt(request),
        }
        frames = self.pipeline(**pipeline_args)
        self.logger.debug(f"Device {self.device_id}: Lightning inference completed")
        return frames

    def get_pipeline_device_params(self):
        return _wan22_dit_device_params(self.settings.device_mesh_shape)

    def _build_warmup_video_request(self) -> VideoI2VGenerateRequest:
        return _wan22_i2v_warmup_request()


# The prompt the pipeline is warmed with. Its *token count* is the load-bearing part, not its
# content: every program in the 50-block stack is keyed on the padded packed length, which the
# prompt length feeds into, so warming at a two-word prompt and serving hundred-token ones can
# warm nothing. Roughly 100 tokens is representative of a real request.
MINIMAX_H3_WARMUP_PASSES = 1

MINIMAX_H3_WARMUP_PROMPT = (
    "A red fox steps through wet grass at dawn, breath fogging in the cold air, while the camera "
    "tracks slowly alongside. Birdsong rises in the background and the fox pauses, ears turning "
    "toward a rustle in the undergrowth, before trotting on through the low golden light."
)

# Reserved only for shapes whose preset enables `trace_denoise` (4x32); matches the model tests'
# `_ring_8k_trace`.
MINIMAX_H3_TRACE_REGION_BYTES = 150_000_000


def _minimax_h3_device_params(mesh_shape: tuple) -> dict:
    """Device params for MiniMax-H3 t2va, keyed on the mesh shape.

    Everything here is derived from the pipeline's per-shape preset so the two can't disagree:
    fabric follows topology (Ring -> FABRIC_1D_RING), and a `trace_region_size` is reserved when the
    preset enables `trace_denoise` (the 4x32 quad) -- without it the pipeline's trace capture is
    fatal. The region is only reserved, so a shape that does not trace pays nothing but address
    space. `l1_small_size` is mandatory (a bare open fails as "bank size is 0 B").
    """
    preset = resolve_mesh_preset(mesh_shape)
    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = 8192
    ring = preset["topology"] == ttnn.Topology.Ring
    params = {
        "fabric_config": (
            ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D
        ),
        "fabric_router_config": router_config,
        "l1_small_size": 65536,
    }
    if preset.get("trace_denoise"):
        params["trace_region_size"] = MINIMAX_H3_TRACE_REGION_BYTES
    return params


class TTMiniMaxH3Runner(TTDiTRunner):
    """MiniMax-H3 `t2va`: text in, a video **and its soundtrack** out.

    Two things make this runner differ from the Wan2.2 one it is otherwise modelled on, and both
    are silent-failure modes rather than crashes:

    1. **Warmup has to be the real shape at a realistic prompt length.** The base class warms with
       a throwaway two-word prompt at `num_inference_steps=2`. For H3 that warms nothing useful --
       the program cache is keyed on the padded packed length, and the AdaLN modulation table is
       cached per *step count*, so a 2-step warm builds a schedule no request will ever use. The
       cost of getting it wrong is ~210 s per request instead of ~73 s, with nothing in the logs
       saying so. `warmup` is overridden and the padded length is asserted.

    2. **The audio has to reach the client.** `t2va` returns a soundtrack alongside the video and
       the delivered MP4 has to carry both; a silent track is a bug. The muxing happens here and
       the runner returns a *path*, which `VideoService.post_process` passes straight through.

    One shape only in v1, validated at the boundary. A request at an unwarmed shape would compile
    inside the request rather than fail, which is worse than a clear error.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        # Mux a/v inline by default; video_runner flips this so its encoder
        # thread muxes off the hot path.
        self.export_in_runner = True

    def _weights_dir(self) -> str | None:
        """A local snapshot directory, or None to let the pipeline read its own env var.

        `settings.model_weights_path` defaults to the **HuggingFace repo id**
        (`MiniMaxAI/MiniMax-H3`), not a path, so passing it through unconditionally would send the
        pipeline looking for `MiniMaxAI/MiniMax-H3/transformer/config.json`. Only an actual
        directory is used; otherwise `MINIMAX_H3_DIFFUSERS_DIR` decides, which is what the model's
        own tests and docs use. The weights are ~62 GB per transformer partition and are always
        mounted, never baked into the image.
        """
        configured = self.settings.model_weights_path
        if configured and os.path.isdir(configured):
            return configured
        if configured:
            self.logger.info(
                f"Device {self.device_id}: model_weights_path={configured!r} is not a directory "
                "(it is the HF repo id); falling back to MINIMAX_H3_DIFFUSERS_DIR"
            )
        return None

    def create_pipeline(self):
        try:
            # No topology arg: the pipeline resolves it from its per-shape preset.
            return MiniMaxH3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                weights_dir=self._weights_dir(),
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "MiniMax-H3 pipeline creation failed",
                e,
            )
            raise

    def load_weights(self):
        return False

    def get_pipeline_device_params(self):
        return _minimax_h3_device_params(self.settings.device_mesh_shape)

    async def warmup(self) -> bool:
        """Build the pipeline, then warm it at the exact shape and step count that will be served.

        Deliberately not the base class's implementation: this calls the pipeline's own `warmup`,
        which runs one full generation at the target working point, rather than a 2-step throwaway.
        It is slow (minutes) and that is why the readiness probe's delay is generous -- readiness
        must mean *warm*, or the first real request pays the compile.
        """
        self.logger.info(f"Device {self.device_id}: Loading MiniMax-H3...")

        def distribute_block():
            self.pipeline = self.create_pipeline()

        weights_distribution_timeout = max(
            self.settings.weights_distribution_timeout_seconds,
            DIT_WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS,
        )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(distribute_block),
                timeout=weights_distribution_timeout,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                f"Device {self.device_id}: pipeline construction timed out after "
                f"{weights_distribution_timeout} seconds"
            )
            raise
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Exception during model loading", e
            )
            raise

        # Nothing is warmed by default. All 18 published working points are servable, and warming
        # them costs 4-16 min each (measured, worst at the 15 s / 1 MPix points), so an eager
        # warmup would trade hours of startup for a latency win on whichever shapes it guessed.
        # Instead the first request at a given shape compiles, once, and says so in the log.
        #
        #   MINIMAX_H3_WARM_SHAPES unset  -> warm nothing (default).
        #   "all"                         -> all 6 ratios x 3 durations. Hours of startup.
        #   "16:9@5,9:16@10,..."          -> an explicit subset, for shapes worth pre-paying.
        shapes = self._warmup_shapes()
        if shapes:
            self.logger.info(
                f"Device {self.device_id}: Model loaded, warming {len(shapes)} shape(s): "
                + ", ".join(f"{w}x{h}/{f}f" for h, w, f in shapes)
            )
        else:
            self.logger.info(
                f"Device {self.device_id}: Model loaded, warming nothing "
                "(MINIMAX_H3_WARM_SHAPES unset). The first request at each shape compiles."
            )
        # The first served request pays ~12 s on its first denoise step against a 1.05 s steady
        # step, then settles to 1.1 s. A second warmup pass does NOT absorb it (measured: still
        # 11.9 s after two passes), so this stays at 1 until the cause is found.
        self._warm_padded_lens = set()
        for height, width, num_frames in shapes:
            for pass_index in range(MINIMAX_H3_WARMUP_PASSES):
                await asyncio.to_thread(
                    lambda h=height, w=width, f=num_frames: self.pipeline.warmup(
                        prompt=MINIMAX_H3_WARMUP_PROMPT,
                        num_frames=f,
                        height=h,
                        width=w,
                        num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
                    )
                )
                self.logger.info(
                    f"Device {self.device_id}: {width}x{height}/{num_frames}f warmup pass "
                    f"{pass_index + 1}/{MINIMAX_H3_WARMUP_PASSES} done"
                )
            self._warm_padded_lens.add(self.pipeline.last_padded_len)
            self.logger.info(
                f"Device {self.device_id}: warm at padded_len={self.pipeline.last_padded_len} "
                f"({width}x{height}, {num_frames} frames, {MINIMAX_H3_NUM_INFERENCE_STEPS} steps)"
            )
        return True

    def _warmup_shapes(self) -> list[tuple[int, int, int]]:
        """`(height, width, num_frames)` per shape to warm, from MINIMAX_H3_WARM_SHAPES."""
        from models.tt_dit.pipelines.minimax_h3.packing import (
            MINIMAX_H3_FPS,
            align_num_frames,
            resolve_canvas_size,
        )

        def shape(ratio, seconds):
            height, width = resolve_canvas_size(*ratio)
            return height, width, align_num_frames(round(seconds * MINIMAX_H3_FPS))

        spec = (os.environ.get("MINIMAX_H3_WARM_SHAPES") or "").strip()
        if not spec:
            return []
        if spec.lower() == "all":
            return [
                shape(ratio, seconds)
                for ratio in MINIMAX_H3_ASPECT_RATIOS
                for seconds in MINIMAX_H3_DURATIONS_S
            ]

        shapes: list[tuple[int, int, int]] = []
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            ratio_text, _, seconds_text = entry.partition("@")
            ratio = minimax_h3_parse_aspect_ratio(ratio_text)
            seconds = (
                int(seconds_text) if seconds_text else MINIMAX_H3_DEFAULT_DURATION_S
            )
            if seconds not in MINIMAX_H3_DURATIONS_S:
                raise ValueError(
                    f"MINIMAX_H3_WARM_SHAPES entry {entry!r}: duration must be one of "
                    f"{', '.join(str(d) for d in MINIMAX_H3_DURATIONS_S)}"
                )
            candidate = shape(ratio, seconds)
            if candidate not in shapes:
                shapes.append(candidate)
        if not shapes:
            raise ValueError("MINIMAX_H3_WARM_SHAPES was set but parsed to no shapes")
        return shapes

    def _resolve_shape(self, request: VideoGenerateRequest) -> tuple[int, int, int]:
        """`(height, width, num_frames)` for this request, or raise with what is served.

        Both levers are validated against the published set and then handed to the model's own
        `resolve_canvas_size` / `align_num_frames`, so the canvas and frame rules live in exactly
        one place. Nothing here rounds: an unsupported ratio or duration is refused, because
        quietly serving a neighbouring shape returns a video the caller did not ask for.
        """
        from models.tt_dit.pipelines.minimax_h3.packing import (
            MINIMAX_H3_FPS,
            align_num_frames,
            resolve_canvas_size,
        )

        ratio = (
            minimax_h3_parse_aspect_ratio(request.aspect_ratio)
            if getattr(request, "aspect_ratio", None) is not None
            else MINIMAX_H3_DEFAULT_ASPECT_RATIO
        )

        seconds = getattr(request, "duration_seconds", None)
        if seconds is None:
            seconds = MINIMAX_H3_DEFAULT_DURATION_S
        elif seconds not in MINIMAX_H3_DURATIONS_S:
            raise ValueError(
                f"duration_seconds must be an integer from {min(MINIMAX_H3_DURATIONS_S)} to "
                f"{max(MINIMAX_H3_DURATIONS_S)}; got {seconds}"
            )

        height, width = resolve_canvas_size(*ratio)
        num_frames = align_num_frames(round(seconds * MINIMAX_H3_FPS))
        if not minimax_h3_frames_are_aligned(num_frames):
            # Unreachable via the duration allow-list; kept so a future edit to it cannot smuggle
            # a frame count the VAE's 17-frame chunking would reject deep inside packing.
            raise ValueError(
                f"num_frames must be 17n + 5; {seconds} s resolved to {num_frames}"
            )
        return height, width, num_frames

    def _validate(self, request: VideoGenerateRequest) -> None:
        """Reject shapes outside the published aspect-ratio/duration set before the request runs."""
        self._resolve_shape(request)

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoGenerateRequest]):
        from utils.video_manager import VideoAudioResult, VideoManager

        request = requests[0]
        self._validate(request)
        height, width, num_frames = self._resolve_shape(request)
        self.logger.debug(
            f"Device {self.device_id}: Running inference at {width}x{height}, {num_frames} frames"
        )

        output = self.pipeline(
            request.prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=request.num_inference_steps
            or MINIMAX_H3_NUM_INFERENCE_STEPS,
            seed=int(request.seed) if request.seed is not None else 0,
        )

        # `_warm_padded_lens` is what is known resident: seeded by warmup (possibly empty) and
        # extended as shapes are served, so this fires once per shape -- when compilation actually
        # happened -- rather than on every request of an un-pre-warmed deployment.
        warm = getattr(self, "_warm_padded_lens", None)
        served = self.pipeline.last_padded_len
        if warm is not None and served not in warm:
            # Not fatal -- the video is fine -- but this request compiled rather than replayed, and
            # the latency looks inexplicable unless it is said out loud. Recorded afterwards so the
            # next request at this shape is quiet.
            self.logger.warning(
                f"Device {self.device_id}: padded_len {served} was not resident "
                f"(resident: {sorted(warm) or 'none'}); this request paid compilation. "
                "Pre-pay it with MINIMAX_H3_WARM_SHAPES if this shape is served often."
            )
            warm.add(served)

        self.logger.debug(f"Device {self.device_id}: Inference completed")
        # (1, 3, F, H, W) in [0, 1] -> (F, H, W, 3), which is what the exporter's rawvideo pipe
        # wants. Without the permute it reads the width as a channel count and raises.
        frames = output.video[0].permute(1, 2, 3, 0).contiguous().numpy()
        audio = output.audio[0].numpy()

        # video_runner set export_in_runner=False: hand the raw a/v to its encoder thread to mux.
        if not self.export_in_runner:
            return VideoAudioResult(frames, audio, output.sampling_rate, output.fps)

        path = VideoManager().export_to_mp4_with_audio(
            frames, audio, output.sampling_rate, fps=output.fps
        )
        # A **list**, one entry per request in the batch -- `base_service.py:40` and
        # `device_worker.py:115` both do `results[0]`. Returning the bare path string is not a type
        # error anywhere; it just gets indexed, and the job's result path becomes "/" -- the first
        # character. Same shape as the Prodia runner's `return [VideoManager().export_to_mp4(...)]`.
        return [path]
