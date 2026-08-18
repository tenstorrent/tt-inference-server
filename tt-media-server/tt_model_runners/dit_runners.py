# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import os
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import ttnn
from config.constants import (
    MINIMAX_H3_NUM_FRAMES,
    WAN22_NUM_FRAMES,
    ModelRunners,
    ModelServices,
    SupportedModels,
    is_large_mesh,
    minimax_h3_target_resolution,
    wan22_target_resolution,
)
from config.settings import get_settings
from domain.image_generate_request import ImageGenerateRequest
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import ImagePromptEntry, VideoI2VGenerateRequest
from huggingface_hub import hf_hub_download
from models.common.utility_functions import is_blackhole
from models.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
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
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.image_manager import ImageManager
from utils.logger import log_exception_chain

dit_runner_log_map = {
    ModelRunners.TT_SD3_5.value: "SD35",
    ModelRunners.TT_FLUX_1_DEV.value: "FLUX.1-dev",
    ModelRunners.TT_FLUX_1_SCHNELL.value: "FLUX.1-schnell",
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
    ModelRunners.TT_QWEN_IMAGE.value: "Qwen-Image",
    ModelRunners.TT_QWEN_IMAGE_2512.value: "Qwen-Image-2512",
    ModelRunners.TT_MINIMAX_H3.value: "MiniMax-H3",
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
        image = self.pipeline.run_single_prompt(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            seed=int(request.seed or 0),
        )
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
# Fixed step count (mirrors the distill forcing 4): AniSora always runs 8 steps,
# the validated good-quality / low-latency point (~9.3s traced). The client's
# num_inference_steps is ignored, same as the distill runner.
WAN22_ANISORA_NUM_STEPS = 8

WAN22_LIGHTNING_NUM_STEPS = 4
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
            "num_inference_steps": 4,
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


# ---------------------------------------------------------------------------
# MiniMax-H3 (text -> video + native stereo audio)
#
# H3 is the first video model here that also emits audio, so its runner returns
# both streams (VideoWithAudio) and the exporter muxes the audio into the mp4
# (utils/video_manager.export_to_mp4). Everything else mirrors the Wan/Mochi
# DiT runner pattern and drives the upstream tt_dit MiniMax-H3 pipeline.
# ---------------------------------------------------------------------------


@dataclass
class VideoWithAudio:
    """Runner result carrying both streams.

    ``export_to_mp4`` already special-cases an object exposing ``.frames``;
    returning this (instead of a bare frame array) is how the audio reaches the
    muxer without changing the video worker/service call sites.
    """

    frames: np.ndarray  # (F, H, W, 3) uint8   -- from MiniMaxH3Output.video
    audio: np.ndarray  # (2, N) float32 [-1, 1] -- from MiniMaxH3Output.audio (stereo)
    sample_rate: int  # MiniMaxH3Output.sampling_rate (e.g. 32000)
    fps: int = 24  # MiniMaxH3Output.fps -- used by export_to_mp4 for the video rate


# Explicit parallel params for the small Blackhole meshes upstream does not yet
# preset (its _PRESETS_BH covers only (4, 8) galaxy and (4, 32) quad). Values
# from a validated 1x4 Blackhole run; (2, 2) mirrors the same factors and is
# still to be validated on hardware. Remove an entry once it lands as a real
# upstream _PRESETS_BH row and create_pipeline can resolve it from the mesh.
# tp_axis=1 puts TP across the 4-chip axis (mirrors the galaxy preset's TP=4 on
# its 4-chip axis 0); sp_axis is the size-1/size-2 axis. topology MUST be non-None
# -- upstream treats any None in (tp_axis, sp_axis, num_links, topology) as "use a
# preset" and rejects the shape. A single QB2 is a physical line, so Linear (not
# the galaxy's Ring, which needs a torus wrap link the box does not have).
_MINIMAX_H3_SMALL_MESH_PARAMS = {
    (1, 4): dict(tp_axis=1, sp_axis=0, num_links=2, topology=ttnn.Topology.Linear),
    (2, 2): dict(tp_axis=1, sp_axis=0, num_links=2, topology=ttnn.Topology.Linear),
}


def _minimax_h3_dit_device_params(mesh_shape: tuple) -> dict:
    """Fabric / trace-region policy for MiniMax-H3, mirroring the Wan2.2 helper.

    Blackhole uses RELAXED_INIT; large (galaxy-class) meshes get the bigger
    trace region and the galaxy router config. Small meshes use plain FABRIC_1D.
    """
    device_params = _wan22_dit_device_params(mesh_shape)
    if is_large_mesh(mesh_shape) and is_blackhole():
        # H3's DiT trace is comparable to Wan2.2's; reuse the galaxy bump.
        device_params["trace_region_size"] = WAN22_GALAXY_BH_TRACE_REGION_BYTES
    return device_params


def _minimax_h3_pipeline_args(request: VideoGenerateRequest, resolution) -> dict:
    """Map an OpenAI-style VideoGenerateRequest onto the upstream __call__.

    H3 is guidance-distilled (cfg=1), so there is no guidance_scale. An I2V/fl2va
    variant would additionally decode request.image_prompts into ``image=`` /
    ``last_image=`` (first-frame conditioning).
    """
    seed = int(request.seed) if request.seed is not None else 0

    # Output geometry: an explicit request overrides the mesh default, clamped to the H3 envelope
    # (short edge <= 768, long edge <= 1344, snapped to /32; duration on the 17n+5 grid, <= ~15 s).
    def _snap32(x):
        return max(32, int(round(float(x) / 32.0)) * 32)

    def _snap_len(n):
        n = max(22, int(n))  # 17n+5 grid, n>=1 -> 22 frames minimum
        k = max(1, round((n - 5) / 17.0))
        # Hard duration ceiling. On the 1x4, clips beyond ~5 s (124f) OOM unreliably: the VAE decode
        # buffer scales with FRAME COUNT (not resolution), so downscaling the picture doesn't save a
        # 10 s clip -- it needs a ~79 MB contiguous DRAM block the fragmented free space can't provide.
        # Default 124 (~5.2 s) is the proven-robust max; H3_MAX_FRAMES raises it on a roomier mesh /
        # after bf8 frees DRAM. Legacy allowed ~15 s; that needs more DRAM headroom, not a bigger cap.
        cap = int(os.environ.get("H3_MAX_FRAMES", "124"))
        return min(17 * k + 5, cap)

    w = _snap32(request.width) if getattr(request, "width", None) else resolution.width
    h = _snap32(request.height) if getattr(request, "height", None) else resolution.height
    lo, hi = min(w, h), max(w, h)
    lo, hi = min(lo, 768), min(hi, 1344)
    w, h = (lo, hi) if w <= h else (hi, lo)

    nf_default = int(os.environ.get("H3_NUM_FRAMES", MINIMAX_H3_NUM_FRAMES))
    num_frames = _snap_len(request.num_frames) if getattr(request, "num_frames", None) else nf_default

    # Cap pixel*frame load and downscale resolution to fit (keeping duration), like the legacy
    # max_load. On the 1x4 this is set by DRAM *fragmentation*: a large (~63M, 10s@512) decode
    # fragments the free space so the NEXT large clip OOMs; clips in the ~42M zone run back-to-back
    # indefinitely. So 5s stays full-res, 10s renders ~415^2. Raise H3_MAX_LOAD on a roomier mesh.
    max_load = int(os.environ.get("H3_MAX_LOAD", "42000000"))
    load = w * h * num_frames
    if load > max_load:
        scale = (max_load / load) ** 0.5
        w = max(256, int(w * scale) // 32 * 32)
        h = max(256, int(h * scale) // 32 * 32)

    return dict(
        prompt=request.prompt,
        num_inference_steps=request.num_inference_steps,
        num_frames=num_frames,
        height=h,
        width=w,
        seed=seed,
    )


# Per-step progress: the media-server job API is coarse and the upstream pipeline exposes no
# callback, but its denoise loop calls module-level build_row_timesteps once per step. Wrap it to
# publish {phase, step, total} to a file keyed by the job id (the bot reads it for a live bar).
_H3_PROGRESS_DIR = os.environ.get("H3_PROGRESS_DIR", "/tmp/h3-progress")
_h3_progress = {"key": None, "step": 0, "total": 0}


def _h3_write_progress(phase: str) -> None:
    key = _h3_progress["key"]
    if not key:
        return
    try:
        import json

        os.makedirs(_H3_PROGRESS_DIR, exist_ok=True)
        path = os.path.join(_H3_PROGRESS_DIR, f"{key}.json")
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            f.write(
                json.dumps(
                    {"phase": phase, "step": _h3_progress["step"], "total": _h3_progress["total"]}
                )
            )
        os.replace(tmp, path)
    except Exception:
        pass


def _h3_install_progress_hook() -> None:
    from models.tt_dit.pipelines.minimax_h3 import pipeline_minimax_h3 as _pm

    if getattr(_pm, "_brt_progress_wrapped", False):
        return
    _orig = _pm.build_row_timesteps

    def _wrapped(*args, **kwargs):
        _h3_progress["step"] += 1
        _h3_write_progress("denoising")
        return _orig(*args, **kwargs)

    _pm.build_row_timesteps = _wrapped
    _pm._brt_progress_wrapped = True


_h3_install_progress_hook()


class TTMiniMaxH3Runner(TTDiTRunner):
    """MiniMax-H3 on the upstream pipeline. One runner serves both t2va (text -> video+audio) and
    fl2va (keyframe -> video+audio): /generations/i2v routes here with image_prompts, which run()
    decodes into the pipeline's image=/last_image=. (Kept out of I2V_MODEL_RUNNERS so t2v isn't
    rejected.)"""

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = minimax_h3_target_resolution(self.settings.device_mesh_shape)
        self.image_manager = ImageManager()

    def create_pipeline(self):
        mesh = tuple(self.settings.device_mesh_shape)
        # {} on (4, 8) / (4, 32): let the upstream _PRESETS_BH pick the shape.
        overrides = _MINIMAX_H3_SMALL_MESH_PARAMS.get(mesh, {})
        try:
            pipe = MiniMaxH3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                weights_dir=self.settings.model_weights_path,
                task="t2va",
                **overrides,
            )
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "MiniMax-H3 pipeline creation failed",
                e,
            )
            raise
        # Upstream auto-enables denoise trace-capture only for the quad galaxy; enable it for our
        # small mesh too (eager denoise is host-dispatch-bound). First gen per (shape, steps) captures.
        if getattr(pipe, "trace_denoise", None) is False and not is_large_mesh(mesh):
            pipe.trace_denoise = True
            self.logger.info(
                f"Device {self.device_id}: enabled denoise trace-capture for mesh {mesh}"
            )
        return pipe

    def load_weights(self):
        return False  # weights load during pipeline creation (as Wan/Mochi)

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        req = requests[0]
        # Arm per-step progress for this job (key == API job id == request._task_id).
        _h3_progress["key"] = getattr(req, "_task_id", None)
        _h3_progress["step"] = 0
        _h3_progress["total"] = int(getattr(req, "num_inference_steps", None) or 8)
        _h3_write_progress("starting")

        args = _minimax_h3_pipeline_args(req, self.resolution)
        # fl2va: /generations/i2v routes here with image_prompts; decode the first (and optional
        # last) keyframe into image=/last_image=. Absent -> plain t2va.
        image_prompts = getattr(req, "image_prompts", None)
        if image_prompts:
            tgt = (args["width"], args["height"])
            entries = sorted(image_prompts, key=lambda e: e.frame_pos)
            args["image"] = self.image_manager.base64_to_pil_image(
                entries[0].image, target_size=tgt, target_mode="RGB"
            )
            if len(entries) > 1 and entries[-1].frame_pos >= args["num_frames"] - 1:
                args["last_image"] = self.image_manager.base64_to_pil_image(
                    entries[-1].image, target_size=tgt, target_mode="RGB"
                )

        try:
            out = self.pipeline(**args)
            _h3_progress["step"] = _h3_progress["total"]
            _h3_write_progress("decoding")
        finally:
            self.logger.debug(f"Device {self.device_id}: Inference completed")
        # MiniMaxH3Output.video is (1, 3, F, H, W) float in [0, 1] -> (F, H, W, 3) uint8; audio is
        # (1, 2, samples) -> (2, samples). Return both so the exporter muxes the audio into the mp4.
        frames = (
            out.video[0]
            .permute(1, 2, 3, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to("cpu")
            .numpy()
            .astype(np.uint8)
        )
        audio = out.audio[0].to("cpu").numpy()
        # Free per-shape device state so a different-shape next job doesn't OOM the tight mesh.
        self._release_per_shape_state()
        # Device worker indexes the return per-request (responses[i]) and checks len(responses),
        # so return a one-element sequence (batch=1), like the Wan/Mochi runners.
        return [
            VideoWithAudio(
                frames=frames,
                audio=audio,
                sample_rate=out.sampling_rate,
                fps=out.fps,
            )
        ]

    def _release_per_shape_state(self) -> None:
        """Drop per-(H, W, frames) device buffers between renders so varying geometry doesn't OOM.

        Best-effort: the denoise trace (re-captured cheaply) and the VAE's per-shape decoder cache
        (rebuilt in ~1 s), then force reclaim. Resident text/DiT weights are untouched.
        """
        pipe = self.pipeline
        try:
            pipe.release_traces()
        except Exception as e:
            self.logger.debug(f"Device {self.device_id}: release_traces skipped: {e}")
        try:
            vae = getattr(pipe, "_vae", None)
            decoders = getattr(vae, "_decoders", None)
            if decoders:
                decoders.clear()
        except Exception as e:
            self.logger.debug(f"Device {self.device_id}: vae decoder clear skipped: {e}")
        try:
            import gc

            gc.collect()
            ttnn.synchronize_device(self.ttnn_device)
        except Exception as e:
            self.logger.debug(f"Device {self.device_id}: reclaim skipped: {e}")

    def get_pipeline_device_params(self):
        return _minimax_h3_dit_device_params(self.settings.device_mesh_shape)
