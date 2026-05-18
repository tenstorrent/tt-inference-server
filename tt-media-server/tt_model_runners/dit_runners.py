# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import base64
import io
import os
from abc import abstractmethod

import ttnn
from PIL import Image
from config.constants import (
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
from models.common.utility_functions import is_blackhole
from models.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
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
    ModelRunners.TT_WAN_2_2_I2V.value: "Wan22-I2V",
    ModelRunners.TT_WAN_2_2_I2V_PRODIA.value: "Wan22-I2V-Prodia",
    ModelRunners.TT_QWEN_IMAGE.value: "Qwen-Image",
    ModelRunners.TT_QWEN_IMAGE_2512.value: "Qwen-Image-2512",
    ModelRunners.SP_RUNNER.value: "SP-Runner",
}

DIT_WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS = 6000


class TTDiTRunner(BaseMetalDeviceRunner):
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
        return {"l1_small_size": 32768, "trace_region_size": 50000000}


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
        # setup environment for Mochi runner
        os.environ["TT_DIT_CACHE_DIR"] = "/tmp/TT_DIT_CACHE"

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
        frames = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=3.5,
            num_frames=168,  # TODO: Parameterize output dimensions.
            height=480,
            width=848,
            output_type="np",
            seed=int(request.seed or 0),
        )
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    def get_pipeline_device_params(self):
        return {}


WAN22_BH_RING_MESH_SHAPES = frozenset({(1, 4)})

WAN22_GALAXY_BH_TRACE_REGION_BYTES = 120_000_000
WAN22_GALAXY_ROUTER_MAX_PAYLOAD_BYTES = 8192


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
    resolution,
    image_prompt=None,
):
    """Build the kwargs dict shared by Wan2.2 T2V and I2V ``__call__`` sites."""
    seed = int(request.seed) if request.seed is not None else None
    pipeline_args = {
        "prompt": request.prompt,
        "height": resolution.height,
        "width": resolution.width,
        "num_frames": WAN22_NUM_FRAMES,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "seed": seed,
        "traced": True,
    }
    if image_prompt is not None:
        pipeline_args["image_prompt"] = image_prompt
    # Only include negative_prompt when set; otherwise the pipeline default applies.
    if bool(request.negative_prompt):
        pipeline_args["negative_prompt"] = request.negative_prompt
    return pipeline_args


class TTWan22Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.resolution = wan22_target_resolution(self.settings.device_mesh_shape)

    def create_pipeline(self):
        try:
            return WanPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
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


class TTWan22I2VProdiaRunner(TTDiTRunner):
    """Wan2.2 I2V runner using the Prodia distilled pipeline.
    Single-image conditioning only — when the broadcast carries
    ``image_prompts`` with multiple entries, the prompt with the lowest
    ``frame_pos`` is selected and the rest are dropped (the distilled pipeline
    does not accept multi-frame conditioning).
    """

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
        # The 4x8 LoudBox trace binary needs ~30.6MB; the default 30MB region
        # rejects it and warmup OOMs. Both 4x8 (32 chips) and 4x32 (128 chips)
        # Blackhole meshes get the bumped trace region.
        device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
        mesh_size = (
            self.settings.device_mesh_shape[0] * self.settings.device_mesh_shape[1]
        )
        if mesh_size >= 32 and is_blackhole():
            device_params["trace_region_size"] = 120_000_000
            config = ttnn.FabricRouterConfig()
            config.max_packet_payload_size_bytes = 8192
            device_params["fabric_router_config"] = config
        return device_params

    def create_pipeline(self):
        try:
            from models.tt_dit.prodia.pipelines.pipeline_i2v import (
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
