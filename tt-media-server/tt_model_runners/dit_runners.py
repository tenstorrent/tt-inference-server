# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from abc import abstractmethod

import ttnn
from config.constants import ModelRunners, ModelServices, SupportedModels
from config.settings import get_settings
from domain.image_generate_request import ImageGenerateRequest
from domain.video_generate_request import VideoGenerateRequest
from models.experimental.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from models.experimental.tt_dit.pipelines.mochi.pipeline_mochi import MochiPipeline
from models.experimental.tt_dit.pipelines.motif.pipeline_motif import MotifPipeline
from models.experimental.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
    QwenImagePipeline,
)
from models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
)
from models.experimental.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time

dit_runner_log_map = {
    ModelRunners.TT_SD3_5.value: "SD35",
    ModelRunners.TT_FLUX_1_DEV.value: "FLUX.1-dev",
    ModelRunners.TT_FLUX_1_SCHNELL.value: "FLUX.1-schnell",
    ModelRunners.TT_MOTIF_IMAGE_6B_PREVIEW.value: "Motif-Image-6B-Preview",
    ModelRunners.TT_MOCHI_1.value: "Mochi1",
    ModelRunners.TT_WAN_2_2.value: "Wan22",
    ModelRunners.TT_QWEN_IMAGE.value: "Qwen-Image",
    ModelRunners.TT_QWEN_IMAGE_2512.value: "Qwen-Image-2512",
}


class TTDiTRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
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
            ttnn.set_fabric_config(
                fabric_config, reliability_mode, None, fabric_tensix_config
            )
            return fabric_config
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Fabric configuration failed: {e}"
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

        # 20 minutes to distribute the model on device
        weights_distribution_timeout = 1200
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
            self.logger.error(
                f"Device {self.device_id}: Exception during model loading: {e}"
            )
            raise

        self.logger.info(f"Device {self.device_id}: Model loaded successfully")

        # we use model construct to create the request without validation
        if self.settings.model_service == ModelServices.IMAGE.value:
            self.run(
                [
                    ImageGenerateRequest.model_construct(
                        prompt="Sunrise on a beach",
                        negative_prompt="",
                        num_inference_steps=2,
                    )
                ]
            )
        elif self.settings.model_service == ModelServices.VIDEO.value:
            self.run(
                [
                    VideoGenerateRequest.model_construct(
                        prompt="Sunrise on a beach",
                        negative_prompt="",
                        num_inference_steps=2,
                    )
                ]
            )

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

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
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def create_pipeline(self):
        return StableDiffusion3Pipeline.create_pipeline(
            mesh_device=self.ttnn_device,
            checkpoint_name=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value,
        )

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 25000000}


# Runner for Flux.1 dev and schnell. Model weights from settings.model_weights_path determine the exact model variant.
class TTFlux1Runner(TTDiTRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def create_pipeline(self):
        return Flux1Pipeline.create_pipeline(
            checkpoint_name=self.settings.model_weights_path,
            mesh_device=self.ttnn_device,
        )

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 50000000}


class TTMotifImage6BPreviewRunner(TTDiTRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def create_pipeline(self):
        return MotifPipeline.create_pipeline(
            mesh_device=self.ttnn_device,
            checkpoint_name=SupportedModels.MOTIF_IMAGE_6B_PREVIEW.value,
        )

    def get_pipeline_device_params(self):
        return {"l1_small_size": 32768, "trace_region_size": 31000000}


# Runner for Qwen-Image and Qwen-Image-2512. Model weights from settings.model_weights_path determine the exact model variant.
class TTQwenImageRunner(TTDiTRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def create_pipeline(self):
        return QwenImagePipeline.create_pipeline(
            mesh_device=self.ttnn_device,
            checkpoint_name=self.settings.model_weights_path,
        )

    def get_pipeline_device_params(self):
        return {"trace_region_size": 47000000}


class TTMochi1Runner(TTDiTRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def create_pipeline(self):
        return MochiPipeline.create_pipeline(
            mesh_device=self.ttnn_device,
            checkpoint_name=SupportedModels.MOCHI_1.value,
        )

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
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


class TTWan22Runner(TTDiTRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    def create_pipeline(self):
        return WanPipeline.create_pipeline(mesh_device=self.ttnn_device)

    def load_weights(self):
        return False

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        request = requests[0]
        # TODO: Move parameterization outside of runner class.
        if tuple(self.pipeline.mesh_device.shape) == (4, 8):
            width = 1280
            height = 720
        else:
            width = 832
            height = 480
        num_frames = 81
        frames = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
            seed=int(request.seed or 0),
        )
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    def get_pipeline_device_params(self):
        device_params = {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
        if ttnn.device.is_blackhole():
            device_params["fabric_tensix_config"] = ttnn.FabricTensixConfig.MUX
            device_params["dispatch_core_axis"] = ttnn.device.DispatchCoreAxis.ROW
        elif tuple(self.settings.device_mesh_shape) == (4, 8):
            device_params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
        return device_params
