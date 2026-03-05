# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging
import os
import sys
import time
from abc import abstractmethod

_import_log = logging.getLogger("TTLogger")


def _log_import(msg):
    _import_log.info(msg)
    sys.stderr.flush()


_log_import("dit_runners: importing ttnn...")
_t = time.time()
import ttnn  # noqa: E402

_log_import(f"dit_runners: ttnn imported in {time.time() - _t:.1f}s")

from config.constants import ModelRunners, ModelServices, SupportedModels  # noqa: E402
from config.settings import get_settings  # noqa: E402
from domain.image_generate_request import ImageGenerateRequest  # noqa: E402
from domain.video_generate_request import VideoGenerateRequest  # noqa: E402
from telemetry.telemetry_client import TelemetryEvent  # noqa: E402
from tt_model_runners.base_metal_device_runner import (  # noqa: E402
    BaseMetalDeviceRunner,
)
from utils.decorators import log_execution_time  # noqa: E402
from utils.logger import log_exception_chain  # noqa: E402

_log_import("dit_runners: all imports complete (pipeline imports are lazy per runner)")

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
    def __init__(self, device_id: str):
        self.logger = logging.getLogger("TTLogger")
        self.logger.info(f"Device {device_id}: TTDiTRunner.__init__ started")
        t0 = time.time()
        super().__init__(device_id)
        self.pipeline = None
        self.logger.info(
            f"Device {device_id}: TTDiTRunner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def _configure_fabric(self, updated_device_params):
        self.logger.info(
            f"Device {self.device_id}: _configure_fabric called with params: "
            f"{list(updated_device_params.keys())}"
        )
        t0 = time.time()
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
            self.logger.info(
                f"Device {self.device_id}: Setting fabric config: "
                f"fabric={fabric_config}, tensix={fabric_tensix_config}, "
                f"reliability={reliability_mode}"
            )
            ttnn.set_fabric_config(
                fabric_config, reliability_mode, None, fabric_tensix_config
            )
            self.logger.info(
                f"Device {self.device_id}: _configure_fabric completed in {time.time() - t0:.1f}s"
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
        self.logger.info(
            f"Device {self.device_id}: warmup started - "
            f"model_service={self.settings.model_service}, "
            f"model_runner={self.settings.model_runner}"
        )
        warmup_start = time.time()
        load_start = time.time()

        def distribute_block():
            self.pipeline = self.create_pipeline()

        async def _heartbeat(interval=60):
            while True:
                await asyncio.sleep(interval)
                elapsed = time.time() - load_start
                self.logger.info(
                    f"Device {self.device_id}: Model loading in progress... elapsed={elapsed:.0f}s"
                )

        self.logger.info(
            f"Device {self.device_id}: Starting pipeline creation (timeout=1200s)..."
        )
        weights_distribution_timeout = 1200
        heartbeat_task = asyncio.create_task(_heartbeat())
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
        finally:
            heartbeat_task.cancel()

        load_elapsed = time.time() - load_start
        self.logger.info(
            f"Device {self.device_id}: Pipeline created successfully in {load_elapsed:.1f}s"
        )

        if self.settings.model_service == ModelServices.IMAGE.value:
            self.logger.info(
                f"Device {self.device_id}: Running warmup inference (image, 2 steps)..."
            )
            t0 = time.time()
            self.run(
                [
                    ImageGenerateRequest.model_construct(
                        prompt="Sunrise on a beach",
                        negative_prompt="",
                        num_inference_steps=2,
                    )
                ],
            )
            self.logger.info(
                f"Device {self.device_id}: Warmup inference completed in {time.time() - t0:.1f}s"
            )
        elif self.settings.model_service == ModelServices.VIDEO.value:
            self.logger.info(
                f"Device {self.device_id}: Running warmup inference (video, 2 steps)..."
            )
            t0 = time.time()
            self.run(
                [
                    VideoGenerateRequest.model_construct(
                        prompt="Sunrise on a beach",
                        negative_prompt="",
                        num_inference_steps=2,
                    )
                ],
            )
            self.logger.info(
                f"Device {self.device_id}: Warmup inference completed in {time.time() - t0:.1f}s"
            )

        self.logger.info(
            f"Device {self.device_id}: warmup completed in {time.time() - warmup_start:.1f}s"
        )
        return True

    @log_execution_time(
        f"{dit_runner_log_map[get_settings().model_runner]} inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[ImageGenerateRequest]):
        self.logger.info(
            f"Device {self.device_id}: run() called with {len(requests)} request(s)"
        )
        t0 = time.time()
        request = requests[0]
        image = self.pipeline.run_single_prompt(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            seed=int(request.seed or 0),
        )
        self.logger.info(
            f"Device {self.device_id}: run() completed in {time.time() - t0:.1f}s"
        )
        return image


class TTSD35Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        _import_log.info(f"Device {device_id}: TTSD35Runner.__init__ started")
        t0 = time.time()
        super().__init__(device_id)
        _import_log.info(
            f"Device {device_id}: TTSD35Runner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def create_pipeline(self):
        try:
            self.logger.info(
                f"Device {self.device_id}: Importing StableDiffusion3Pipeline..."
            )
            t_imp = time.time()
            from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
                StableDiffusion3Pipeline,
            )

            self.logger.info(
                f"Device {self.device_id}: StableDiffusion3Pipeline imported in {time.time() - t_imp:.1f}s"
            )
            self.logger.info(
                f"Device {self.device_id}: Creating SD3.5 pipeline - "
                f"mesh_shape={tuple(self.ttnn_device.shape)}, "
                f"num_devices={self.ttnn_device.get_num_devices()}"
            )
            start = time.time()
            pipeline = StableDiffusion3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value,
            )
            self.logger.info(
                f"Device {self.device_id}: SD3.5 pipeline created in {time.time() - start:.1f}s"
            )
            return pipeline
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
        _import_log.info(f"Device {device_id}: TTFlux1Runner.__init__ started")
        t0 = time.time()
        super().__init__(device_id)
        _import_log.info(
            f"Device {device_id}: TTFlux1Runner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def create_pipeline(self):
        try:
            self.logger.info(f"Device {self.device_id}: Importing Flux1Pipeline...")
            t_imp = time.time()
            from models.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline

            self.logger.info(
                f"Device {self.device_id}: Flux1Pipeline imported in {time.time() - t_imp:.1f}s"
            )
            self.logger.info(
                f"Device {self.device_id}: Creating Flux1 pipeline - "
                f"checkpoint={self.settings.model_weights_path}, "
                f"mesh_shape={tuple(self.ttnn_device.shape)}, "
                f"num_devices={self.ttnn_device.get_num_devices()}"
            )
            start = time.time()
            pipeline = Flux1Pipeline.create_pipeline(
                checkpoint_name=self.settings.model_weights_path,
                mesh_device=self.ttnn_device,
            )
            self.logger.info(
                f"Device {self.device_id}: Flux1 pipeline created in {time.time() - start:.1f}s"
            )
            return pipeline
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
        _import_log.info(
            f"Device {device_id}: TTMotifImage6BPreviewRunner.__init__ started"
        )
        t0 = time.time()
        super().__init__(device_id)
        _import_log.info(
            f"Device {device_id}: TTMotifImage6BPreviewRunner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def create_pipeline(self):
        try:
            self.logger.info(f"Device {self.device_id}: Importing MotifPipeline...")
            t_imp = time.time()
            from models.tt_dit.pipelines.motif.pipeline_motif import MotifPipeline

            self.logger.info(
                f"Device {self.device_id}: MotifPipeline imported in {time.time() - t_imp:.1f}s"
            )
            self.logger.info(
                f"Device {self.device_id}: Creating Motif pipeline - "
                f"mesh_shape={tuple(self.ttnn_device.shape)}, "
                f"num_devices={self.ttnn_device.get_num_devices()}"
            )
            start = time.time()
            pipeline = MotifPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=SupportedModels.MOTIF_IMAGE_6B_PREVIEW.value,
            )
            self.logger.info(
                f"Device {self.device_id}: Motif pipeline created in {time.time() - start:.1f}s"
            )
            return pipeline
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
        _import_log.info(f"Device {device_id}: TTQwenImageRunner.__init__ started")
        t0 = time.time()
        super().__init__(device_id)
        _import_log.info(
            f"Device {device_id}: TTQwenImageRunner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def create_pipeline(self):
        try:
            self.logger.info(f"Device {self.device_id}: Importing QwenImagePipeline...")
            t_imp = time.time()
            from models.tt_dit.pipelines.qwenimage.pipeline_qwenimage import (
                QwenImagePipeline,
            )

            self.logger.info(
                f"Device {self.device_id}: QwenImagePipeline imported in {time.time() - t_imp:.1f}s"
            )
            self.logger.info(
                f"Device {self.device_id}: Creating Qwen-Image pipeline - "
                f"checkpoint={self.settings.model_weights_path}, "
                f"mesh_shape={tuple(self.ttnn_device.shape)}, "
                f"num_devices={self.ttnn_device.get_num_devices()}"
            )
            start = time.time()
            pipeline = QwenImagePipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
            )
            self.logger.info(
                f"Device {self.device_id}: Qwen-Image pipeline created in {time.time() - start:.1f}s"
            )
            return pipeline
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
        _import_log.info(f"Device {device_id}: TTMochi1Runner.__init__ started")
        t0 = time.time()
        super().__init__(device_id)
        os.environ["TT_DIT_CACHE_DIR"] = "/tmp/TT_DIT_CACHE"
        _import_log.info(
            f"Device {device_id}: TTMochi1Runner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def create_pipeline(self):
        try:
            self.logger.info(f"Device {self.device_id}: Importing MochiPipeline...")
            t_imp = time.time()
            from models.tt_dit.pipelines.mochi.pipeline_mochi import MochiPipeline

            self.logger.info(
                f"Device {self.device_id}: MochiPipeline imported in {time.time() - t_imp:.1f}s"
            )
            self.logger.info(
                f"Device {self.device_id}: Creating Mochi pipeline - "
                f"mesh_shape={tuple(self.ttnn_device.shape)}, "
                f"num_devices={self.ttnn_device.get_num_devices()}"
            )
            start = time.time()
            pipeline = MochiPipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=SupportedModels.MOCHI_1.value,
            )
            self.logger.info(
                f"Device {self.device_id}: Mochi pipeline created in {time.time() - start:.1f}s"
            )
            return pipeline
        except Exception as e:
            log_exception_chain(
                self.logger,
                self.device_id,
                "Mochi pipeline creation failed",
                e,
            )
            raise

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.info(
            f"Device {self.device_id}: Mochi run() called with {len(requests)} request(s)"
        )
        t0 = time.time()
        request = requests[0]
        frames = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=3.5,
            num_frames=168,
            height=480,
            width=848,
            output_type="np",
            seed=int(request.seed or 0),
        )
        self.logger.info(
            f"Device {self.device_id}: Mochi run() completed in {time.time() - t0:.1f}s"
        )
        return frames

    def get_pipeline_device_params(self):
        return {}


class TTWan22Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        _import_log.info(f"Device {device_id}: TTWan22Runner.__init__ started")
        t0 = time.time()
        super().__init__(device_id)
        _import_log.info(
            f"Device {device_id}: TTWan22Runner.__init__ completed in {time.time() - t0:.1f}s"
        )

    def create_pipeline(self):
        try:
            self.logger.info(f"Device {self.device_id}: Importing WanPipeline...")
            t_imp = time.time()
            from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

            self.logger.info(
                f"Device {self.device_id}: WanPipeline imported in {time.time() - t_imp:.1f}s"
            )
            self.logger.info(
                f"Device {self.device_id}: Creating Wan pipeline - "
                f"mesh_shape={tuple(self.ttnn_device.shape)}, "
                f"num_devices={self.ttnn_device.get_num_devices()}"
            )
            start = time.time()
            pipeline = WanPipeline.create_pipeline(mesh_device=self.ttnn_device)
            self.logger.info(
                f"Device {self.device_id}: Wan pipeline created in {time.time() - start:.1f}s"
            )
            return pipeline
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

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
    def run(self, requests: list[VideoGenerateRequest]):
        self.logger.info(
            f"Device {self.device_id}: Wan run() called with {len(requests)} request(s)"
        )
        t0 = time.time()
        request = requests[0]
        if tuple(self.pipeline.mesh_device.shape) == (4, 8):
            width = 1280
            height = 720
        else:
            width = 832
            height = 480
        num_frames = 81
        self.logger.info(
            f"Device {self.device_id}: Wan inference params: {width}x{height}, "
            f"{num_frames} frames, {request.num_inference_steps} steps"
        )
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
        self.logger.info(
            f"Device {self.device_id}: Wan run() completed in {time.time() - t0:.1f}s"
        )
        return frames

    def get_pipeline_device_params(self):
        device_params = {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
        is_bh = ttnn.device.is_blackhole()
        mesh_shape = tuple(self.settings.device_mesh_shape)
        if is_bh:
            device_params["fabric_tensix_config"] = ttnn.FabricTensixConfig.MUX
            device_params["dispatch_core_axis"] = ttnn.device.DispatchCoreAxis.ROW
        elif mesh_shape == (4, 8):
            device_params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
        self.logger.info(
            f"Device {self.device_id}: Wan get_pipeline_device_params: "
            f"is_blackhole={is_bh}, mesh_shape={mesh_shape}, params={device_params}"
        )
        return device_params
