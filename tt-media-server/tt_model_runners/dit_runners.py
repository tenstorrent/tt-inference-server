# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from config.settings import get_settings
from config.constants import SupportedModels, ModelRunners
from abc import abstractmethod
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from utils.logger import TTLogger
import ttnn
import torch
from models.experimental.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import StableDiffusion3Pipeline
from models.experimental.tt_dit.pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from models.experimental.tt_dit.pipelines.mochi.pipeline_mochi import MochiPipeline
from models.experimental.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, VaeHWParallelConfig, ParallelFactor, MochiVAEParallelConfig
from domain.image_generate_request import ImageGenerateRequest

dit_runner_log_map={
    ModelRunners.TT_SD3_5.value: "SD35",
    ModelRunners.TT_FLUX_1_DEV.value: "FLUX.1-dev",
    ModelRunners.TT_FLUX_1_SCHNELL.value: "FLUX.1-schnell",
    ModelRunners.TT_MOCHI_1.value: "Mochi1",
    ModelRunners.TT_WAN_2_2.value: "Wan22",
}

class TTDiTRunner(BaseDeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.settings = get_settings()
        self.pipeline = None
        self.logger = TTLogger()
        self.mesh_device = self._mesh_device(ttnn.MeshShape(*self.settings.device_mesh_shape))

    @staticmethod
    @abstractmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice):
        """Create a pipeline for the model"""

    @staticmethod
    @abstractmethod
    def get_pipeline_device_params(mesh_shape):
        """Get the device parameters for the pipeline"""

    def get_device(self):
        return self.mesh_device

    def _mesh_device(self, mesh_shape):
        device_params = self.get_pipeline_device_params(mesh_shape)
        fabric_config = device_params.pop("fabric_config", ttnn.FabricConfig.FABRIC_1D)
        updated_device_params = self.get_updated_device_params(device_params)
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"Device {self.device_id}: multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def close_device(self, device) -> bool:
        ttnn.close_mesh_device(device)
        return True

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} warmup")
    async def load_model(self, device)->bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        if self.mesh_device != device:
            raise Exception(f"Device {self.device_id}: Passed in device is not the same as device used for SD35 runner initialization")

        distribute_block = lambda: setattr(self,"pipeline",self.create_pipeline(mesh_device=self.mesh_device))

        # 12 minutes to distribute the model on device
        weights_distribution_timeout = 720
        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id}: ttnn.distribute block timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Exception during model loading: {e}")
            raise

        self.logger.info(f"Device {self.device_id}: Model loaded successfully")

        # we use model construct to create the request without validation
        self.run_inference([ImageGenerateRequest.model_construct(
                prompt="Sunrise on a beach",
                negative_prompt="",
                num_inference_steps=2  # Some models (e.g. Mochi) will hit divide-by-zero errors if only 1 step is used.
            )])

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
    def run_inference(self, requests: list[ImageGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        prompt = requests[0].prompt
        negative_prompt = requests[0].negative_prompt
        seed = int(requests[0].seed or 0)
        num_inference_steps = requests[0].num_inference_steps or self.settings.num_inference_steps
        image = self.pipeline.run_single_prompt(prompt=prompt,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,seed=seed)
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return image

class TTSD35Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice):
        return StableDiffusion3Pipeline.create_pipeline(mesh_device=mesh_device, model_checkpoint_path=SupportedModels.STABLE_DIFFUSION_3_5_LARGE.value)

    @staticmethod
    def get_pipeline_device_params(mesh_shape):
        return {"l1_small_size": 32768, "trace_region_size": 25000000}

#TODO: Merge dev and schnell
class TTFlux1DevRunner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice):
        return Flux1Pipeline.create_pipeline(
            checkpoint_name=SupportedModels.FLUX_1_DEV.value,
            mesh_device=mesh_device,
        )

    @staticmethod
    def get_pipeline_device_params(mesh_shape):
        return {"l1_small_size": 32768, "trace_region_size": 34000000}

class TTFlux1SchnellRunner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice):
        return Flux1Pipeline.create_pipeline(
            checkpoint_name=SupportedModels.FLUX_1_SCHNELL.value,
            mesh_device=mesh_device,
        )

    @staticmethod
    def get_pipeline_device_params(mesh_shape):
        return {"l1_small_size": 32768, "trace_region_size": 34000000}

class TTMochi1Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice):

        # TODO: Set optimal configuration settings in tt-metal code.
        device_configs = {
            (2, 4): {"sp_axis": 0, "tp_axis": 1, "vae_mesh_shape": (1, 8), "vae_sp_axis": 0, "vae_tp_axis": 1, "num_links": 1},
            (4, 8): {"sp_axis": 1, "tp_axis": 0, "vae_mesh_shape": (4, 8), "vae_sp_axis": 0, "vae_tp_axis": 1, "num_links": 4},
        }

        config = device_configs[tuple(mesh_device.shape)]

        sp_factor = tuple(mesh_device.shape)[config["sp_axis"]]
        tp_factor = tuple(mesh_device.shape)[config["tp_axis"]]

        # Create parallel config
        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=config["tp_axis"]),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=config["sp_axis"]),
        )

        if config["vae_mesh_shape"][config["vae_sp_axis"]] == 1:
            w_parallel_factor = 1
        else:
            w_parallel_factor = 2

        vae_parallel_config = MochiVAEParallelConfig(
            time_parallel=ParallelFactor(factor=config["vae_mesh_shape"][config["vae_tp_axis"]], mesh_axis=config["vae_tp_axis"]),
            w_parallel=ParallelFactor(factor=w_parallel_factor, mesh_axis=config["vae_sp_axis"]),
            h_parallel=ParallelFactor(factor=config["vae_mesh_shape"][config["vae_sp_axis"]] // w_parallel_factor, mesh_axis=config["vae_sp_axis"]),
        )
        assert vae_parallel_config.h_parallel.factor * vae_parallel_config.w_parallel.factor == config["vae_mesh_shape"][config["vae_sp_axis"]]
        assert vae_parallel_config.h_parallel.mesh_axis == vae_parallel_config.w_parallel.mesh_axis

        return MochiPipeline(
            mesh_device=mesh_device,
            vae_mesh_shape=config["vae_mesh_shape"],
            parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            num_links=config["num_links"],
            use_cache=False,
            use_reference_vae=False,
            model_name=SupportedModels.MOCHI_1.value
        )

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
    def run_inference(self, requests: list[ImageGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        prompt = requests[0].prompt
        generator = torch.Generator("cpu").manual_seed(int(requests[0].seed or 0))
        num_inference_steps = requests[0].num_inference_steps or self.settings.num_inference_steps
        frames = self.pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            num_frames=168,  # TODO: Parameterize output dimensions.
            height=480,
            width=848,
        ).frames[0]
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    @staticmethod
    def get_pipeline_device_params(mesh_shape):
        return {}

class TTWan22Runner(TTDiTRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice):

        # TODO: Set optimal configuration settings in tt-metal code.
        # FIXME: How do we distinguish between WH and BH here?
        device_configs = {
            (2, 4): {"sp_axis": 0, "tp_axis": 1, "num_links": 1, "dynamic_load": True, "topology": ttnn.Topology.Linear},
            (4, 8): {"sp_axis": 1, "tp_axis": 0, "num_links": 4, "dynamic_load": False, "topology": ttnn.Topology.Ring},
        }

        config = device_configs[tuple(mesh_device.shape)]

        sp_factor = tuple(mesh_device.shape)[config["sp_axis"]]
        tp_factor = tuple(mesh_device.shape)[config["tp_axis"]]

        parallel_config = DiTParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=config["tp_axis"], factor=tp_factor),
            sequence_parallel=ParallelFactor(mesh_axis=config["sp_axis"], factor=sp_factor),
            cfg_parallel=None,
        )
        vae_parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[config["sp_axis"]], mesh_axis=config["sp_axis"]),
            width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[config["tp_axis"]], mesh_axis=config["tp_axis"]),
        )

        return WanPipeline(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            vae_parallel_config=vae_parallel_config,
            num_links=config["num_links"],
            use_cache=False,
            boundary_ratio=0.875,
            dynamic_load=config["dynamic_load"],
            topology=config["topology"],
         )

    @log_execution_time(f"{dit_runner_log_map[get_settings().model_runner]} inference")
    def run_inference(self, requests: list[ImageGenerateRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        prompt = requests[0].prompt
        negative_prompt = requests[0].negative_prompt
        generator = torch.Generator("cpu").manual_seed(int(requests[0].seed or 0))
        num_inference_steps = requests[0].num_inference_steps or self.settings.num_inference_steps
        frames = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height = 480,
            width = 832,
            num_frames = 81,  # TODO: Parameterize output dimensions.
            num_inference_steps=num_inference_steps,
            guidance_scale=3.0,
            guidance_scale_2=4.0,
        )
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return frames

    @staticmethod
    def get_pipeline_device_params(mesh_shape):
        # FIXME: How can we switch based on WH or BH configuration here?
        device_params = {"l1_small_size": 32768, "trace_region_size": 34000000}
        if tuple(mesh_shape) == (4, 8):
            device_params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
        return device_params
