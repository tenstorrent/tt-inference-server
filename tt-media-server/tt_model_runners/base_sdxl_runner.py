# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import abstractmethod
import asyncio
from config.settings import get_settings
from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from utils.logger import TTLogger
import ttnn
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG
)
from models.common.utility_functions import profiler
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline

class BaseSDXLRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.tt_sdxl: TtSDXLPipeline = None
        self.settings = get_settings()
        self.logger = TTLogger()
        # setup is tensor parallel if device mesh shape first param starts with 2
        self.is_tensor_parallel = self.settings.device_mesh_shape[0] > 1
        if (self.is_tensor_parallel):
            self.logger.info(f"Device {self.device_id}: Tensor parallel mode enabled with mesh shape {self.settings.device_mesh_shape}")
        self.batch_size = 0
        self.pipeline = None

    @log_execution_time("SDXL warmup")
    async def load_model(self, device)->bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")
        if device is None:
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device
        
        self.batch_size = self.settings.max_batch_size

        # 1. Load components
        self._load_pipeline()
        
        self.logger.info(f"Device {self.device_id}: Model weights downloaded successfully")

        # 6 minutes to distribute the model on device
        weights_distribution_timeout = 720

        try:
            await asyncio.wait_for(asyncio.to_thread(self._distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id}: ttnn.distribute block timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Exception during model loading: {e}")
            raise

        self.logger.info(f"Device {self.device_id}: Model loaded successfully")


        warmup_inference_timeout = 1000
    
        try:
            await asyncio.wait_for(asyncio.to_thread(self._warmup_inference_block), timeout=warmup_inference_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Device {self.device_id}: warmup inference timed out after {warmup_inference_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Exception during warmup inference: {e}")
            raise

        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @abstractmethod
    def run_inference(self, requests: list[ImageGenerateRequest]):
        pass

    def close_device(self, device) -> bool:
        if device is None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
        else:
            ttnn.close_mesh_device(device)
        return True
    
    def get_device(self):
        # for now use all available devices
        return self._mesh_device()

    def _set_fabric(self, fabric_config):
        # If fabric_config is not None, set it to fabric_config
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def _mesh_device(self):
        device_params = {'l1_small_size': SDXL_L1_SMALL_SIZE, 'trace_region_size': self.settings.trace_region_size or SDXL_TRACE_REGION_SIZE}
        if self.is_tensor_parallel:
            device_params["fabric_config"] = SDXL_FABRIC_CONFIG

        mesh_shape = ttnn.MeshShape(self.settings.device_mesh_shape)

        updated_device_params = self.get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"Device {self.device_id}: multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device
    
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

    def _process_prompts(self, requests: list[ImageGenerateRequest]) -> tuple[list[str], str, int]:
        prompts = [request.prompt for request in requests]
        negative_prompt = requests[0].negative_prompt
        if isinstance(prompts, str):
            prompts = [prompts]

        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        prompts = prompts + [""] * needed_padding

        prompts_2 = requests[0].prompt_2
        negative_prompt_2 = requests[0].negative_prompt_2
        if prompts_2 is not None:
            prompts_2 = [request.prompt_2 for request in requests]
            if isinstance(prompts_2, str):
                prompts_2 = [prompts_2]

            needed_padding = (self.batch_size - len(prompts_2) % self.batch_size) % self.batch_size
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
            
            self.logger.info(
                f"Device {self.device_id}: Prepare input tensors for {self.batch_size} prompts completed in {profiler.times['prepare_input_tensors'][-1]:.2f} seconds"
            )
            self.logger.info(f"Device {self.device_id}: Image gen for {self.batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
            self.logger.info(
                f"Device {self.device_id}: Denoising loop for {self.batch_size} prompts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
            )
            self.logger.info(
                f"Device {self.device_id}: On device VAE decoding completed in {profiler.times['vae_decode'][-1]:.2f} seconds"
            )
            self.logger.info(f"Device {self.device_id}: Output tensor read completed in {profiler.times['read_output_tensor'][-1]:.2f} seconds")

            for idx, img in enumerate(imgs):
                if iter == len(prompts) // self.batch_size - 1 and idx >= self.batch_size - needed_padding:
                    break
                img = img.unsqueeze(0)
                img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
                images.append(img)

        return images
