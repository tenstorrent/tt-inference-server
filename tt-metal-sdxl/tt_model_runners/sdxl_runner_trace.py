import asyncio
from typing import List
from config.settings import settings
from tests.scripts.common import get_updated_device_params
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger
import ttnn
import torch
from diffusers import DiffusionPipeline
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    retrieve_timesteps,
    run_tt_image_gen,
    create_user_tensors,
    allocate_input_tensors,
    prepare_input_tensors,
)
from models.utility_functions import profiler


class TTSDXLRunnerTrace(DeviceRunner):
    device = None
    batch_size = 0
    tt_unet = None
    tt_scheduler = None
    ttnn_prompt_embeds = None
    ttnn_time_ids = None
    ttnn_text_embeds = None
    ttnn_timesteps = []
    extra_step_kwargs = None
    scaling_factor = None
    tt_vae = None
    pipeline = None
    latents = None

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()

    def _set_fabric(self,fabric_config):
        # If fabric_config is not None, set it to fabric_config
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        # for now use all availalbe devices
        return self._mesh_device()

    def _mesh_device(self):
        device_params = {'l1_small_size': 57344, 'trace_region_size': 33575936}
        device_ids = ttnn.get_device_ids()

        param = len(device_ids)  # Default to using all available devices

        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if num_devices_requested > len(device_ids):
                print("Requested more devices than available. Test not applicable for machine")
            mesh_shape = ttnn.MeshShape(*grid_dims)
            assert num_devices_requested <= len(device_ids), "Requested more devices than available."
        else:
            num_devices_requested = min(param, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)


        updated_device_params = get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device

    def get_devices(self) -> List[ttnn.MeshDevice]:
        device = self._mesh_device()
        device_shape = settings.device_mesh_shape
        return (device, device.create_submeshes(ttnn.MeshShape(*device_shape)))

    def close_device(self, device) -> bool:
        if device is None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
        else:
            ttnn.close_mesh_device(device)
        return True

    async def load_model(self, device)->bool:
        self.logger.info("Loading model...")
        if (device is None):
            self.ttnn_device = self._mesh_device()
        else:
            self.ttnn_device = device

        # 1. Load components
        # TODO check how to point to a model file
        self.pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

        self.logger.info("Model weights downloaded successfully")

        self.batch_size = self.ttnn_device.get_num_devices()

        def distribute_block():
            try:
                with ttnn.distribute(ttnn.ReplicateTensorToMesh(self.ttnn_device)):
                    tt_model_config = ModelOptimisations()
                    self.tt_unet = TtUNet2DConditionModel(
                        self.ttnn_device,
                        self.pipeline.unet.state_dict(),
                        "unet",
                        model_config=tt_model_config,
                    )
                    self.tt_vae = (
                        TtAutoencoderKL(self.ttnn_device, self.pipeline.vae.state_dict(), tt_model_config, self.batch_size)
                    )
                    self.tt_scheduler = TtEulerDiscreteScheduler(
                        self.ttnn_device,
                        self.pipeline.scheduler.config.num_train_timesteps,
                        self.pipeline.scheduler.config.beta_start,
                        self.pipeline.scheduler.config.beta_end,
                        self.pipeline.scheduler.config.beta_schedule,
                        self.pipeline.scheduler.config.trained_betas,
                        self.pipeline.scheduler.config.prediction_type,
                        self.pipeline.scheduler.config.interpolation_type,
                        self.pipeline.scheduler.config.use_karras_sigmas,
                        self.pipeline.scheduler.config.use_exponential_sigmas,
                        self.pipeline.scheduler.config.use_beta_sigmas,
                        self.pipeline.scheduler.config.sigma_min,
                        self.pipeline.scheduler.config.sigma_max,
                        self.pipeline.scheduler.config.timestep_spacing,
                        self.pipeline.scheduler.config.timestep_type,
                        self.pipeline.scheduler.config.steps_offset,
                        self.pipeline.scheduler.config.rescale_betas_zero_snr,
                        self.pipeline.scheduler.config.final_sigmas_type,
                    )
            except Exception as e:
                self.logger.error(f"Error in ttnn.distribute block: {e}")
                raise

        # 6 minutes to distribute the model on device
        weights_distribution_timeout = 360

        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"ttnn.distribute block timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Exception during model loading: {e}")
            raise

        self.pipeline.scheduler = self.tt_scheduler

        self.logger.info("Model loaded successfully")

        self.warmup_inference(["Sunrise on a beach"], 20)

        self.logger.info("Model warmup completed")

        return True

    def warmup_inference(self, prompts: list[str], num_inference_steps: int = 50, negative_prompt: str = None):
        height = 1024
        width = 1024

        cpu_device = "cpu"

        all_embeds = [
            self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=cpu_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            for prompt in prompts
        ]
        
        text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim
        

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = zip(*all_embeds)
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = self.pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds[0].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        scaling_factor = ttnn.from_torch(
            torch.Tensor([self.pipeline.vae.config.scaling_factor]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        
        num_channels_latents = self.pipeline.unet.config.in_channels
        
        latents = self.pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds[0].dtype,
            cpu_device,
            None,
            None,
        )


        B, C, H, W = latents.shape
        
        prompt_embeds_torch = torch.split(torch.cat(prompt_embeds, dim=0), self.batch_size, dim=0)
        negative_prompt_embeds_torch = torch.split(torch.cat(negative_prompt_embeds, dim=0), self.batch_size, dim=0)
        negative_pooled_prompt_embeds_torch = torch.split(
            torch.cat(negative_pooled_prompt_embeds, dim=0), self.batch_size, dim=0
        )
        pooled_prompt_embeds_torch = torch.split(torch.cat(pooled_prompt_embeds, dim=0), self.batch_size, dim=0)

        # All device code will work with channel last tensors
        tt_latents = torch.permute(latents, (0, 2, 3, 1))
        tt_latents = tt_latents.reshape(1, 1, B * H * W, C)
        tt_latents, tt_prompt_embeds, tt_add_text_embeds = create_user_tensors(
            ttnn_device=self.ttnn_device,
            latents=tt_latents,
            negative_prompt_embeds=negative_prompt_embeds_torch,
            prompt_embeds=prompt_embeds_torch,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_torch,
            add_text_embeds=pooled_prompt_embeds_torch,
        )

        self.tt_latents_device, self.tt_prompt_embeds_device, self.tt_text_embeds_device, self.tt_time_ids_device = allocate_input_tensors(
            ttnn_device=self.ttnn_device,
            tt_latents=tt_latents,
            tt_prompt_embeds=tt_prompt_embeds,
            tt_text_embeds=tt_add_text_embeds,
            tt_time_ids=[negative_add_time_ids, add_time_ids],
        )

        self.logger.info("Performing warmup run, to make use of program caching in actual inference...")
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[0],
                tt_add_text_embeds[0][0],
                tt_add_text_embeds[0][1],
            ],
            [self.tt_latents_device, *self.tt_prompt_embeds_device, *self.tt_text_embeds_device],
        )

        self.ttnn_timesteps, num_inference_steps = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps, cpu_device, None, None)

        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(None, 0.0)
        
        guidance_scale = 5.0

        _, _, _, output_shape, _ = run_tt_image_gen(
            self.ttnn_device,
            self.tt_unet,
            self.tt_scheduler,
            self.tt_latents_device,
            self.tt_prompt_embeds_device,
            self.tt_time_ids_device,
            self.tt_text_embeds_device,
            [self.ttnn_timesteps[0]],
            extra_step_kwargs,
            guidance_scale,
            scaling_factor,
            [B, C, H, W],
            self.tt_vae,
            self.batch_size,
            capture_trace=False,
        )

        capture_trace = True
        if capture_trace:
            self.logger.info("Capturing model trace...")
            prepare_input_tensors(
                [
                    tt_latents,
                    *tt_prompt_embeds[0],
                    tt_add_text_embeds[0][0],
                    tt_add_text_embeds[0][1],
                ],
                [self.tt_latents_device, *self.tt_prompt_embeds_device, *self.tt_text_embeds_device],
            )
            _, self.tid, self.output_device, self.output_shape, self.tid_vae = run_tt_image_gen(
                self.ttnn_device,
                self.tt_unet,
                self.tt_scheduler,
                self.tt_latents_device,
                self.tt_prompt_embeds_device,
                self.tt_time_ids_device,
                self.tt_text_embeds_device,
                [self.ttnn_timesteps[0]],
                extra_step_kwargs,
                guidance_scale,
                scaling_factor,
                [B, C, H, W],
                self.tt_vae,
                self.batch_size,
                capture_trace=True,
            )
        profiler.clear()


    def run_inference(self, prompts: list[str], num_inference_steps: int = 50, negative_prompt: str = None):
        if isinstance(prompts, str):
            prompts = [prompts]

        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        prompts = prompts + [""] * needed_padding

        guidance_scale = 5.0

        # 0. Set up default height and width for unet
        height = 1024
        width = 1024

        cpu_device = "cpu"

        all_embeds = [
            self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=cpu_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            for prompt in prompts
        ]

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = zip(*all_embeds)

        ##########################################################################

        prompt_embeds_torch = torch.split(torch.cat(prompt_embeds, dim=0), self.batch_size, dim=0)
        negative_prompt_embeds_torch = torch.split(torch.cat(negative_prompt_embeds, dim=0), self.batch_size, dim=0)
        pooled_prompt_embeds_torch = torch.split(torch.cat(pooled_prompt_embeds, dim=0), self.batch_size, dim=0)
        negative_pooled_prompt_embeds_torch = torch.split(
            torch.cat(negative_pooled_prompt_embeds, dim=0), self.batch_size, dim=0
        )

        num_channels_latents = self.pipeline.unet.config.in_channels
        assert num_channels_latents == 4, f"num_channels_latents is {num_channels_latents}, but it should be 4"

        latents = self.pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds[0].dtype,
            cpu_device,
            None,
            None,
        )

        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(None, 0.0)
        text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim
        assert (
            text_encoder_projection_dim == 1280
        ), f"text_encoder_projection_dim is {text_encoder_projection_dim}, but it should be 1280"

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = self.pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds[0].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids


        scaling_factor = ttnn.from_torch(
            torch.Tensor([self.pipeline.vae.config.scaling_factor]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )

        B, C, H, W = latents.shape

        ##########################################################
        images = []
        tt_latents = torch.permute(latents, (0, 2, 3, 1))
        tt_latents = tt_latents.reshape(1, 1, B * H * W, C)
        tt_latents, tt_prompt_embeds, tt_add_text_embeds = create_user_tensors(
            ttnn_device=self.ttnn_device,
            latents=tt_latents,
            negative_prompt_embeds=negative_prompt_embeds_torch,
            prompt_embeds=prompt_embeds_torch,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_torch,
            add_text_embeds=pooled_prompt_embeds_torch,
        )

        tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device = allocate_input_tensors(
            ttnn_device=self.ttnn_device,
            tt_latents=tt_latents,
            tt_prompt_embeds=tt_prompt_embeds,
            tt_text_embeds=tt_add_text_embeds,
            tt_time_ids=[negative_add_time_ids, add_time_ids],
        )

        self.logger.info("Performing warmup run, to make use of program caching in actual inference...")
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[0],
                tt_add_text_embeds[0][0],
                tt_add_text_embeds[0][1],
            ],
            [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
        )
        
        self.logger.info("Starting ttnn inference...")
        for iter in range(len(prompts) // self.batch_size):
            self.logger.info(
                f"Running inference for prompts {iter * self.batch_size + 1}-{iter * self.batch_size + self.batch_size}/{len(prompts)}"
            )
            prepare_input_tensors(
                [
                    tt_latents,
                    *tt_prompt_embeds[iter],
                    tt_add_text_embeds[iter][0],
                    tt_add_text_embeds[iter][1],
                ],
                [self.tt_latents_device, *self.tt_prompt_embeds_device, *self.tt_text_embeds_device],
            )
            imgs, self.tid, self.output_device, self.output_shape, self.tid_vae = run_tt_image_gen(
                self.ttnn_device,
                self.tt_unet,
                self.tt_scheduler,
                self.tt_latents_device,
                self.tt_prompt_embeds_device,
                self.tt_time_ids_device,
                self.tt_text_embeds_device,
                self.ttnn_timesteps,
                extra_step_kwargs,
                guidance_scale,
                scaling_factor,
                [B, C, H, W],
                self.tt_vae,
                self.batch_size,
                tid=self.tid,
                output_device=self.output_device,
                output_shape=self.output_shape,
                tid_vae=self.tid_vae,
            )

            self.logger.info(f"Image gen for {self.batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
            self.logger.info(
                f"Denoising loop for {self.batch_size} promts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
            )
        # images = []
        # self.logger.info("Starting ttnn inference...")
        # for iter in range(len(prompts) // self.batch_size):
        #     self.logger.info(
        #         f"Running inference for prompts {iter * self.batch_size + 1}-{iter * self.batch_size + self.batch_size}/{len(prompts)}"
        #     )
        #     imgs = run_tt_image_gen(
        #         self.ttnn_device,
        #         self.tt_unet,
        #         self.tt_scheduler,
        #         latents,
        #         ttnn_prompt_embeds,
        #         ttnn_time_ids,
        #         ttnn_text_embeds,
        #         ttnn_timesteps,
        #         extra_step_kwargs,
        #         guidance_scale,
        #         scaling_factor,
        #         [B, C, H, W],
        #         self.tt_vae,
        #         self.batch_size,
        #         iter,
        #     )

        #     self.logger.info(f"Denoising loop for {self.batch_size} promts completed in {profiler.get('denoising_loop'):.2f} seconds")
        #     self.logger.info(
        #         f"{'On device VAE'} decoding completed in {profiler.get('vae_decode'):.2f} seconds"
        #     )
        #     profiler.clear()

            for idx, img in enumerate(imgs):
                if iter == len(prompts) // self.batch_size - 1 and idx >= self.batch_size - needed_padding:
                    break
                img = img.unsqueeze(0)
                img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
                images.append(img)

            # latents = latents_clone.clone()
            # latents = ttnn.from_torch(
            #     latents,
            #     dtype=ttnn.bfloat16,
            #     device=self.ttnn_device,
            #     layout=ttnn.TILE_LAYOUT,
            #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
            #     mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
            # )

        return images