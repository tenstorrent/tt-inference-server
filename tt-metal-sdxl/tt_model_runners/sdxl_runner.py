import asyncio
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
)
from models.utility_functions import profiler

class TTSDXLRunner(DeviceRunner):
    device = None
    batch_size = 0
    tt_unet = None
    tt_scheduler = None
    ttnn_prompt_embeds = None
    ttnn_time_ids = None
    ttnn_text_embeds = None
    ttnn_timesteps = []
    extra_step_kwargs = None
    guidance_scale = 5.0
    scaling_factor = None
    tt_vae = None
    pipeline = None
    latents = None

    def __init__(self):
        pass
        self.logger = TTLogger()

    def _set_fabric(self,fabric_config):
        # If fabric_config is not None, set it to fabric_config
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def _mesh_device(self):
        device_params = {'l1_small_size': 57344}
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

    def close_device(self) -> bool:
        for submesh in self.mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(self.mesh_device)

    async def load_model(self)->bool:
        self.logger.info("Loading model...")
        self.ttnn_device = self._mesh_device()

        # 1. Load components
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

        self.runInference("Sunrise on a beach", 20)

        self.logger.info("Model warmup completed")

        return True

    def runInference(self, prompt: str, num_inference_steps: int = 50):
        prompts = [prompt]

        torch.manual_seed(0)

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
                negative_prompt=None,
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

        prompt_embeds_torch = torch.cat(prompt_embeds, dim=0)
        negative_prompt_embeds_torch = torch.cat(negative_prompt_embeds, dim=0)
        pooled_prompt_embeds_torch = torch.cat(pooled_prompt_embeds, dim=0)
        negative_pooled_prompt_embeds_torch = torch.cat(negative_pooled_prompt_embeds, dim=0)

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps, cpu_device, None, None)

        # Convert timesteps to ttnn
        ttnn_timesteps = []
        for t in timesteps:
            scalar_tensor = torch.tensor(t).unsqueeze(0)
            ttnn_timesteps.append(
                ttnn.from_torch(
                    scalar_tensor,
                    dtype=ttnn.bfloat16,
                    device=self.ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
                )
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

        torch_prompt_embeds = torch.stack([negative_prompt_embeds_torch, prompt_embeds_torch], dim=1)
        torch_add_text_embeds = torch.stack([negative_pooled_prompt_embeds_torch, pooled_prompt_embeds_torch], dim=1)
        ttnn_prompt_embeds = ttnn.from_torch(
            torch_prompt_embeds,
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.ttnn_device, dim=0),
        )
        ttnn_add_text_embeds = ttnn.from_torch(
            torch_add_text_embeds,
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.ttnn_device, dim=0),
        )

        ttnn_add_time_id1 = ttnn.from_torch(
            negative_add_time_ids.squeeze(0),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        ttnn_add_time_id2 = ttnn.from_torch(
            add_time_ids.squeeze(0),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        ttnn_time_ids = [ttnn_add_time_id1, ttnn_add_time_id2]
        ttnn_text_embeds = [
            [
                ttnn_add_text_embed[0],
                ttnn_add_text_embed[1],
            ]
            for ttnn_add_text_embed in ttnn_add_text_embeds
        ]

        scaling_factor = ttnn.from_torch(
            torch.Tensor([self.pipeline.vae.config.scaling_factor]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )

        B, C, H, W = latents.shape

        # All device code will work with channel last tensors
        latents = torch.permute(latents, (0, 2, 3, 1))
        latents = latents.reshape(1, 1, B * H * W, C)

        latents_clone = latents.clone()

        latents = ttnn.from_torch(
            latents,
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )


        images = []
        self.logger.info("Starting ttnn inference...")
        for iter in range(len(prompts) // self.batch_size):
            self.logger.info(
                f"Running inference for prompts {iter * self.batch_size + 1}-{iter * self.batch_size + self.batch_size}/{len(prompts)}"
            )
            imgs = run_tt_image_gen(
                self.ttnn_device,
                self.tt_unet,
                self.tt_scheduler,
                latents,
                ttnn_prompt_embeds,
                ttnn_time_ids,
                ttnn_text_embeds,
                ttnn_timesteps,
                extra_step_kwargs,
                guidance_scale,
                scaling_factor,
                [B, C, H, W],
                self.tt_vae,
                self.batch_size,
                iter,
            )

            self.logger.info(f"Denoising loop for {self.batch_size} promts completed in {profiler.get('denoising_loop'):.2f} seconds")
            self.logger.info(
                f"{'On device VAE'} decoding completed in {profiler.get('vae_decode'):.2f} seconds"
            )
            profiler.clear()

            for idx, img in enumerate(imgs):
                if iter == len(prompts) // self.batch_size - 1 and idx >= self.batch_size - needed_padding:
                    break
                img = img.unsqueeze(0)
                img = self.pipeline.image_processor.postprocess(img, output_type="pil")[0]
                images.append(img)

            latents = latents_clone.clone()
            latents = ttnn.from_torch(
                latents,
                dtype=ttnn.bfloat16,
                device=self.ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
            )

        return images