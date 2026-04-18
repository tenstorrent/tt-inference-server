# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
import time

os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

import torch
from config.constants import SupportedModels
from domain.image_generate_request import ImageGenerateRequest
from PIL import Image
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.logger import log_exception_chain


class SDXLForgeRunner(BaseDeviceRunner):
    """SDXL text-to-image pipeline via Forge compiler (tt-xla).

    Runs on Blackhole (P100/P150) devices. One instance per chip.
    """

    LOAD_TIMEOUT_SECONDS = 300
    WARMUP_TIMEOUT_SECONDS = 2000
    DEFAULT_RESOLUTION = 512

    def __init__(self, device_id: str):
        super().__init__(device_id, cpu_threads="8", num_torch_threads=8)
        self._setup_done = False
        self.device = None  # XLA or CPU device

        # Pipeline components (set during _load_pipeline)
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.scheduler = None

        env_resolution = os.getenv("TTXLA_SDXL_RESOLUTION")
        if env_resolution is not None:
            assert env_resolution in ("1024", "512"), (
                f"TTXLA_SDXL_RESOLUTION must be 1024 or 512, got '{env_resolution}'"
            )
            self.resolution = int(env_resolution)
            self.logger.info(
                f"Device {self.device_id}: Resolution overridden by TTXLA_SDXL_RESOLUTION={self.resolution}"
            )
        else:
            self.resolution = self.DEFAULT_RESOLUTION
            self.logger.info(
                f"Device {self.device_id}: TTXLA_SDXL_RESOLUTION not set, using default {self.resolution} resolution"
            )
        self.latent_size = self.resolution // 8

    def _init_device(self):
        """Initialize TT XLA device or fall back to CPU."""
        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        if runs_on_cpu:
            self.device = torch.device("cpu")
            self.logger.info(f"Device {self.device_id}: Using CPU fallback")
        else:
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr

            xr.set_device_type("TT")
            self.device = xm.xla_device()
            self.logger.info(f"Device {self.device_id}: Using TT XLA device")

    @log_execution_time("SDXL Forge warmup")
    async def warmup(self) -> bool:
        if self._setup_done:
            self.logger.info(f"Device {self.device_id}: Already warmed up")
            return True

        self._init_device()

        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "1"))
        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        if not runs_on_cpu:
            import torch_xla

            torch_xla.set_custom_compile_options(
                {"optimization_level": optimization_level}
            )

        # Run load and warmup synchronously on the current thread.
        # This is critical: torch_xla/torch._dynamo compilation state is
        # thread-local. If warmup runs on a different thread (e.g. via
        # asyncio.to_thread) than inference (run()), the compiled XLA graph
        # cache won't be found and the model will fully recompile on the
        # first real request. The worker process has nothing else to do
        # during warmup, so blocking the event loop is safe.

        # Phase 1: Load pipeline (download weights, compile UNet)
        try:
            self._load_pipeline()
        except Exception as e:
            log_exception_chain(self.logger, self.device_id, "Pipeline load failed", e)
            raise

        self.logger.info(f"Device {self.device_id}: Pipeline loaded")

        # Phase 2: Warmup inference (trigger JIT compilation)
        try:
            self._warmup_inference_pass()
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Warmup inference failed", e
            )
            raise

        self.logger.info(f"Device {self.device_id}: Warmup completed")
        self._setup_done = True
        return True

    def _load_pipeline(self):
        """Download weights, instantiate models, compile UNet for TT device."""
        from diffusers import (
            AutoencoderKL,
            EulerDiscreteScheduler,
            UNet2DConditionModel,
        )
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        # Select model based on resolution.
        # 512px uses hotshotco/SDXL-512, 1024px uses stabilityai/stable-diffusion-xl-base-1.0.
        if self.resolution == 512:
            model_id = SupportedModels.STABLE_DIFFUSION_XL_512.value
            if (
                self.settings.model_weights_path
                and self.settings.model_weights_path
                != SupportedModels.STABLE_DIFFUSION_XL_BASE.value
            ):
                self.logger.warning(
                    f"Device {self.device_id}: model_weights_path={self.settings.model_weights_path!r} "
                    f"is set but resolution is 512 — using {model_id} instead"
                )
        else:
            model_id = (
                self.settings.model_weights_path
                or SupportedModels.STABLE_DIFFUSION_XL_BASE.value
            )

        # hotshotco/SDXL-512 doesn't ship native fp16 weights; load full precision and cast
        variant = (
            "fp16"
            if model_id == SupportedModels.STABLE_DIFFUSION_XL_BASE.value
            else None
        )

        self.logger.info(
            f"Device {self.device_id}: Loading models from {model_id} "
            f"(resolution={self.resolution}, variant={variant})"
        )

        # VAE — float32, stays on CPU
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

        # UNet — bfloat16, compiled for TT
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", variant=variant, torch_dtype=torch.bfloat16
        )
        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        if not runs_on_cpu:
            self.unet.compile(backend="tt")
        self.unet = self.unet.to(self.device)

        # Text encoders — float16, stay on CPU
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            variant=variant,
            torch_dtype=torch.float16,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            variant=variant,
            torch_dtype=torch.float16,
        )

        # Tokenizers
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )

        # Scheduler
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

    def _warmup_inference_pass(self):
        """Run a short inference pass to trigger JIT compilation."""
        self.logger.info(f"Device {self.device_id}: Running warmup inference")
        self._generate(
            prompt="a photo of a cat",
            negative_prompt="",
            cfg_scale=7.5,
            num_inference_steps=20,
            seed=42,
        )
        self.logger.info(f"Device {self.device_id}: Warmup inference done")

    def _encode_prompts(self, prompt: str, negative_prompt: str, cpu_cast):
        """Encode prompts through both CLIP encoders. Returns hidden_states and pooled_embeds."""
        encoder_hidden_states = []
        pooled_text_embeds = None

        for tokenizer, text_encoder in [
            (self.tokenizer, self.text_encoder),
            (self.tokenizer_2, self.text_encoder_2),
        ]:
            cond_tokens = torch.tensor(
                tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids,
                dtype=torch.long,
            )
            uncond_tokens = torch.tensor(
                tokenizer.batch_encode_plus(
                    [negative_prompt or ""], padding="max_length", max_length=77
                ).input_ids,
                dtype=torch.long,
            )

            cond_output = text_encoder(cond_tokens, output_hidden_states=True)
            uncond_output = text_encoder(uncond_tokens, output_hidden_states=True)

            cond_hidden = cond_output.hidden_states[-2]  # penultimate layer
            uncond_hidden = uncond_output.hidden_states[-2]

            if text_encoder is self.text_encoder_2:
                pooled_text_embeds = torch.cat(
                    [uncond_output.text_embeds, cond_output.text_embeds], dim=0
                )  # (2, D)

            curr_hidden = torch.cat([uncond_hidden, cond_hidden], dim=0)  # (2, T, Di)
            encoder_hidden_states.append(curr_hidden)

        encoder_hidden_states = torch.cat(
            encoder_hidden_states, dim=-1
        )  # (2, T, D1+D2)
        return encoder_hidden_states, pooled_text_embeds

    def _generate(
        self,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        num_inference_steps: int,
        seed: int | None,
    ) -> torch.Tensor:
        """Run the full SDXL pipeline. Returns image tensor (B, 3, H, W) in [-1, 1]."""
        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"

        if runs_on_cpu:

            def tt_cast(x):
                return x.to(dtype=torch.bfloat16)
        else:

            def tt_cast(x):
                if x.device == torch.device("cpu"):
                    return x.to(dtype=torch.bfloat16).to(device=self.device)
                return x.to(dtype=torch.bfloat16)

        def cpu_cast(x):
            return x.to("cpu").to(dtype=torch.float16)

        with torch.no_grad():
            # --- Text Encoding (CPU) ---
            t0 = time.time()
            encoder_hidden_states, pooled_text_embeds = self._encode_prompts(
                prompt, negative_prompt, cpu_cast
            )
            self.logger.info(
                f"Device {self.device_id}: Text encoding took {time.time() - t0:.4f}s"
            )

            # --- Latent Initialization ---
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()
            latents = torch.randn(
                (1, 4, self.latent_size, self.latent_size),
                generator=generator,
                dtype=torch.float16,
            )

            self.scheduler.set_timesteps(num_inference_steps)
            latents = latents * self.scheduler.init_noise_sigma

            # time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
            time_ids = torch.tensor(
                [
                    self.resolution,
                    self.resolution,
                    0,
                    0,
                    self.resolution,
                    self.resolution,
                ]
            ).repeat(2, 1)  # (2, 6)

            # --- Diffusion Loop ---
            t0 = time.time()
            for i, timestep in enumerate(self.scheduler.timesteps):
                model_input = latents.repeat(2, 1, 1, 1)  # (2, 4, H, W) for CFG
                model_input = self.scheduler.scale_model_input(model_input, timestep)

                model_input = tt_cast(model_input)
                t = tt_cast(timestep.unsqueeze(0))
                enc_hs = tt_cast(encoder_hidden_states)
                pooled = tt_cast(pooled_text_embeds)
                tids = tt_cast(time_ids)

                unet_output = self.unet(
                    model_input,
                    t,
                    enc_hs,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": tids},
                ).sample

                unet_output = cpu_cast(unet_output)

                # CFG guidance
                uncond_out, cond_out = unet_output.chunk(2)
                model_output = uncond_out + (cond_out - uncond_out) * cfg_scale

                timestep = cpu_cast(timestep)
                latents = cpu_cast(latents)
                latents = self.scheduler.step(
                    model_output, timestep, latents
                ).prev_sample

            self.logger.info(
                f"Device {self.device_id}: UNet diffusion ({num_inference_steps} steps) took {time.time() - t0:.4f}s"
            )

            # --- VAE Decode (CPU) ---
            t0 = time.time()
            latents = latents / self.vae.config.scaling_factor
            latents = latents.to(dtype=torch.float32)
            images = self.vae.decode(latents).sample  # (1, 3, H, W)
            self.logger.info(
                f"Device {self.device_id}: VAE decode took {time.time() - t0:.4f}s"
            )

            return images

    @log_execution_time("SDXL Forge inference")
    def run(self, requests: list[ImageGenerateRequest]) -> list[Image.Image]:
        """Run SDXL inference for a batch of requests. Returns list of PIL Images."""
        if not requests:
            raise ValueError("Empty requests list")

        # We process one request at a time (batch_size=1)
        request = requests[0]

        prompt = request.prompt
        negative_prompt = request.negative_prompt or ""
        cfg_scale = request.guidance_scale
        num_inference_steps = request.num_inference_steps
        seed = request.seed

        image_tensor = self._generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        # Convert tensor to PIL Image
        # image_tensor is (1, 3, H, W) in range [-1, 1]
        image_tensor = torch.clamp(image_tensor / 2 + 0.5, 0.0, 1.0)  # -> [0, 1]
        image_tensor = (image_tensor * 255.0).to(dtype=torch.uint8)
        image_np = image_tensor.cpu().squeeze(0).numpy()  # (3, H, W)
        image_np = image_np.transpose(1, 2, 0)  # (H, W, 3)
        pil_image = Image.fromarray(image_np)

        return [pil_image]
