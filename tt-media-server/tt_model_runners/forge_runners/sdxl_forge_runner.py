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

        # LoRA state — set by _ensure_lora_state on each request. _model_id and
        # _variant are stashed at _load_pipeline so we can reload the base
        # components from HF on each LoRA switch (Strategy A: recompile-on-change).
        self._model_id: str | None = None
        self._variant: str | None = None
        self._current_lora_path: str | None = None
        self._current_lora_scale: float | None = None

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
        """Download weights, instantiate models, compile UNet for TT device.

        When TTXLA_SDXL_FULL_ON_DEVICE=true (and not running on CPU), the two
        CLIP text encoders and the VAE are also compiled with backend="tt" and
        moved to the device. Scheduler always stays on CPU. Default is UNet-only.
        """
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

        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        full_on_device = (
            os.getenv("TTXLA_SDXL_FULL_ON_DEVICE", "false").lower() == "true"
            and not runs_on_cpu
        )

        self.logger.info(
            f"Device {self.device_id}: Loading models from {model_id} "
            f"(resolution={self.resolution}, variant={variant}, "
            f"full_on_device={full_on_device})"
        )

        # VAE — on device as bf16 when full_on_device; otherwise CPU fp32.
        vae_dtype = torch.bfloat16 if full_on_device else torch.float32
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=vae_dtype
        )
        if full_on_device:
            self.vae.compile(backend="tt")
            self.vae = self.vae.to(self.device)

        # UNet — bfloat16, compiled for TT
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", variant=variant, torch_dtype=torch.bfloat16
        )
        if not runs_on_cpu:
            self.unet.compile(backend="tt")
        self.unet = self.unet.to(self.device)

        # Text encoders — on device as bf16 when full_on_device; otherwise CPU fp16.
        te_dtype = torch.bfloat16 if full_on_device else torch.float16
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            variant=variant,
            torch_dtype=te_dtype,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            variant=variant,
            torch_dtype=te_dtype,
        )
        if full_on_device:
            self.text_encoder.compile(backend="tt")
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder_2.compile(backend="tt")
            self.text_encoder_2 = self.text_encoder_2.to(self.device)

        # Tokenizers
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )

        # Scheduler stays on CPU; EulerDiscreteScheduler.step is stateful Python
        # and isn't compile-friendly. Per-step latents are moved CPU-ward inside _generate.
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        # Stash for _rebuild_components: each LoRA switch reloads base UNet (and
        # text encoders, since some adapters target them) from HF using these.
        self._model_id = model_id
        self._variant = variant

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

    def _ensure_lora_state(self, request: ImageGenerateRequest) -> None:
        """Apply request.lora_path / request.lora_scale, recompiling on change.

        Strategy A (recompile-on-change): each LoRA switch reloads UNet and
        the two text encoders from HF, applies the adapter via diffusers'
        StableDiffusionXLPipeline.fuse_lora, then recompiles UNet for TT.
        Costs one UNet compile per switch; safe but slow.
        """
        requested_path = request.lora_path
        requested_scale = request.lora_scale

        needs_change = requested_path != self._current_lora_path or (
            requested_path is not None and requested_scale != self._current_lora_scale
        )
        if not needs_change:
            return

        try:
            self._rebuild_components(
                lora_path=requested_path, lora_scale=requested_scale
            )
            self._current_lora_path = requested_path
            self._current_lora_scale = requested_scale
        except Exception as e:
            self._current_lora_path = None
            self._current_lora_scale = None
            raise RuntimeError(
                f"Failed to apply LoRA state '{requested_path}': {e}"
            ) from e

    def _rebuild_components(
        self, lora_path: str | None, lora_scale: float | None
    ) -> None:
        """Reload UNet + text encoders from HF, optionally apply LoRA, recompile.

        UNet is the only on-device component in the UNet-only configuration,
        so only UNet needs recompile. Text encoders are CPU-resident; we
        still rebuild them because some SDXL adapters mutate TE weights via
        fuse_lora, and the trace runner's contract is that a fresh
        unload+load returns to base state.
        """
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
        from utils.lora_utils import resolve_lora_path

        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"

        self.logger.info(
            f"Device {self.device_id}: Rebuilding components "
            f"(lora_path={lora_path!r}, lora_scale={lora_scale})"
        )

        # Free old components. The old UNet is on the XLA device; deleting the
        # reference lets the device-side allocation be reclaimed.
        del self.unet
        del self.text_encoder
        del self.text_encoder_2

        # Reload on CPU (uncompiled)
        new_unet = UNet2DConditionModel.from_pretrained(
            self._model_id,
            subfolder="unet",
            variant=self._variant,
            torch_dtype=torch.bfloat16,
        )
        new_te = CLIPTextModel.from_pretrained(
            self._model_id,
            subfolder="text_encoder",
            variant=self._variant,
            torch_dtype=torch.float16,
        )
        new_te2 = CLIPTextModelWithProjection.from_pretrained(
            self._model_id,
            subfolder="text_encoder_2",
            variant=self._variant,
            torch_dtype=torch.float16,
        )

        if lora_path is not None:
            local_path = resolve_lora_path(lora_path)
            self.logger.info(
                f"Device {self.device_id}: Loading LoRA from {local_path} "
                f"(scale={lora_scale})"
            )
            # Diffusers' SDXL LoRA loader needs a pipeline-shaped object so it
            # can dispatch adapter weights into UNet and (optionally) the text
            # encoders. We assemble a transient pipeline from the freshly
            # loaded components; load_lora_weights + fuse_lora mutate them
            # in place, after which the pipeline object is discarded.
            pipe = StableDiffusionXLPipeline(
                vae=self.vae,
                text_encoder=new_te,
                text_encoder_2=new_te2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                unet=new_unet,
                scheduler=self.scheduler,
            )
            pipe.load_lora_weights(local_path)
            pipe.fuse_lora(lora_scale=lora_scale)

        # Recompile UNet for TT and move to device
        if not runs_on_cpu:
            new_unet.compile(backend="tt")
        new_unet = new_unet.to(self.device)

        self.unet = new_unet
        self.text_encoder = new_te
        self.text_encoder_2 = new_te2

    def _encode_prompts(self, prompt: str, negative_prompt: str, cpu_cast):
        """Encode prompts through both CLIP encoders. Returns hidden_states and pooled_embeds."""
        encoder_hidden_states = []
        pooled_text_embeds = None

        runs_on_cpu = os.getenv("RUNS_ON_CPU", "false").lower() == "true"
        full_on_device = (
            os.getenv("TTXLA_SDXL_FULL_ON_DEVICE", "false").lower() == "true"
            and not runs_on_cpu
        )

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

            if full_on_device:
                cond_tokens = cond_tokens.to(self.device)
                uncond_tokens = uncond_tokens.to(self.device)

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

            # --- VAE Decode ---
            full_on_device = (
                os.getenv("TTXLA_SDXL_FULL_ON_DEVICE", "false").lower() == "true"
                and not runs_on_cpu
            )
            t0 = time.time()
            latents = latents / self.vae.config.scaling_factor
            if full_on_device:
                vae_in = latents.to(dtype=torch.bfloat16).to(self.device)
                images = self.vae.decode(vae_in).sample  # (1, 3, H, W)
                images = images.to("cpu").to(dtype=torch.float32)
            else:
                latents = latents.to(dtype=torch.float32)
                images = self.vae.decode(latents).sample  # (1, 3, H, W)
            self.logger.info(
                f"Device {self.device_id}: VAE decode took {time.time() - t0:.4f}s "
                f"(full_on_device={full_on_device})"
            )

            return images

    @log_execution_time("SDXL Forge inference")
    def run(self, requests: list[ImageGenerateRequest]) -> list[Image.Image]:
        """Run SDXL inference for a batch of requests. Returns list of PIL Images."""
        if not requests:
            raise ValueError("Empty requests list")

        # We process one request at a time (batch_size=1)
        request = requests[0]

        # Apply LoRA changes before encoding. Recompiles UNet on switch.
        self._ensure_lora_state(request)

        prompt = request.prompt
        if request.lora_path:
            from utils.lora_utils import prepare_prompt_with_lora

            prompt = prepare_prompt_with_lora(prompt, request.lora_path)

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
