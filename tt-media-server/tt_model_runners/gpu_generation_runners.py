# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GPU stand-ins for the generations API's image runners, for testing the API,
the evals and the release workflow on a CUDA host without TT hardware.

Each runs its model through the GPU reference tool of the model's validation
dataset (``tools/*_gpu_reference.py`` on Hugging Face), whose path is
``GPU_REFERENCE_TOOL``: the tool's pipeline, settings and initial noise, drawn
from the request's seed at the request's size. A request with a dataset case's
inputs and seed therefore comes back as that case's reference image.

One worker per GPU: the worker's device id (``DEVICE_IDS``, e.g.
``(0),(1),...``) is its ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from types import ModuleType

from domain.flux2_klein_4b_request import Flux2klein4bRequest
from domain.ideogram4_request import IDEOGRAM_PRESETS, Ideogram4Request
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from tt_model_runners.base_device_runner import BaseDeviceRunner


def _reference_tool() -> ModuleType:
    path = os.environ.get("GPU_REFERENCE_TOOL")
    if not path or not os.path.isfile(path):
        raise ValueError(
            "GPU runners need GPU_REFERENCE_TOOL set to the model's "
            f"tools/*_gpu_reference.py; got {path!r}"
        )
    spec = importlib.util.spec_from_file_location("gpu_reference_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _GPUGenerationRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        # Before CUDA starts in this process.
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        super().__init__(device_id)
        self.tool = _reference_tool()
        self.pipe = None

    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: loading the GPU pipeline")
        self.pipe = await asyncio.to_thread(self.tool.load_pipeline, *self._load_args())
        self.logger.info(f"Device {self.device_id}: GPU pipeline loaded")
        return True

    def _load_args(self) -> tuple:
        return ()


class GPUQwenImageEdit2511Runner(_GPUGenerationRunner):
    # The tool's base tier, no cache: 40 steps with true CFG 4.0, as TT runs it
    # (generation_runners.QWEN_STEPS, QWEN_GUIDANCE_SCALE; that module needs ttnn).
    tier = "base"

    def _load_args(self) -> tuple:
        return (self.tier, None)

    def run(self, requests: list[QwenImageEdit2511Request]):
        request = requests[0]
        tier = self.tool.TIERS[self.tier]
        noise = self.tool.initial_noise(request._seed, request._width, request._height)
        image = self.pipe(
            image=[image.convert("RGB") for image in request._images["images"]],
            prompt=request._prompt,
            negative_prompt=request.negative_prompt,
            true_cfg_scale=tier["true_cfg_scale"],
            num_inference_steps=tier["steps"],
            height=request._height,
            width=request._width,
            latents=self.tool.pack(noise),
        ).images[0]
        return [image]


class GPUFlux2Klein4BRunner(_GPUGenerationRunner):
    def run(self, requests: list[Flux2klein4bRequest]):
        request = requests[0]
        noise = self.tool.initial_noise(request._seed, request._width, request._height)
        images = [image.convert("RGB") for image in request._images.get("images", [])]
        image = self.pipe(
            image=images or None,
            prompt=request._prompt,
            height=request._height,
            width=request._width,
            num_inference_steps=self.tool.STEPS,
            guidance_scale=self.tool.GUIDANCE_SCALE,
            max_sequence_length=self.tool.MAX_SEQUENCE_LENGTH,
            latents=noise,
        ).images[0]
        return [image]


class GPUIdeogram4Runner(_GPUGenerationRunner):
    def run(self, requests: list[Ideogram4Request]):
        import torch
        from ideogram4 import PRESETS

        request = requests[0]
        preset = PRESETS[IDEOGRAM_PRESETS[request.rendering_speed]]
        noise = self.tool.initial_noise(
            request._seed,
            request._width,
            request._height,
            self.pipe.conditional_transformer.config.in_channels,
        )
        with torch.inference_mode(), self.tool.noise_for_the_call(noise):
            image = self.pipe(
                request._prompt,
                height=request._height,
                width=request._width,
                num_steps=preset.num_steps,
                guidance_schedule=preset.guidance_schedule,
                mu=preset.mu,
                std=preset.std,
                seed=request._seed,
            )[0]
        return [image]
