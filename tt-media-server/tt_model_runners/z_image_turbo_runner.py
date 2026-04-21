# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import copy
import gc
import os
import sys
import time

import torch
import ttnn
from domain.image_generate_request import ImageGenerateRequest
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time

# Path to the Z-Image-Turbo TTNN model implementations.
# Fetched via: bash scripts/fetch_z_image_turbo.sh
_ZIT_DEMO_DIR = os.environ.get(
    "Z_IMAGE_TURBO_MODEL_DIR",
    os.path.join(
        os.path.dirname(__file__), "..", "models", "z_image_turbo_repo", "z_image_turbo"
    ),
)

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16
DEFAULT_STEPS = 9  # Runs N-1 DiT steps (8)

DRAM_RM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

WARMUP_TIMEOUT_SECONDS = 6000


def _ensure_zit_imports():
    if _ZIT_DEMO_DIR not in sys.path:
        sys.path.insert(0, _ZIT_DEMO_DIR)


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def _copy_to_persistent(host_pt, persistent_tt, dtype=ttnn.DataType.BFLOAT16):
    host_tt = ttnn.from_torch(
        host_pt.bfloat16() if dtype == ttnn.DataType.BFLOAT16 else host_pt,
        dtype=dtype,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    for shard in ttnn.get_device_tensors(persistent_tt):
        ttnn.copy_host_to_device_tensor(host_tt, shard, cq_id=0)


def _compute_mu(
    h=IMG_LATENT_H,
    w=IMG_LATENT_W,
    base_seq=256,
    max_seq=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    seq = (h // 2) * (w // 2)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq * m + (base_shift - m * base_seq)


def _make_scheduler(template, steps):
    scheduler = copy.deepcopy(template)
    mu = _compute_mu()
    try:
        scheduler.set_timesteps(steps, mu=mu)
    except TypeError:
        scheduler.set_timesteps(steps)
    return scheduler


class ZImageTurboRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.te = None
        self.tr = None
        self.vae = None
        self.tokenizer = None
        self.vae_processor = None
        self.scheduler_template = None
        self._trace_id = None
        self._lat_buf = None
        self._ts_buf = None
        self._output_ref = None

    def get_pipeline_device_params(self):
        return {
            "l1_small_size": 1 << 15,
            "trace_region_size": 60_000_000,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }

    def _configure_fabric(self, updated_device_params):
        fabric_config = updated_device_params.pop(
            "fabric_config", ttnn.FabricConfig.FABRIC_1D
        )
        ttnn.set_fabric_config(fabric_config)
        return fabric_config

    def _load_models(self):
        _ensure_zit_imports()
        from text_encoder.model_ttnn import TextEncoderTTNN
        from dit.model_ttnn import ZImageTransformerTTNN
        from vae.model_ttnn import VaeDecoderTTNN
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer

        mesh = self.ttnn_device
        mesh.enable_program_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        self.vae_processor = VaeImageProcessor(vae_scale_factor=16)
        self.scheduler_template = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler"
        )
        self.scheduler_template.sigma_min = 0.0

        self.te = TextEncoderTTNN(mesh)
        self.tr = ZImageTransformerTTNN(mesh)
        self.vae = VaeDecoderTTNN(mesh)

    def _encode_prompt(self, prompt):
        mesh = self.ttnn_device
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        input_ids = self.tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=CAP_TOKENS,
            return_tensors="pt",
        )["input_ids"]

        tt_ids = _to_device_int32(input_ids, mesh)
        tt_out = self.te(tt_ids)
        cap_cpu = _tt_to_torch(tt_out, mesh)[:CAP_TOKENS].bfloat16()
        ttnn.deallocate(tt_ids, force=True)
        ttnn.deallocate(tt_out, force=True)
        return cap_cpu.unsqueeze(0)

    def _decode_latents(self, latents):
        image_tensor = self.vae(latents)
        gc.collect()
        return self.vae_processor.postprocess(image_tensor, output_type="pil")[0]

    def _capture_trace(self):
        mesh = self.ttnn_device
        steps = DEFAULT_STEPS

        # Phase 1: Compile ALL programs before trace capture.
        self.logger.info(f"Device {self.device_id}: Compiling TE ...")
        cap_cpu = self._encode_prompt("a cat sitting on a mat")

        self.logger.info(f"Device {self.device_id}: Compiling DIT ...")
        self.tr.set_cap_feats(cap_cpu)
        torch.manual_seed(42)
        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)

        t0_step = scheduler.timesteps[0]
        t_norm_0 = max((1000.0 - float(t0_step)) / 1000.0, 1e-3)
        lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
        ts_pt = torch.tensor([t_norm_0], dtype=torch.bfloat16)

        self._lat_buf = _to_device_bf16(lat_pt, mesh)
        self._ts_buf = _to_device_bf16(ts_pt, mesh)

        compile_out = self.tr._forward_impl([self._lat_buf], self._ts_buf)
        dit_out_cpu = _tt_to_torch(compile_out[0], mesh)
        dit_out_cpu = dit_out_cpu.squeeze(1).unsqueeze(0).bfloat16()
        compile_latents = scheduler.step(
            -dit_out_cpu.float(), scheduler.timesteps[0], latents, return_dict=False
        )[0]
        for t in compile_out:
            ttnn.deallocate(t, force=True)

        self.logger.info(f"Device {self.device_id}: Compiling VAE ...")
        self._decode_latents(compile_latents)

        # Phase 2: Capture DIT trace.
        self.logger.info(f"Device {self.device_id}: Capturing DIT trace ...")
        self._trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_out = self.tr._forward_impl([self._lat_buf], self._ts_buf)
        ttnn.end_trace_capture(mesh, self._trace_id, cq_id=0)
        self._output_ref = trace_out[0]
        self.logger.info(f"Device {self.device_id}: DIT trace captured")

    @log_execution_time(
        "Z-Image-Turbo warmup",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading Z-Image-Turbo ...")

        def load_and_trace():
            self._load_models()
            self._capture_trace()

        await asyncio.wait_for(
            asyncio.to_thread(load_and_trace),
            timeout=WARMUP_TIMEOUT_SECONDS,
        )

        self.logger.info(f"Device {self.device_id}: Running warmup generation ...")
        self.run(
            [
                ImageGenerateRequest.model_construct(
                    prompt="a cat sitting on a mat",
                    num_inference_steps=DEFAULT_STEPS,
                    seed=42,
                )
            ]
        )

        self.logger.info(f"Device {self.device_id}: Z-Image-Turbo warmup complete")
        return True

    @log_execution_time(
        "Z-Image-Turbo inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[ImageGenerateRequest]):
        request = requests[0]
        mesh = self.ttnn_device
        steps = DEFAULT_STEPS
        seed = int(request.seed or 0)

        torch.manual_seed(seed)
        t_start = time.time()

        t0 = time.time()
        cap_cpu = self._encode_prompt(request.prompt)
        _copy_to_persistent(cap_cpu, self.tr._cap_feats_buf)
        te_ms = (time.time() - t0) * 1000

        latents = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
        scheduler = _make_scheduler(self.scheduler_template, steps)
        active_timesteps = scheduler.timesteps[:-1]

        t0 = time.time()
        for t in active_timesteps:
            t_norm = max((1000.0 - float(t)) / 1000.0, 1e-3)
            lat_pt = latents.squeeze(0).unsqueeze(1).bfloat16()
            ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)

            _copy_to_persistent(lat_pt, self._lat_buf)
            _copy_to_persistent(ts_pt, self._ts_buf)

            ttnn.execute_trace(mesh, self._trace_id, cq_id=0, blocking=True)

            out = _tt_to_torch(self._output_ref, mesh)
            out = out.squeeze(1).unsqueeze(0).bfloat16()
            latents = scheduler.step(-out.float(), t, latents, return_dict=False)[0]
        dit_ms = (time.time() - t0) * 1000

        t0 = time.time()
        image = self._decode_latents(latents)
        vae_ms = (time.time() - t0) * 1000

        elapsed = time.time() - t_start
        self.logger.info(
            f"Device {self.device_id}: Generated in {elapsed:.2f}s  "
            f"TE={te_ms:.0f}ms  DIT={dit_ms:.0f}ms ({len(active_timesteps)} steps)  "
            f"VAE={vae_ms:.0f}ms  seed={seed}"
        )
        return [image]
