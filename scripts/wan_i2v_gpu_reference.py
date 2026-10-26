#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.5",
#     "torchvision",
#     "diffusers==0.39.0",
#     "transformers>=4.51,<5",
#     "accelerate",
#     "ftfy",
#     "imageio",
#     "imageio-ffmpeg",
#     "pillow",
#     "numpy",
# ]
# ///

"""Generate GPU reference clips for the Wan2.2 I2V eval, like-for-like with TT.

Runs upstream diffusers on a CUDA GPU over the same image-caption pairs the TT
video eval uses (test_fixtures/datasets/coco_i2v), with the request the eval
sends and the settings the TT pipeline runs (tt-media-server dit_runners.py,
tt-metal models/tt_dit/pipelines/wan):

- 81 frames at 16 fps, 40 steps, seed 42, the eval's negative prompt
- guidance 4.0 on the high-noise expert and 3.0 on the low-noise one
- the conditioning image resized to the canvas with bicubic antialiasing
- ``--schedule tt`` (default): UniPC flow_shift 12.0 and expert boundary 0.875,
  which is what TT runs. ``--schedule checkpoint`` keeps the Hugging Face
  checkpoint's own I2V values (flow_shift 3.0, boundary 0.9) instead.

It writes the clips and an index.json in the layout the video eval saves, so
the VBench scorer runs on them unchanged:

    uv run scripts/wan_i2v_gpu_reference.py --out gpu_ref_720p --resolution 720p
    python3 -m test_module._test_common.vbench_i2v --videos-dir gpu_ref_720p
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS_DIR = REPO_ROOT / "test_fixtures" / "datasets" / "coco_i2v"
MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
RESOLUTIONS = {"720p": (1280, 720), "480p": (832, 480)}
NUM_FRAMES = 81
FPS = 16
# What the TT eval sends (test_module/_test_common/video_generation_routing.py).
STEPS = 40
SEED = 42
NEGATIVE_PROMPT = "blurry, low quality, distorted"
GUIDANCE_SCALE = 4.0
GUIDANCE_SCALE_2 = 3.0
# What the TT pipeline runs; the checkpoint ships 3.0 and 0.9 for I2V.
TT_FLOW_SHIFT = 12.0
TT_BOUNDARY_RATIO = 0.875


def load_pipeline(schedule: str, offload: bool):
    import torch
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanImageToVideoPipeline,
    )

    vae = AutoencoderKLWan.from_pretrained(
        MODEL, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, vae=vae, torch_dtype=torch.bfloat16
    )
    if schedule == "tt":
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=TT_FLOW_SHIFT
        )
        pipe.register_to_config(boundary_ratio=TT_BOUNDARY_RATIO)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def conditioning_image(path: Path, width: int, height: int):
    """The image as TT feeds it: resized to the canvas, bicubic with antialiasing."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    tensor = TF.to_tensor(Image.open(path).convert("RGB"))[None]
    resized = torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bicubic", antialias=True
    )
    return TF.to_pil_image(resized[0].clamp(0, 1))


def provenance(args, width: int, height: int) -> dict:
    import diffusers
    import torch

    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "tt_inference_server_sha": sha,
        "width": width,
        "height": height,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "steps": args.steps,
        "seed": args.seed,
        "negative_prompt": NEGATIVE_PROMPT,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_scale_2": GUIDANCE_SCALE_2,
        "schedule": args.schedule,
        "flow_shift": TT_FLOW_SHIFT if args.schedule == "tt" else "checkpoint",
        "boundary_ratio": TT_BOUNDARY_RATIO if args.schedule == "tt" else "checkpoint",
        "cpu_offload": args.offload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="720p")
    parser.add_argument("--schedule", choices=("tt", "checkpoint"), default="tt")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, help="first N pairs only")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="CPU-offload idle components; needed below ~80 GB of GPU memory",
    )
    args = parser.parse_args()

    import torch
    from diffusers.utils import export_to_video

    width, height = RESOLUTIONS[args.resolution]
    pairs = json.loads((PAIRS_DIR / "manifest.json").read_text())[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.json").write_text(
        json.dumps(provenance(args, width, height), indent=2) + "\n"
    )
    pipe = load_pipeline(args.schedule, args.offload)

    index = []
    for i, pair in enumerate(pairs):
        started = time.monotonic()
        frames = pipe(
            image=conditioning_image(PAIRS_DIR / pair["image"], width, height),
            prompt=pair["caption"],
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            num_inference_steps=args.steps,
            guidance_scale=GUIDANCE_SCALE,
            guidance_scale_2=GUIDANCE_SCALE_2,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).frames[0]
        name = f"{i:02d}_gpu.mp4"
        export_to_video(frames, str(args.out / name), fps=FPS)
        seconds = round(time.monotonic() - started, 1)
        index.append(
            {
                "file": name,
                "prompt": pair["caption"],
                "conditioning_image": pair["image"],
                "seconds": seconds,
            }
        )
        # Rewritten after every clip, so a run cut short is still scoreable.
        (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        print(f"{name}: {seconds} s  {pair['caption']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
