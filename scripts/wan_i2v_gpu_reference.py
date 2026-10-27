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
video eval uses (``--pairs``: the five COCO pairs in test_fixtures/datasets/coco_i2v,
or the fifteen VBench-I2V pairs in .../vbench_i2v), with the request the eval
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
    uv run scripts/wan_i2v_gpu_reference.py --out gpu_ref_720p_vbench --pairs vbench_i2v
    python3 -m test_module._test_common.vbench_i2v --videos-dir gpu_ref_720p

A run resumes where it stopped: a clip whose record is in ``<out>/.clips`` is
not made again, so rerunning the same command after an interruption finishes
the set. ``--shard i/N`` splits a long set (the full VBench-I2V suite, from
scripts/build_vbench_i2v_full_pairs.py) across GPUs or hosts; shards may share
one folder, or their folders can be merged afterwards, ``.clips`` included:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i uv run scripts/wan_i2v_gpu_reference.py \\
        --out gpu_ref_480p_vbench_full --pairs vbench_i2v_full --resolution 480p \\
        --shard $i/4 &
    done
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "test_fixtures" / "datasets"
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
# Per-clip records the index is built from (see write_index).
CLIP_RECORDS = ".clips"
# Provenance that must match for clips to share a folder: everything that
# changes a clip. The GPU, versions and repo sha may differ between shards.
SETTINGS_KEYS = (
    "model",
    "pairs",
    "width",
    "height",
    "num_frames",
    "fps",
    "steps",
    "seed",
    "negative_prompt",
    "guidance_scale",
    "guidance_scale_2",
    "schedule",
    "flow_shift",
    "boundary_ratio",
)


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
        "pairs": args.pairs,
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


def pairs_dir_for(pairs: str) -> Path:
    """A pair set by name under test_fixtures/datasets, or a folder with a manifest.json."""
    folder = Path(pairs)
    return folder if (folder / "manifest.json").is_file() else DATASETS_DIR / pairs


def parse_shard(value: str) -> tuple[int, int]:
    """``i/N``: this run makes every Nth clip, starting at clip i (0-based)."""
    try:
        i, n = (int(part) for part in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected i/N, got {value!r}") from None
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"need 0 <= i < N, got {value!r}")
    return i, n


def check_same_settings(path: Path, provenance_now: dict) -> None:
    """Refuse to add clips to a folder made with other settings."""
    if not path.is_file():
        return
    before = json.loads(path.read_text())
    changed = {
        key: (before.get(key), provenance_now[key])
        for key in SETTINGS_KEYS
        if before.get(key) != provenance_now[key]
    }
    if changed:
        raise SystemExit(
            f"{path.parent} was made with other settings: {changed}. "
            "Use a new --out folder."
        )


def write_index(out: Path) -> None:
    """index.json from every finished clip's record, in pair order.

    Each clip's record is its own file, so shards sharing a folder never lose
    each other's clips; the index is replaced in one rename.
    """
    records = sorted(
        (json.loads(p.read_text()) for p in (out / CLIP_RECORDS).glob("*.json")),
        key=lambda r: r["file"],
    )
    staged = out / f"index.json.{os.getpid()}.tmp"
    staged.write_text(json.dumps(records, indent=2) + "\n")
    staged.replace(out / "index.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        default="coco_i2v",
        help="a pair set under test_fixtures/datasets (coco_i2v, vbench_i2v, "
        "vbench_i2v_full once built), or a folder holding a manifest.json",
    )
    parser.add_argument("--resolution", choices=sorted(RESOLUTIONS), default="720p")
    parser.add_argument("--schedule", choices=("tt", "checkpoint"), default="tt")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, help="first N pairs only")
    parser.add_argument(
        "--shard",
        type=parse_shard,
        default=(0, 1),
        help="i/N: make only clips i, i+N, i+2N, ... (e.g. one shard per GPU)",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="CPU-offload idle components; needed below ~80 GB of GPU memory",
    )
    args = parser.parse_args()

    import torch
    from diffusers.utils import export_to_video

    width, height = RESOLUTIONS[args.resolution]
    pairs_dir = pairs_dir_for(args.pairs)
    args.pairs = pairs_dir.name
    pairs = json.loads((pairs_dir / "manifest.json").read_text())[: args.limit]
    # Two digits for the small sets, as the committed references have.
    digits = max(2, len(str(len(pairs) - 1)))
    shard, shards = args.shard
    records = args.out / CLIP_RECORDS
    records.mkdir(parents=True, exist_ok=True)
    provenance_now = provenance(args, width, height)
    check_same_settings(args.out / "provenance.json", provenance_now)
    (args.out / "provenance.json").write_text(
        json.dumps(provenance_now, indent=2) + "\n"
    )

    # A clip is done once its record is written, after its video; anything
    # else is made (again), so a run can be stopped and restarted at will.
    todo = [
        (i, pair)
        for i, pair in enumerate(pairs)
        if i % shards == shard and not (records / f"{i:0{digits}d}_gpu.json").is_file()
    ]
    print(f"{len(todo)} clips to make in shard {shard}/{shards}", flush=True)
    if not todo:
        write_index(args.out)
        return 0
    pipe = load_pipeline(args.schedule, args.offload)

    for i, pair in todo:
        started = time.monotonic()
        frames = pipe(
            image=conditioning_image(pairs_dir / pair["image"], width, height),
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
        name = f"{i:0{digits}d}_gpu.mp4"
        export_to_video(frames, str(args.out / name), fps=FPS)
        seconds = round(time.monotonic() - started, 1)
        entry = {
            "file": name,
            "prompt": pair["caption"],
            "conditioning_image": pair["image"],
            "seconds": seconds,
        }
        if pair.get("camera_motion"):
            entry["camera_motion"] = pair["camera_motion"]
        (records / f"{Path(name).stem}.json").write_text(json.dumps(entry) + "\n")
        # Rewritten after every clip, so a run cut short is still scoreable.
        write_index(args.out)
        print(f"{name}: {seconds} s  {pair['caption']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
