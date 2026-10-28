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

"""Generate GPU reference clips for the Wan2.2 I2V eval with upstream diffusers.

Runs Wan2.2-I2V-A14B on a CUDA GPU over the I2V eval's validation set: 65
videos over 50 VBench-I2V pairs at four shapes
(scripts/select_vbench_i2v_pairs.py). ``--data`` is the folder with its
``pairs/manifest.json`` and prepared ``images/`` (the eval data on Hugging
Face). Each video has its own shape and seed; ``--subset dev|ci`` and
``--shape`` pick some of them.

Every clip is 81 frames at 16 fps with 40 steps. ``--schedule`` sets the
sampler and guidance:

- ``wan`` (default): Wan2.2's own I2V settings (wan/configs/wan_i2v_A14B.py):
  UniPC flow_shift 5.0, expert boundary 0.900, guidance 3.5 on both experts
- ``tt``: what TT runs today: flow_shift 12.0, boundary 0.875, guidance 4.0 on
  the high-noise expert and 3.0 on the low-noise one (Wan2.2's T2V settings)

The negative prompt defaults to Wan2.2's own (``sample_neg_prompt``), which is
what the generation API spec serves when a request leaves it out.

The initial noise is drawn from torch's CPU generator in float32, as TT draws
it, passed to the pipeline, and saved, one tensor per video. ``--noise-dir``
loads each video's saved tensor instead (``<dir>/<WxH>/<video_id>.pt``, e.g.
an earlier run's ``noise/``), so a later run starts from exactly the same
latents on any host. The output is the layout the video eval saves, so the
VBench scorer runs on it unchanged:

    uv run scripts/wan_i2v_gpu_reference.py --data hf_upload --out refs/r1 --subset dev
    python3 -m test_module._test_common.vbench_i2v --videos-dir refs/r1/832x480 \\
        --images-dir hf_upload/images/832x480 --gpu-reference none

``--cache-dit PRESET`` turns on cache-dit's DBCache on both experts
(``uv run --with cache-dit==1.5.3 ...``): a step whose first blocks' residual
changes less than the threshold reuses the cached rest of the transformer.
The presets are TT's own recipes (``tt``: tt-metal's I2V default; ``tt-quality``:
its higher-fidelity setting), so a GPU run caches the way TT does. All the
settings are in provenance.json. Each clip's record lists the steps each expert
took from the cache, with the residual diffs:

    uv run --with cache-dit==1.5.3 scripts/wan_i2v_gpu_reference.py \\
        --data hf_upload --out refs/r2/tt --cache-dit tt --noise-dir hf_upload/noise

A run resumes where it stopped: a clip whose record is in its folder's
``.clips`` is not made again, so rerunning the same command after an
interruption finishes the set. ``--shard i/N`` splits the set across GPUs or
hosts; shards may share one output folder:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i uv run scripts/wan_i2v_gpu_reference.py \\
        --data hf_upload --out refs/r1 --shard $i/4 &
    done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Recorded in each folder's provenance, so folders from earlier runs still resume.
PAIR_SET = "vbench_i2v_50"
# Per-clip records each output folder's index is built from (see write_index).
CLIP_RECORDS = ".clips"
MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
NUM_FRAMES = 81
FPS = 16
STEPS = 40
# Wan2.2's sample_neg_prompt (wan/configs/shared_config.py), the API spec's default.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)
# guidance_scale drives the high-noise expert, guidance_scale_2 the low-noise one.
SCHEDULES = {
    "wan": {
        "flow_shift": 5.0,
        "boundary_ratio": 0.900,
        "guidance_scale": 3.5,
        "guidance_scale_2": 3.5,
    },
    "tt": {
        "flow_shift": 12.0,
        "boundary_ratio": 0.875,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
    },
}
# TT's DBCache recipes as cache-dit settings: tt-metal 9b40b665
# models/tt_dit/pipelines/wan/dbcache.py, WanDBCacheConfig.default() and its
# documented quality override (threshold 0.06, no TaylorSeer, warmup interval 1).
# taylorseer_order 0 runs without the TaylorSeer calibrator.
CACHE_DIT_PRESETS = {
    "tt": {
        "residual_diff_threshold": 0.12,
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_continuous_cached_steps": 2,
        "warmup_interval": 2,
        "cfg_diff_compute_separate": True,
        "taylorseer_order": 1,
    },
    "tt-quality": {
        "residual_diff_threshold": 0.06,
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_continuous_cached_steps": 2,
        "warmup_interval": 1,
        "cfg_diff_compute_separate": True,
        "taylorseer_order": 0,
    },
}
# Each expert's own warmup and cached-step budget, the same in both presets.
CACHE_DIT_EXPERT_BUDGETS = {
    "high_noise": {"max_warmup_steps": 4, "max_cached_steps": 8},
    "low_noise": {"max_warmup_steps": 2, "max_cached_steps": 20},
}
# diffusers' Wan2.2 experts: transformer takes the high-noise steps. cache-dit's
# Wan2.2 adapter caches them in this order too.
EXPERTS = (("high_noise", "transformer"), ("low_noise", "transformer_2"))


def load_pipeline(schedule: str, offload: bool, cache_preset: str | None):
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
    settings = SCHEDULES[schedule]
    # The checkpoint ships flow_shift 3.0 and boundary 0.9; both are set here.
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=settings["flow_shift"]
    )
    pipe.register_to_config(boundary_ratio=settings["boundary_ratio"])
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if cache_preset is not None:
        enable_cache_dit(pipe, cache_preset)
    return pipe


def enable_cache_dit(pipe, preset: str) -> None:
    import cache_dit
    from cache_dit import DBCacheConfig, ParamsModifier, TaylorSeerCalibratorConfig

    shared = dict(CACHE_DIT_PRESETS[preset])
    order = shared.pop("taylorseer_order")
    cache_dit.enable_cache(
        pipe,
        cache_config=DBCacheConfig(**shared),
        calibrator_config=(
            TaylorSeerCalibratorConfig(taylorseer_order=order) if order else None
        ),
        # reset() leaves every other field None, so a modifier changes only the
        # expert's budget and keeps the shared settings.
        params_modifiers=[
            ParamsModifier(
                cache_config=DBCacheConfig().reset(**CACHE_DIT_EXPERT_BUDGETS[expert])
            )
            for expert, _ in EXPERTS
        ],
    )


def cache_dit_settings(preset: str | None) -> dict | None:
    if preset is None:
        return None
    return {"preset": preset, **CACHE_DIT_PRESETS[preset], **CACHE_DIT_EXPERT_BUDGETS}


def cache_log(pipe) -> dict:
    """The steps each expert took from the cache in the last call, with their residual diffs.

    Steps count from each expert's own first step. cache-dit resets its context
    at the start of every pipeline call, so this is one clip's log.
    """
    import cache_dit

    log = {}
    for expert, name in EXPERTS:
        stats = cache_dit.summary(getattr(pipe, name), logging=False)[0]
        log[expert] = {
            "cached_steps": list(stats.cached_steps),
            "cfg_cached_steps": list(stats.cfg_cached_steps),
            "residual_diffs": dict(stats.residual_diffs),
            "cfg_residual_diffs": dict(stats.cfg_residual_diffs),
        }
    return log


def conditioning_image(path: Path, width: int, height: int):
    """The prepared first frame, which is already at the canvas size."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if image.size != (width, height):
        raise SystemExit(f"{path} is {image.size}, expected {(width, height)}")
    return image


def noise_shape(pipe, width: int, height: int) -> tuple[int, ...]:
    return (
        1,
        pipe.vae.config.z_dim,
        (NUM_FRAMES - 1) // pipe.vae_scale_factor_temporal + 1,
        height // pipe.vae_scale_factor_spatial,
        width // pipe.vae_scale_factor_spatial,
    )


def initial_noise(pipe, seed: int, width: int, height: int):
    """The latent noise diffusers would draw from a CPU generator with ``seed``."""
    import torch

    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(
        noise_shape(pipe, width, height), generator=generator, dtype=torch.float32
    )


def saved_noise(pipe, path: Path, width: int, height: int):
    """A video's noise saved by an earlier run, checked to fit its shape."""
    import torch

    noise = torch.load(path)
    expected = noise_shape(pipe, width, height)
    if tuple(noise.shape) != expected or noise.dtype != torch.float32:
        raise SystemExit(
            f"{path} is {noise.dtype} {tuple(noise.shape)}, expected float32 {expected}"
        )
    return noise


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_shard(value: str) -> tuple[int, int]:
    """``i/N``: this run makes every Nth clip, starting at clip i (0-based)."""
    try:
        i, n = (int(part) for part in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected i/N, got {value!r}") from None
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"need 0 <= i < N, got {value!r}")
    return i, n


def jobs(args) -> list[dict]:
    """One job per clip in this shard: what to generate and where it goes."""
    shard, shards = args.shard
    return [job for i, job in enumerate(all_jobs(args)) if i % shards == shard]


def all_jobs(args) -> list[dict]:
    manifest = json.loads((args.data / "pairs" / "manifest.json").read_text())
    shapes = set(args.shape or [])
    picked = [
        e
        for e in manifest
        if (args.subset == "all" or e[args.subset])
        and (not shapes or f"{e['width']}x{e['height']}" in shapes)
    ][: args.limit]
    out = []
    for e in picked:
        image_path = args.data / e["image"]
        if sha256_file(image_path) != e["image_sha256"]:
            raise SystemExit(f"{image_path} does not match the manifest's sha256")
        shape = f"{e['width']}x{e['height']}"
        saved = (
            args.noise_dir / shape / f"{e['video_id']}.pt" if args.noise_dir else None
        )
        if saved and not saved.is_file():
            raise SystemExit(f"no saved noise at {saved}")
        out.append(
            {
                "file": f"{e['video_id']}.mp4",
                "out_dir": args.out / shape,
                "noise_path": args.out / "noise" / shape / f"{e['video_id']}.pt",
                "saved_noise": saved,
                "image_path": image_path,
                "conditioning_image": Path(e["image"]).name,
                "caption": e["caption"],
                "camera_motion": e.get("camera_motion"),
                "width": e["width"],
                "height": e["height"],
                "seed": e["seed"],
                "video_id": e["video_id"],
                "pair_id": e["pair_id"],
            }
        )
    return out


def provenance(args, width: int, height: int) -> dict:
    from importlib.metadata import PackageNotFoundError, version

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
    record = {
        "model": MODEL,
        "pairs": PAIR_SET,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "tt_inference_server_sha": sha,
        "script_sha256": sha256_file(Path(__file__)),
        "width": width,
        "height": height,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "steps": args.steps,
        "negative_prompt": args.negative_prompt,
        "schedule": args.schedule,
        **SCHEDULES[args.schedule],
        "noise": "torch.randn on a CPU generator seeded per clip, float32",
        "noise_from": str(args.noise_dir) if args.noise_dir else None,
        "cpu_offload": args.offload,
        "cache_dit": cache_dit_settings(args.cache_dit),
    }
    if args.cache_dit is not None:
        record["cache_dit_version"] = version("cache-dit")
    try:
        record["transformers"] = version("transformers")
    except PackageNotFoundError:
        pass
    record["manifest_sha256"] = sha256_file(args.data / "pairs" / "manifest.json")
    return record


# Settings every clip in one folder must share; resuming or adding a shard with
# others is refused. The GPU, versions and repo sha may differ between shards.
GENERATION_SETTINGS = (
    "model",
    "pairs",
    "width",
    "height",
    "fps",
    "seed",
    "steps",
    "negative_prompt",
    "flow_shift",
    "boundary_ratio",
    "guidance_scale",
    "guidance_scale_2",
    "num_frames",
    "manifest_sha256",
    "cache_dit",
)


def check_resumable(path: Path, record: dict) -> None:
    """Stop if ``path`` records clips made with other generation settings."""
    if not path.exists():
        return
    before = json.loads(path.read_text())
    changed = [k for k in GENERATION_SETTINGS if before.get(k) != record.get(k)]
    if changed:
        raise SystemExit(
            f"{path.parent} holds clips made with other {', '.join(changed)}; "
            "use a new --out"
        )


def clip_records(out_dir: Path) -> dict[str, dict]:
    """The finished clips in ``out_dir``, by file.

    Each clip has its own record, so shards sharing a folder never lose each
    other's clips. A folder made before records existed has only its
    index.json; its clips are recorded from that first.
    """
    records = out_dir / CLIP_RECORDS
    if not records.is_dir():
        records.mkdir(parents=True)
        index_path = out_dir / "index.json"
        for entry in json.loads(index_path.read_text()) if index_path.exists() else []:
            if (out_dir / entry["file"]).exists():
                write_record(out_dir, entry)
    return {
        entry["file"]: entry
        for entry in (json.loads(p.read_text()) for p in records.glob("*.json"))
    }


def write_record(out_dir: Path, entry: dict) -> None:
    path = out_dir / CLIP_RECORDS / f"{Path(entry['file']).stem}.json"
    path.write_text(json.dumps(entry) + "\n")


def write_index(out_dir: Path) -> None:
    """index.json from every finished clip's record, in file order, replaced in one rename."""
    records = sorted(clip_records(out_dir).values(), key=lambda e: e["file"])
    staged = out_dir / f"index.json.{os.getpid()}.tmp"
    staged.write_text(json.dumps(records, indent=2) + "\n")
    staged.replace(out_dir / "index.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="folder with pairs/manifest.json and images/ (the eval data)",
    )
    parser.add_argument("--subset", choices=("all", "ci", "dev"), default="all")
    parser.add_argument(
        "--shape", action="append", help="only this WxH (e.g. 832x480); repeatable"
    )
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), default="wan")
    parser.add_argument("--negative-prompt", default=WAN_NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--limit", type=int, help="first N clips only")
    parser.add_argument(
        "--cache-dit",
        choices=sorted(CACHE_DIT_PRESETS),
        help="cache-dit DBCache with TT's recipe: tt (its I2V default) or tt-quality",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        help="load each video's noise from <dir>/<WxH>/<video_id>.pt instead of drawing it",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check the data and list the clips, without loading the model",
    )
    args = parser.parse_args()

    todo = jobs(args)
    shard, shards = args.shard
    print(f"{len(todo)} clips in shard {shard}/{shards}", flush=True)
    if args.dry_run:
        for job in todo:
            print(
                f"{job['out_dir'] / job['file']}  seed {job['seed']}  {job['caption']}"
            )
        return 0

    from diffusers.utils import export_to_video

    settings = SCHEDULES[args.schedule]
    pipe = None
    done: dict[Path, dict[str, dict]] = {}
    for job in todo:
        out_dir = job["out_dir"]
        if out_dir not in done:
            out_dir.mkdir(parents=True, exist_ok=True)
            record = provenance(args, job["width"], job["height"])
            check_resumable(out_dir / "provenance.json", record)
            (out_dir / "provenance.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            done[out_dir] = clip_records(out_dir)
        # A clip is done once its record is written, after its video.
        if job["file"] in done[out_dir] and (out_dir / job["file"]).exists():
            print(f"{job['file']}: already made, skipping", flush=True)
            continue
        if pipe is None:
            pipe = load_pipeline(args.schedule, args.offload, args.cache_dit)

        started = time.monotonic()
        if job["saved_noise"]:
            noise = saved_noise(pipe, job["saved_noise"], job["width"], job["height"])
        else:
            noise = initial_noise(pipe, job["seed"], job["width"], job["height"])
        frames = pipe(
            image=conditioning_image(job["image_path"], job["width"], job["height"]),
            prompt=job["caption"],
            negative_prompt=args.negative_prompt,
            height=job["height"],
            width=job["width"],
            num_frames=NUM_FRAMES,
            num_inference_steps=args.steps,
            guidance_scale=settings["guidance_scale"],
            guidance_scale_2=settings["guidance_scale_2"],
            latents=noise,
        ).frames[0]
        export_to_video(frames, str(out_dir / job["file"]), fps=FPS)
        seconds = round(time.monotonic() - started, 1)

        entry = {
            "file": job["file"],
            "prompt": job["caption"],
            "conditioning_image": job["conditioning_image"],
            "seconds": seconds,
            "seed": job["seed"],
            "noise_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
        }
        if job.get("video_id"):
            entry.update(
                video_id=job["video_id"],
                pair_id=job["pair_id"],
                width=job["width"],
                height=job["height"],
            )
        if job.get("camera_motion"):
            entry["camera_motion"] = job["camera_motion"]
        if args.cache_dit is not None:
            entry["cache"] = cache_log(pipe)
        # Noise loaded from an earlier run is already saved there.
        if job.get("noise_path") and not job["saved_noise"]:
            import torch

            job["noise_path"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(noise, job["noise_path"])
        write_record(out_dir, entry)
        # Rewritten after every clip, so a run cut short is still scoreable.
        write_index(out_dir)
        print(f"{job['file']}: {seconds} s  {job['caption']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
