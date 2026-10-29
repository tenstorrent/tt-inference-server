# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Score generated I2V clips with VBench's I2V and video-quality dimensions.

Each dimension is a fixed vision model rather than an LLM, so the same clips
score the same, and clips from another platform (the GPU references) can be
scored the same way and compared dimension by dimension:

- i2v_subject, i2v_background: the clip stays faithful to the conditioning
  image (DINO, DreamSim)
- subject_consistency, background_consistency: subjects and background stay
  the same across frames (DINO, CLIP)
- motion_smoothness, dynamic_degree: motion is smooth (AMT interpolation) and
  there is motion at all (RAFT optical flow)
- aesthetic_quality, imaging_quality, temporal_flickering: per-frame aesthetics
  (LAION), technical quality (MUSIQ) and flicker

- camera_motion: the camera makes the move the prompt asks for (CoTracker),
  scored only on clips whose index entry names one (``camera_motion``)

When the GPU reference run in the eval data on Hugging Face (GPU_REFERENCE_RUN,
or GPU_REFERENCE_RUN_CACHED for a deployment that caches, in I2V_EVAL_HF_REPO
at I2V_EVAL_HF_REVISION, made by its tools/wan_i2v_gpu_reference.py) has clips for
the same videos, matched by video id, each dimension is also reported against
them, and each clip is compared with its GPU clip frame by frame:

- PSNR, SSIM and LPIPS (AlexNet) of every frame to the GPU clip's same frame
- first-frame fidelity: frame 0 of the TT clip and of the GPU clip, each
  against the conditioning image
- initial noise: whether this host, from the clip's seed, redraws the noise
  the GPU run saved, bit for bit

Both sides draw the initial noise from the same seed on the host, so the clips
start out the same and the frame metrics show how far TT drifts from the GPU.

Informational: nothing gates on it yet.

The video eval runs it over the clips it saved
(``<run output>/videos/evals/index.json``) and puts the tables in the report.
It provisions the VBENCH_I2V venv (pinned deps plus a VBench checkout) and
re-runs itself there, so it also runs on its own:

    python -m test_module._test_common.vbench_i2v --videos-dir <dir>
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
# The I2V eval data too large for the repo: first frames (images/<W>x<H>/) and
# GPU reference runs (one folder per output shape: clips, index.json,
# provenance.json, vbench.json).
# Private to the Tenstorrent org, so reading it needs HF_TOKEN. The revision is
# pinned, so a new reference or image set reaches CI only through a commit here.
I2V_EVAL_HF_REPO = "Tenstorrent/wan2.2-i2v-vbench-refs"
I2V_EVAL_HF_REVISION = "821c6cb061c5d4109d10c467d54d478e5fa0095a"
# The GPU run TT is compared with: Wan2.2's I2V settings, Cache-DiT off.
GPU_REFERENCE_RUN = "refs/r1"
# The same with Cache-DiT on, set to TT's DBCache recipe: what a deployment
# that caches is compared with.
GPU_REFERENCE_RUN_CACHED = "refs/r2/tt"


def reference_metadata(run: str) -> list[str]:
    """A reference run's small files: enough to match clips before fetching them."""
    return [
        f"{run}/*/{name}" for name in ("index.json", "provenance.json", "vbench.json")
    ]


I2V_DIMENSIONS = ("i2v_subject", "i2v_background", "camera_motion")
DEFAULT_DIMENSIONS = (
    "i2v_subject",
    "i2v_background",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "camera_motion",
)
# VBench's camera_motion reads the expected move from the video's file name.
CAMERA_PHRASES = {
    "pan_left": "camera pans left",
    "pan_right": "camera pans right",
    "tilt_up": "camera tilts up",
    "tilt_down": "camera tilts down",
    "zoom_in": "camera zooms in",
    "zoom_out": "camera zooms out",
    "static": "camera static",
}
# VBench reports imaging_quality's aggregate on a 0-1 scale but each clip's
# score on MUSIQ's 0-100; per-clip scores here use the aggregate's scale.
PER_CLIP_SCALE = {"imaging_quality": 0.01}
# VBench's own RAFT fetch shells out to `unzip`, which CI hosts lack.
RAFT_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
# CLIP's weight download sometimes arrives corrupt; clip.load refetches on retry.
DOWNLOAD_ATTEMPTS = 3


def hf_snapshot(patterns: list[str] | tuple[str, ...]) -> Path:
    """The eval data's files matching ``patterns``, fetched once into the HF cache."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            I2V_EVAL_HF_REPO,
            repo_type="dataset",
            revision=I2V_EVAL_HF_REVISION,
            allow_patterns=list(patterns),
            token=os.environ.get("HF_TOKEN") or None,
        )
    )


def find_conditioning_image(
    name: str,
    images_dir: Path | None = None,
    shape: tuple[int, int] | None = None,
) -> Path:
    """The conditioning image: in ``images_dir`` if given, else in the eval data.

    The eval data holds each first frame prepared at every output size it is
    used at, so a clip finds its own by its ``shape``.
    """
    if images_dir:
        path = images_dir / name
    elif shape:
        relative = f"images/{shape[0]}x{shape[1]}/{name}"
        path = hf_snapshot([relative]) / relative
    else:
        raise ValueError(f"conditioning image {name}: no images dir and no shape")
    if not path.is_file():
        raise FileNotFoundError(f"conditioning image {name} not found at {path}")
    return path


def clip_shape(entry: dict) -> tuple[int, int] | None:
    """The ``(width, height)`` a clip's index entry names, if any."""
    if entry.get("width") and entry.get("height"):
        return (entry["width"], entry["height"])
    return None


def build_full_info(
    index: list[dict],
    videos_dir: Path,
    images_dir: Path | None,
    dimensions: list[str],
) -> list[dict]:
    """VBench's custom-input full-info list: one entry per saved clip.

    camera_motion is left out: it scores only some clips, under other names.
    """
    dimensions = [d for d in dimensions if d != "camera_motion"]
    entries = []
    for entry in index:
        record = {
            "prompt_en": entry["prompt"],
            "dimension": dimensions,
            "video_list": [str(videos_dir / entry["file"])],
        }
        if entry.get("conditioning_image"):
            record["custom_image_path"] = str(
                find_conditioning_image(
                    entry["conditioning_image"], images_dir, clip_shape(entry)
                )
            )
        entries.append(record)
    return entries


def link_camera_clips(index: list[dict], videos_dir: Path, links_dir: Path) -> dict:
    """Link each clip that asks for a camera move under a name VBench reads it from.

    Returns link name to the clip's own file name.
    """
    names = {}
    for entry in index:
        phrase = CAMERA_PHRASES.get(entry.get("camera_motion") or "")
        if not phrase:
            continue
        link = links_dir / f"{Path(entry['file']).stem} {phrase}.mp4"
        link.symlink_to((videos_dir / entry["file"]).resolve())
        names[link.name] = entry["file"]
    return names


def per_clip_scores(dimension: str, video_results: list[dict]) -> dict[str, float]:
    """File name to score, from a VBench dimension's per-video results."""
    scale = PER_CLIP_SCALE.get(dimension, 1.0)
    return {
        Path(r["video_path"]).name: round(float(r["video_results"]) * scale, 4)
        for r in video_results
        if isinstance(r.get("video_results"), (int, float))
    }


def is_corrupt_download(error: Exception) -> bool:
    """A weight download that failed its checksum, which a fresh fetch can fix."""
    return "checksum" in str(error).lower()


def ensure_raft_weights(cache_dir: Path) -> None:
    """Put RAFT where VBench looks for it, so it skips its own `unzip` fetch."""
    target = cache_dir / "raft_model" / "models" / "raft-things.pth"
    if target.is_file():
        return
    archive = cache_dir / "raft_model" / "models.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RAFT_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(archive.parent)
    archive.unlink()


def find_gpu_references(
    index: list[dict],
    run_dir: Path | None = None,
    hf_run: str = GPU_REFERENCE_RUN,
) -> dict[str, tuple[Path, str]]:
    """Each clip's GPU reference clip, by video id: ``{file: (folder, reference file)}``.

    Looks in ``run_dir`` (a reference run: one folder per shape, each with an
    index.json), or else in the eval data's ``hf_run`` on Hugging Face,
    fetching only the matched clips. Clips without a reference are left out.
    """
    if run_dir is None:
        metadata = hf_snapshot(reference_metadata(hf_run)) / hf_run
        matches = find_gpu_references(index, metadata)
        if matches:
            # The same snapshot, now holding the matched clips too.
            hf_snapshot([f"{hf_run}/{f.name}/{ref}" for f, ref in matches.values()])
        return matches
    by_video = {
        entry["video_id"]: (index_path.parent, entry["file"])
        for index_path in sorted(run_dir.glob("*/index.json"))
        for entry in json.loads(index_path.read_text())
        if entry.get("video_id")
    }
    return {
        entry["file"]: by_video[entry["video_id"]]
        for entry in index
        if entry.get("video_id") in by_video
    }


def read_frames(path: Path):
    """Each frame of a clip, as a uint8 HxWx3 array."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


# Identical frames have infinite PSNR, which JSON cannot hold.
PSNR_IDENTICAL_DB = 100.0


def psnr(frame, reference) -> float:
    """PSNR in dB between two uint8 frames of the same size."""
    import numpy as np

    error = np.mean((frame.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if error == 0:
        return PSNR_IDENTICAL_DB
    return min(PSNR_IDENTICAL_DB, float(10 * np.log10(255.0**2 / error)))


def ssim(frame, reference) -> float:
    """SSIM between two uint8 frames of the same size, over all three channels."""
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(frame, reference, channel_axis=-1, data_range=255)
    )


def lpips_scorer(device: str):
    """LPIPS (AlexNet) between two uint8 BGR frames; lower is closer."""
    import pyiqa
    import torch

    metric = pyiqa.create_metric("lpips", device=torch.device(device))

    def to_tensor(frame):
        rgb = torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1)
        return rgb[None].float().div(255).to(device)

    def score(frame, reference) -> float:
        with torch.no_grad():
            return float(metric(to_tensor(frame), to_tensor(reference)))

    return score


def summarize_frames(values: list[float], worst=min) -> dict:
    """A per-frame metric: frame 0, mean, worst frame and every frame."""
    return {
        "frame0": values[0],
        "mean": round(sum(values) / len(values), 4),
        "min" if worst is min else "max": worst(values),
        "frames": values,
    }


def frame_metrics(clip: Path, reference: Path, lpips=None) -> dict:
    """Each frame's PSNR, SSIM and (given a scorer) LPIPS to the reference clip's same frame."""
    psnrs, ssims, lpipses = [], [], []
    for a, b in zip(read_frames(clip), read_frames(reference)):
        psnrs.append(round(psnr(a, b), 2))
        ssims.append(round(ssim(a, b), 4))
        if lpips:
            lpipses.append(round(lpips(a, b), 4))
    if not psnrs:
        raise ValueError(f"no frames decoded from {clip.name} or {reference.name}")
    metrics = {"psnr": summarize_frames(psnrs), "ssim": summarize_frames(ssims)}
    if lpipses:
        metrics["lpips"] = summarize_frames(lpipses, worst=max)
    return metrics


def first_frame_fidelity(clip: Path, image: Path, lpips=None) -> dict:
    """Frame 0 against the conditioning image: PSNR, SSIM and (given a scorer) LPIPS."""
    import cv2

    frame = next(read_frames(clip), None)
    target = cv2.imread(str(image))
    if frame is None or target is None:
        raise ValueError(f"cannot read frame 0 of {clip.name} or {image.name}")
    if frame.shape != target.shape:
        raise ValueError(
            f"{clip.name} frame 0 is {frame.shape[1]}x{frame.shape[0]}, "
            f"{image.name} is {target.shape[1]}x{target.shape[0]}"
        )
    record = {
        "psnr": round(psnr(frame, target), 2),
        "ssim": round(ssim(frame, target), 4),
    }
    if lpips:
        record["lpips"] = round(lpips(frame, target), 4)
    return record


def redraws_saved_noise(seed: int, path: Path) -> dict:
    """Whether this host draws the GPU run's saved noise from ``seed``, bit for bit.

    The GPU script draws ``torch.randn`` on a CPU generator in float32; so does
    TT. A redraw can differ in the last bits on another CPU architecture.
    """
    import torch

    saved = torch.load(path, weights_only=True)
    drawn = torch.randn(
        saved.shape,
        generator=torch.Generator("cpu").manual_seed(seed),
        dtype=torch.float32,
    )
    return {
        "identical": bool(torch.equal(saved, drawn)),
        "max_abs_diff": float((saved - drawn).abs().max()),
        "torch": torch.__version__,
    }


def saved_noise_dir(index: list[dict], matches: dict) -> Path:
    """The eval data's noise/ with each matched clip's saved noise fetched."""
    patterns = [
        f"noise/{entry['width']}x{entry['height']}/{entry['video_id']}.pt"
        for entry in index
        if entry["file"] in matches and clip_shape(entry)
    ]
    return hf_snapshot(patterns) / "noise"


def describe_reference(folder: Path) -> str:
    """A reference folder as the report names it: its path in the repo or the HF dataset."""
    parts = folder.parts
    if "snapshots" in parts:  # <HF cache>/datasets--org--name/snapshots/<rev>/...
        inside = parts[parts.index("snapshots") + 2 :]
        return f"{I2V_EVAL_HF_REPO}/{'/'.join(inside)}"
    try:
        return str(folder.relative_to(REPO_ROOT))
    except ValueError:
        return str(folder)


def compare_to_gpu_reference(
    result: dict,
    videos_dir: Path,
    matches: dict[str, tuple[Path, str]],
    run_dir: Path,
    *,
    index: list[dict] = (),
    images_dir: Path | None = None,
    noise_dir: Path | None = None,
    device: str = "cpu",
) -> dict:
    """Each dimension against the matched GPU clips, and each clip against its GPU clip.

    Dimension means are over the clips both sides scored, so a dimension that
    scores only some clips (camera_motion) compares like with like. Per clip:
    PSNR, SSIM and LPIPS of every frame to the GPU clip's; frame 0 of both
    clips against the conditioning image; and whether the clip's seed redraws
    the GPU run's saved noise (in ``noise_dir``) on this host.
    """
    folders = {folder for folder, _ in matches.values()}
    scores = {f: json.loads((f / "vbench.json").read_text()) for f in folders}
    gpus = {json.loads((f / "provenance.json").read_text()).get("gpu") for f in folders}

    dimensions = {}
    for name, record in result["dimensions"].items():
        if "per_clip" not in record:
            continue
        shared = []
        for clip, (folder, ref) in matches.items():
            reference = scores[folder]["dimensions"].get(name, {}).get("per_clip", {})
            if clip in record["per_clip"] and ref in reference:
                shared.append((clip, record["per_clip"][clip], reference[ref]))
        if not shared:
            continue
        tt = [a for _, a, _ in shared]
        gpu = [b for _, _, b in shared]
        dimensions[name] = {
            "tt": round(sum(tt) / len(tt), 4),
            "gpu": round(sum(gpu) / len(gpu), 4),
            "delta": round((sum(tt) - sum(gpu)) / len(tt), 4),
            "clips": len(shared),
            "tt_lower": sum(a < b for a, b in zip(tt, gpu)),
            "tt_higher": sum(a > b for a, b in zip(tt, gpu)),
            "per_clip": {clip: round(a - b, 4) for clip, a, b in shared},
        }

    try:
        lpips = lpips_scorer(device)
        lpips_error = None
    except Exception as e:  # its weights are fetched; PSNR and SSIM need none
        logger.exception("LPIPS unavailable")
        lpips, lpips_error = None, f"{type(e).__name__}: {e}"[:500]

    entries = {entry["file"]: entry for entry in index}
    metrics = {"psnr": {}, "ssim": {}, "lpips": {}}
    first_frame, noise = {}, {}
    for clip, (folder, ref) in matches.items():
        try:
            for name, record in frame_metrics(
                videos_dir / clip, folder / ref, lpips
            ).items():
                metrics[name][clip] = record
        except Exception as e:  # one unreadable clip should not hide the rest
            logger.exception("Frame metrics of %s to the GPU reference failed", clip)
            metrics["psnr"][clip] = {"error": f"{type(e).__name__}: {e}"[:500]}

        entry = entries.get(clip, {})
        if entry.get("conditioning_image"):
            try:
                image = find_conditioning_image(
                    entry["conditioning_image"], images_dir, clip_shape(entry)
                )
                first_frame[clip] = {
                    "tt": first_frame_fidelity(videos_dir / clip, image, lpips),
                    "gpu": first_frame_fidelity(folder / ref, image, lpips),
                }
            except Exception as e:
                logger.exception("First-frame fidelity of %s failed", clip)
                first_frame[clip] = {"error": f"{type(e).__name__}: {e}"[:500]}

        if noise_dir is None or entry.get("seed") is None or not clip_shape(entry):
            continue
        path = (
            noise_dir
            / f"{entry['width']}x{entry['height']}"
            / f"{entry['video_id']}.pt"
        )
        try:
            if not path.is_file():
                noise[clip] = {"skipped": "no saved noise for this video"}
            else:
                noise[clip] = redraws_saved_noise(entry["seed"], path)
        except Exception as e:
            logger.exception("Noise check of %s failed", clip)
            noise[clip] = {"error": f"{type(e).__name__}: {e}"[:500]}

    comparison = {
        "reference": describe_reference(run_dir),
        "gpu": ", ".join(sorted(g for g in gpus if g)) or None,
        "clips": len(matches),
        "dimensions": dimensions,
        **metrics,
        "first_frame": first_frame,
        "noise": noise,
    }
    if lpips_error:
        comparison["lpips_error"] = lpips_error
    return comparison


def score_directory(
    videos_dir: Path,
    *,
    vbench_dir: Path,
    images_dir: Path | None = None,
    dimensions: tuple[str, ...] | list[str] = DEFAULT_DIMENSIONS,
    device: str | None = None,
    out: Path | None = None,
    gpu_reference: Path | str | None = "auto",
    gpu_reference_run: str = GPU_REFERENCE_RUN,
) -> dict:
    """Score every clip in a saved eval folder; one failing dimension skips one dimension.

    Writes the result after each dimension, so a run cut short keeps what finished.
    ``gpu_reference`` is a reference run to compare against (one folder per
    shape), "auto" for ``gpu_reference_run`` in the eval data, or None for no
    comparison.
    """
    import importlib

    import torch

    # Resolve before the chdir below.
    videos_dir = videos_dir.resolve()
    images_dir = images_dir.resolve() if images_dir else None
    out = (out or videos_dir / "vbench.json").resolve()
    sys.path.insert(0, str(vbench_dir.resolve()))
    # VBench resolves some third-party configs relative to its repo root.
    os.chdir(vbench_dir)
    from vbench2_beta_i2v.utils import CACHE_DIR, init_submodules

    index = json.loads((videos_dir / "index.json").read_text())
    info_path = out.with_name("vbench_full_info.json")
    info_path.write_text(
        json.dumps(build_full_info(index, videos_dir, images_dir, dimensions), indent=1)
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "device": device,
        "num_clips": len(index),
        "clips": [
            {
                "file": e["file"],
                "video_id": e.get("video_id"),
                "prompt": e["prompt"],
                "conditioning_image": e.get("conditioning_image"),
                "camera_motion": e.get("camera_motion"),
            }
            for e in index
        ],
        "dimensions": {},
    }

    def run_camera_motion() -> dict:
        links_dir = Path(tempfile.mkdtemp(prefix="vbench_camera_"))
        names = link_camera_clips(index, videos_dir, links_dir)
        if not names:
            return {"skipped": "no clip asks for a camera move"}
        camera_info = out.with_name("vbench_camera_info.json")
        camera_info.write_text(
            json.dumps(
                [
                    {
                        "prompt_en": link,
                        "dimension": ["camera_motion"],
                        "video_list": [str(links_dir / link)],
                    }
                    for link in names
                ],
                indent=1,
            )
        )
        submodules = init_submodules(["camera_motion"], local=False, read_frame=False)
        module = importlib.import_module("vbench2_beta_i2v.camera_motion")
        score, _, video_results = module.compute_camera_motion(
            str(camera_info), torch.device(device), submodules["camera_motion"]
        )
        return {
            "score": round(float(score), 4),
            "per_clip": {
                names[link]: value
                for link, value in per_clip_scores(
                    "camera_motion", video_results
                ).items()
            },
            "predicted": {
                names[Path(r["video_path"]).name]: r.get("predict_type")
                for r in video_results
            },
        }

    def run_dimension(dimension: str) -> dict:
        if dimension == "camera_motion":
            return run_camera_motion()
        package = "vbench2_beta_i2v" if dimension in I2V_DIMENSIONS else "vbench"
        if dimension == "dynamic_degree":
            ensure_raft_weights(Path(CACHE_DIR))
        # Per dimension: VBench fetches each one's models here, and one failed
        # fetch should cost only that dimension.
        submodules = init_submodules([dimension], local=False, read_frame=False)
        module = importlib.import_module(f"{package}.{dimension}")
        if dimension == "i2v_background":
            # VBench builds DreamSim without a device, and DreamSim defaults to cuda.
            module.dreamsim = functools.partial(module.dreamsim, device=device)
        score, video_results = getattr(module, f"compute_{dimension}")(
            str(info_path), torch.device(device), submodules[dimension]
        )
        return {
            "score": round(float(score), 4),
            "per_clip": per_clip_scores(dimension, video_results),
        }

    for dimension in dimensions:
        started = time.monotonic()
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                record = run_dimension(dimension)
                break
            except Exception as e:  # one broken dimension should not hide the rest
                if is_corrupt_download(e) and attempt < DOWNLOAD_ATTEMPTS:
                    logger.warning("VBench %s: %s; retrying", dimension, e)
                    continue
                logger.exception("VBench %s failed", dimension)
                record = {"error": f"{type(e).__name__}: {e}"[:500]}
                break
        record["seconds"] = round(time.monotonic() - started, 1)
        result["dimensions"][dimension] = record
        logger.info(
            "VBench %s: %s",
            dimension,
            {k: v for k, v in record.items() if k != "per_clip"},
        )
        out.write_text(json.dumps(result, indent=2) + "\n")

    if gpu_reference is None:
        return result
    run_dir = None if gpu_reference == "auto" else Path(gpu_reference)
    shown = (
        f"{I2V_EVAL_HF_REPO}/{gpu_reference_run}"
        if run_dir is None
        else describe_reference(run_dir)
    )
    try:
        matches = find_gpu_references(index, run_dir, gpu_reference_run)
        if matches:
            if run_dir is None:
                run_dir = (
                    hf_snapshot(reference_metadata(gpu_reference_run))
                    / gpu_reference_run
                )
                noise_dir = saved_noise_dir(index, matches)
            else:
                # The GPU script's own layout: <run>/noise/<WxH>/<video_id>.pt.
                noise_dir = run_dir / "noise"
            result["gpu_reference"] = compare_to_gpu_reference(
                result,
                videos_dir,
                matches,
                run_dir,
                index=index,
                images_dir=images_dir,
                noise_dir=noise_dir,
                device=device,
            )
        else:
            logger.info("No GPU reference for these clips; skipping the comparison")
    except Exception as e:  # e.g. no HF_TOKEN: the scores above stand
        logger.exception("Comparing with the GPU reference failed")
        result["gpu_reference"] = {
            "reference": shown,
            "error": f"{type(e).__name__}: {e}"[:500],
        }
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def to_markdown(result: dict, title: str) -> str:
    """Dimension means, then each clip's scores, as GitHub markdown tables."""
    dims = result["dimensions"]
    lines = [
        f"### VBench: {title}",
        "",
        f"{result['num_clips']} clips, scored on {result['device']}. Informational, nothing gates on it.",
        "",
        "| Dimension | Score | Min clip | Time (s) |",
        "|---|---|---|---|",
    ]
    for name, record in dims.items():
        if "error" in record:
            lines.append(
                f"| {name} | error: {record['error'][:80]} | | {record['seconds']} |"
            )
            continue
        if "skipped" in record:
            lines.append(f"| {name} | skipped: {record['skipped']} | | |")
            continue
        clip_scores = list(record["per_clip"].values())
        low = f"{min(clip_scores):.4f}" if clip_scores else ""
        lines.append(
            f"| {name} | {record['score']:.4f} | {low} | {record['seconds']} |"
        )
    scored = [n for n, r in dims.items() if "per_clip" in r]
    if scored:
        lines += ["", "| Clip | Prompt | " + " | ".join(scored) + " |"]
        lines.append("|---|---|" + "---|" * len(scored))
        for clip in result["clips"]:
            cells = [
                f"{dims[n]['per_clip'][clip['file']]:.3f}"
                if clip["file"] in dims[n]["per_clip"]
                else ""
                for n in scored
            ]
            lines.append(
                f"| {clip_label(clip)} | {clip['prompt']} | " + " | ".join(cells) + " |"
            )
    if result.get("gpu_reference"):
        lines += reference_markdown(result["gpu_reference"], result["clips"])
    return "\n".join(lines) + "\n"


def reference_markdown(reference: dict, clips: list[dict]) -> list[str]:
    """Each dimension against the GPU reference, then each clip's frame PSNR."""
    lines = ["", f"#### vs GPU reference: `{reference['reference']}`", ""]
    if "error" in reference:
        return lines + [f"Comparison failed: {reference['error'][:200]}"]
    lines += [
        (
            f"{reference['clips']} of these clips' requests on "
            f"{reference.get('gpu') or 'GPU'}, from the same initial noise."
        ),
        "",
        "| Dimension | TT | GPU | TT − GPU | Clips TT lower / higher |",
        "|---|---|---|---|---|",
    ]
    for name, d in reference["dimensions"].items():
        lines.append(
            f"| {name} | {d['tt']:.4f} | {d['gpu']:.4f} | {d['delta']:+.4f} "
            f"| {d['tt_lower']} / {d['tt_higher']} of {d['clips']} |"
        )
    lines += [
        "",
        "Each frame against the GPU clip's same frame. PSNR and SSIM: higher is "
        "closer; LPIPS: lower is closer. First frame: frame 0 against the "
        "conditioning image, TT / GPU.",
        "",
        "| Clip | Prompt | PSNR f0 / mean / min (dB) | SSIM mean / min "
        "| LPIPS mean / max | First frame PSNR TT / GPU | First frame SSIM TT / GPU "
        "| Noise |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for clip in clips:
        p = reference["psnr"].get(clip["file"])
        if p is None:
            continue
        if "error" in p:
            lines.append(
                f"| {clip_label(clip)} | {clip['prompt']} | error: {p['error'][:80]} "
                "| | | | | |"
            )
            continue
        s = reference.get("ssim", {}).get(clip["file"])
        lp = reference.get("lpips", {}).get(clip["file"])
        ff = reference.get("first_frame", {}).get(clip["file"], {})
        cells = [
            f"{p['frame0']:.1f} / {p['mean']:.1f} / {p['min']:.1f}",
            f"{s['mean']:.3f} / {s['min']:.3f}" if s else "",
            f"{lp['mean']:.3f} / {lp['max']:.3f}" if lp else "",
        ]
        if "tt" in ff:
            cells += [
                f"{ff['tt']['psnr']:.1f} / {ff['gpu']['psnr']:.1f}",
                f"{ff['tt']['ssim']:.3f} / {ff['gpu']['ssim']:.3f}",
            ]
        else:
            cells += [f"error: {ff['error'][:60]}" if "error" in ff else "", ""]
        cells.append(noise_cell(reference.get("noise", {}).get(clip["file"])))
        lines.append(
            f"| {clip_label(clip)} | {clip['prompt']} | " + " | ".join(cells) + " |"
        )
    if reference.get("lpips_error"):
        lines += ["", f"LPIPS unavailable: {reference['lpips_error'][:200]}"]
    noise = [n for n in reference.get("noise", {}).values() if "identical" in n]
    if noise:
        same = sum(n["identical"] for n in noise)
        lines += [
            "",
            f"Initial noise: {same} of {len(noise)} clips' seeds redraw the GPU run's "
            f"saved noise bit for bit on this host (torch {noise[0]['torch']}).",
        ]
    return lines


def clip_label(clip: dict) -> str:
    """A clip's video id (e.g. vbi2v_0913_832x480), or its index for a T2V clip."""
    return clip.get("video_id") or clip["file"][:2]


def noise_cell(record: dict | None) -> str:
    """One clip's noise check, as a table cell."""
    if not record:
        return ""
    if "identical" in record:
        return (
            "same" if record["identical"] else f"differs ({record['max_abs_diff']:.1e})"
        )
    return record.get("skipped") or f"error: {record.get('error', '')[:60]}"


def summarize(result: dict) -> dict:
    """Dimension name to score, for dimensions that ran."""
    return {
        name: record["score"]
        for name, record in result["dimensions"].items()
        if "score" in record
    }


def _provision_and_reexec(argv: list[str]) -> int:
    """Set up the VBENCH_I2V venv, then run this file under its python."""
    sys.path.insert(0, str(REPO_ROOT))
    from workflow_module.engine_types import WorkflowVenvType
    from workflows.bootstrap_uv import UV_EXEC, bootstrap_uv
    from workflows.workflow_venvs import VENV_CONFIGS

    # run.py bootstraps uv in CI; a standalone run (e.g. on a GPU box) may not have.
    if not UV_EXEC.exists():
        bootstrap_uv()
    venv = VENV_CONFIGS[WorkflowVenvType.VBENCH_I2V]
    venv.setup(model_spec=None)
    # Run the file, not the module: the venv lacks test_module's package imports.
    command = [str(venv.venv_python), str(Path(__file__).resolve()), *argv, "--in-venv"]
    command += ["--vbench-dir", str(venv.venv_path / "VBench")]
    os.execv(command[0], command)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="default: the eval data on Hugging Face, at each clip's shape",
    )
    parser.add_argument(
        "--dimensions",
        default=",".join(DEFAULT_DIMENSIONS),
        help="comma-separated VBench dimensions",
    )
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    parser.add_argument("--out", type=Path, help="default: <videos-dir>/vbench.json")
    parser.add_argument("--summary-md", type=Path, help="append markdown tables here")
    parser.add_argument(
        "--gpu-reference",
        default="auto",
        help="reference run to compare with (one folder per shape); 'auto' "
        "(default) is --gpu-reference-run in the eval data, 'none' skips it",
    )
    parser.add_argument(
        "--gpu-reference-run",
        default=GPU_REFERENCE_RUN,
        help=f"the eval data's reference run for 'auto' (default {GPU_REFERENCE_RUN}; "
        f"{GPU_REFERENCE_RUN_CACHED} for a deployment that caches)",
    )
    parser.add_argument("--vbench-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.videos_dir = args.videos_dir.resolve()

    if not (args.videos_dir / "index.json").is_file():
        print(f"VBench skipped: no index.json in {args.videos_dir}.")
        return 0
    if not args.in_venv:
        return _provision_and_reexec(argv)
    summary_md = args.summary_md.resolve() if args.summary_md else None
    gpu_reference = {"auto": "auto", "none": None}.get(
        args.gpu_reference, Path(args.gpu_reference).resolve()
    )

    result = score_directory(
        args.videos_dir,
        vbench_dir=args.vbench_dir,
        images_dir=args.images_dir,
        dimensions=[d.strip() for d in args.dimensions.split(",") if d.strip()],
        device=args.device,
        out=args.out.resolve() if args.out else None,
        gpu_reference=gpu_reference,
        gpu_reference_run=args.gpu_reference_run,
    )
    markdown = to_markdown(result, args.videos_dir.parent.parent.name)
    print(markdown)
    if args.summary_md:
        with open(summary_md, "a") as f:
            f.write(markdown + "\n")
    # Scored nothing at all (e.g. no model could be fetched): say so in the step.
    return 0 if summarize(result) else 1


if __name__ == "__main__":
    sys.exit(main())
