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

When the GPU reference run in the eval data on Hugging Face (GPU_REFERENCE_RUN
in I2V_EVAL_HF_REPO at I2V_EVAL_HF_REVISION, from
scripts/wan_i2v_gpu_reference.py) has clips for the same videos, matched by
video id, each dimension is also reported against them, with each frame's PSNR
to the GPU clip. Both sides draw the initial noise from the same seed on the host, so the clips
start out the same and the PSNR shows how far TT drifts from the GPU.

Informational: nothing gates on it yet.

Runs as its own CI step after the tests, over the clips the video eval saved
(``<run output>/videos/evals/index.json``). It provisions the VBENCH_I2V venv
(pinned deps plus a VBench checkout) and re-runs itself there:

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
I2V_EVAL_HF_REVISION = "51206b33c4877cbfec4cf06e70478666ead6e1de"
# The GPU run TT is compared with: Wan2.2's I2V settings, Cache-DiT off.
GPU_REFERENCE_RUN = "refs/r1"
# A reference run's small files: enough to match clips before fetching them.
REFERENCE_METADATA = tuple(
    f"{GPU_REFERENCE_RUN}/*/{name}"
    for name in ("index.json", "provenance.json", "vbench.json")
)
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
    index: list[dict], run_dir: Path | None = None
) -> dict[str, tuple[Path, str]]:
    """Each clip's GPU reference clip, by video id: ``{file: (folder, reference file)}``.

    Looks in ``run_dir`` (a reference run: one folder per shape, each with an
    index.json), or else in the eval data's GPU_REFERENCE_RUN on Hugging Face,
    fetching only the matched clips. Clips without a reference are left out.
    """
    if run_dir is None:
        metadata = hf_snapshot(REFERENCE_METADATA) / GPU_REFERENCE_RUN
        matches = find_gpu_references(index, metadata)
        if matches:
            # The same snapshot, now holding the matched clips too.
            hf_snapshot(
                [f"{GPU_REFERENCE_RUN}/{f.name}/{ref}" for f, ref in matches.values()]
            )
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


def frame_psnr(clip: Path, reference: Path) -> dict:
    """Each frame's PSNR to the reference clip's same frame, with frame 0, mean and min."""
    frames = [
        round(psnr(a, b), 2) for a, b in zip(read_frames(clip), read_frames(reference))
    ]
    if not frames:
        raise ValueError(f"no frames decoded from {clip.name} or {reference.name}")
    return {
        "frame0": frames[0],
        "mean": round(sum(frames) / len(frames), 2),
        "min": min(frames),
        "frames": frames,
    }


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
) -> dict:
    """Each dimension against the matched GPU clips, and each clip's frame PSNR.

    Dimension means are over the clips both sides scored, so a dimension that
    scores only some clips (camera_motion) compares like with like.
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

    frames = {}
    for clip, (folder, ref) in matches.items():
        try:
            frames[clip] = frame_psnr(videos_dir / clip, folder / ref)
        except Exception as e:  # one unreadable clip should not hide the rest
            logger.exception("PSNR of %s to the GPU reference failed", clip)
            frames[clip] = {"error": f"{type(e).__name__}: {e}"[:500]}

    return {
        "reference": describe_reference(run_dir),
        "gpu": ", ".join(sorted(g for g in gpus if g)) or None,
        "clips": len(matches),
        "dimensions": dimensions,
        "psnr": frames,
    }


def score_directory(
    videos_dir: Path,
    *,
    vbench_dir: Path,
    images_dir: Path | None = None,
    dimensions: tuple[str, ...] | list[str] = DEFAULT_DIMENSIONS,
    device: str | None = None,
    out: Path | None = None,
    gpu_reference: Path | str | None = "auto",
) -> dict:
    """Score every clip in a saved eval folder; one failing dimension skips one dimension.

    Writes the result after each dimension, so a run cut short keeps what finished.
    ``gpu_reference`` is a reference run to compare against (one folder per
    shape), "auto" for GPU_REFERENCE_RUN in the eval data, or None for no
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
        f"{I2V_EVAL_HF_REPO}/{GPU_REFERENCE_RUN}"
        if run_dir is None
        else describe_reference(run_dir)
    )
    try:
        matches = find_gpu_references(index, run_dir)
        if matches:
            if run_dir is None:
                run_dir = hf_snapshot(REFERENCE_METADATA) / GPU_REFERENCE_RUN
            result["gpu_reference"] = compare_to_gpu_reference(
                result, videos_dir, matches, run_dir
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
                f"| {clip['file'][:2]} | {clip['prompt']} | " + " | ".join(cells) + " |"
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
        "| Clip | Prompt | PSNR frame 0 (dB) | Mean | Min |",
        "|---|---|---|---|---|",
    ]
    for clip in clips:
        p = reference["psnr"].get(clip["file"])
        if p is None:
            continue
        if "error" in p:
            cells = f"error: {p['error'][:80]} | | "
        else:
            cells = f"{p['frame0']:.1f} | {p['mean']:.1f} | {p['min']:.1f}"
        lines.append(f"| {clip['file'][:2]} | {clip['prompt']} | {cells} |")
    return lines


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
        f"(default) is {GPU_REFERENCE_RUN} in the eval data, 'none' skips it",
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
