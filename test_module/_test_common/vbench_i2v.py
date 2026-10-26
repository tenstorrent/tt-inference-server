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

camera_motion is left out by default: it only means something when the prompt
asks for a camera move. Informational: nothing gates on it yet.

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
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = REPO_ROOT / "test_fixtures" / "datasets" / "coco_i2v"
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
)
# VBench reports imaging_quality's aggregate on a 0-1 scale but each clip's
# score on MUSIQ's 0-100; per-clip scores here use the aggregate's scale.
PER_CLIP_SCALE = {"imaging_quality": 0.01}
# VBench's own RAFT fetch shells out to `unzip`, which CI hosts lack.
RAFT_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
# CLIP's weight download sometimes arrives corrupt; clip.load refetches on retry.
DOWNLOAD_ATTEMPTS = 3


def build_full_info(
    index: list[dict], videos_dir: Path, images_dir: Path, dimensions: list[str]
) -> list[dict]:
    """VBench's custom-input full-info list: one entry per saved clip."""
    entries = []
    for entry in index:
        record = {
            "prompt_en": entry["prompt"],
            "dimension": list(dimensions),
            "video_list": [str(videos_dir / entry["file"])],
        }
        if entry.get("conditioning_image"):
            record["custom_image_path"] = str(images_dir / entry["conditioning_image"])
        entries.append(record)
    return entries


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


def score_directory(
    videos_dir: Path,
    *,
    vbench_dir: Path,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    dimensions: tuple[str, ...] | list[str] = DEFAULT_DIMENSIONS,
    device: str | None = None,
    out: Path | None = None,
) -> dict:
    """Score every clip in a saved eval folder; one failing dimension skips one dimension.

    Writes the result after each dimension, so a run cut short keeps what finished.
    """
    import importlib

    import torch

    # Resolve before the chdir below.
    videos_dir, images_dir = videos_dir.resolve(), images_dir.resolve()
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
            }
            for e in index
        ],
        "dimensions": {},
    }

    def run_dimension(dimension: str) -> dict:
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
        clip_scores = list(record["per_clip"].values())
        low = f"{min(clip_scores):.4f}" if clip_scores else ""
        lines.append(
            f"| {name} | {record['score']:.4f} | {low} | {record['seconds']} |"
        )
    scored = [n for n, r in dims.items() if "error" not in r]
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
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--dimensions",
        default=",".join(DEFAULT_DIMENSIONS),
        help="comma-separated VBench dimensions",
    )
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    parser.add_argument("--out", type=Path, help="default: <videos-dir>/vbench.json")
    parser.add_argument("--summary-md", type=Path, help="append markdown tables here")
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

    result = score_directory(
        args.videos_dir,
        vbench_dir=args.vbench_dir,
        images_dir=args.images_dir,
        dimensions=[d.strip() for d in args.dimensions.split(",") if d.strip()],
        device=args.device,
        out=args.out.resolve() if args.out else None,
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
