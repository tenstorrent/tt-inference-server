# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image models' eval data: cases, prepared inputs and GPU references on Hugging Face.

One private dataset per model, pinned to a revision. Each holds
``cases/manifest.json`` (one entry per image to generate, with its prompt,
inputs, ``aspect_ratio``, ``size``, seed, and ``dev`` and ``ci`` flags),
``inputs/`` and the GPU runs under ``refs/`` (``refs/r1/<case_id>.png``); the
dataset's README says how each was made. The run's eval scope picks the
subset, as for Wan2.2 I2V: dev under ``--limit-samples-mode smoke-test``, ci
under ``--ci-mode`` and the other CI modes, all on a full-evals run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# model_id -> (Hugging Face dataset, pinned revision)
EVAL_DATA = {
    "black-forest-labs/flux.2-klein-4b": (
        "Tenstorrent/flux.2-klein-4b-refs",
        "4067a49b40b78acc50278520fcc36d5afbcbbbf4",
    ),
    "qwen/qwen-image-edit-2511": (
        "Tenstorrent/qwen-image-edit-2511-refs",
        "39d9c000c0e84200bd9da744ba5c1cb0695a44a2",
    ),
}
SUBSETS = ("dev", "ci", "all")
# TT's Qwen-Image-Edit runner today (tt-metal tvardhineni/qwen-image-edit-2511-rc,
# QwenImageEditPipeline.__call__): one input, padded to a square canvas.
SQUARE_SIDE = 1024


def eval_subset(limit_samples_mode: str | None) -> str:
    """The subset for the run's ``--limit-samples-mode``: all with no limit (full
    evals), dev for smoke-test, ci for ``--ci-mode`` and the other CI modes."""
    if not limit_samples_mode:
        return "all"
    if limit_samples_mode.lower().replace("_", "-") == "smoke-test":
        return "dev"
    return "ci"


def snapshot(model_id: str, patterns: list[str]) -> Path:
    """The model's eval data files matching ``patterns``, fetched once into the HF cache."""
    from huggingface_hub import snapshot_download

    repo, revision = EVAL_DATA[model_id]
    return Path(
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=list(patterns),
            token=os.environ.get("HF_TOKEN") or None,
        )
    )


def select_cases(
    manifest: list[dict], subset: str, max_images: int | None = None
) -> list[dict]:
    """The manifest's cases in ``subset``; ``max_images`` leaves out cases with more inputs."""
    if subset not in SUBSETS:
        raise ValueError(f"eval subset {subset!r}; expected one of {SUBSETS}")
    return [
        case
        for case in manifest
        if (subset == "all" or case[subset])
        and (max_images is None or len(case.get("images") or []) <= max_images)
    ]


def load_cases(
    model_id: str, subset: str, reference: str, max_images: int | None = None
) -> tuple[Path, list[dict]]:
    """Fetch the subset's cases, their inputs and their ``reference`` run's images.

    Returns the local dataset folder and the cases.
    """
    root = snapshot(model_id, ["cases/*"])
    cases = select_cases(
        json.loads((root / "cases" / "manifest.json").read_text()), subset, max_images
    )
    files = {path for case in cases for path in case.get("images") or []}
    files |= {f"{reference}/{case['case_id']}.png" for case in cases}
    files |= {f"{reference}/index.json", f"{reference}/provenance.json"}
    return snapshot(model_id, ["cases/*", *sorted(files)]), cases


def letterbox(image, side: int = SQUARE_SIDE):
    """The input as TT's Qwen-Image-Edit runner pads it: onto a ``side`` square,
    aspect kept, filled with the color at (2, 2) (PIL ``ImageOps.pad``)."""
    from PIL import ImageOps

    pad_color = (
        image.getpixel((2, 2)) if image.size[0] > 2 and image.size[1] > 2 else (0, 0, 0)
    )
    return ImageOps.pad(image.convert("RGB"), (side, side), color=pad_color)
