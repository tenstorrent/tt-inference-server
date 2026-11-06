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
under ``--ci-mode`` and the other CI modes, all on a full-evals run. A model is
compared with its ``REFERENCES`` run, ``refs/r1`` by default; a reference that
covers only some cases (its ``index.json``) leaves the others out. A case with a
``caption_key`` (FLUX 3) also gets its prompt as the GPU run's upsampler wrote it,
``captions/frozen/<caption_key>.json``, as ``frozen_prompt``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# model_id -> (Hugging Face dataset, pinned revision)
EVAL_DATA = {
    "black-forest-labs/flux.2-klein-4b": (
        "Tenstorrent/flux.2-klein-4b-refs",
        "6fa54cad3af7730e811cd9284324ff2c1e030c25",
    ),
    "qwen/qwen-image-edit-2511": (
        "Tenstorrent/qwen-image-edit-2511-refs",
        "8104d15d8e6edb0da8f43db00269ed8a24592aca",
    ),
    "ideogram-ai/ideogram-4": (
        "Tenstorrent/ideogram-4-refs",
        "9b8d74ca4fe1ad0fe811db1199c765e5c81faf90",
    ),
    "black-forest-labs/flux-3-image-klein": (
        "Tenstorrent/flux-3-image-klein-refs",
        "91da28bd0bf46f32c27d380fe6dcae602bc7992c",
    ),
}
# model_id -> the GPU run TT is compared with, where it isn't refs/r1. Qwen's
# server runs DBCache with cache-dit's `tt` recipe, so R2.
REFERENCES = {"qwen/qwen-image-edit-2511": "refs/r2/tt"}
SUBSETS = ("dev", "ci", "all")
# Case tasks that check a dataset's GPU run itself, not a deployment.
GPU_RUN_CHECKS = ("bfl_check",)
# A reference run's text-to-image scores (longtext_score.py, dpg_score.py).
REFERENCE_SCORES = ("longtext.json", "dpg.json")


def reference_for(model_id: str) -> str:
    return REFERENCES.get(model_id, "refs/r1")


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
    """The manifest's cases in ``subset``; ``max_images`` leaves out cases with more inputs.

    A dataset's checks of its own GPU run (FLUX 3's ``bfl_check``, against the
    model publisher's images) are not cases for a deployment.
    """
    if subset not in SUBSETS:
        raise ValueError(f"eval subset {subset!r}; expected one of {SUBSETS}")
    return [
        case
        for case in manifest
        if (subset == "all" or case[subset])
        and case.get("task") not in GPU_RUN_CHECKS
        and (max_images is None or len(case.get("images") or []) <= max_images)
    ]


def load_cases(
    model_id: str, subset: str, reference: str, max_images: int | None = None
) -> tuple[Path, list[dict]]:
    """Fetch the subset's cases, their inputs and their ``reference`` run's images.

    Returns the local dataset folder and the cases.
    """
    root = snapshot(model_id, ["cases/*", f"{reference}/index.json"])
    cases = select_cases(
        json.loads((root / "cases" / "manifest.json").read_text()), subset, max_images
    )
    covered = {
        entry["case_id"]
        for entry in json.loads((root / reference / "index.json").read_text())
    }
    if left_out := [case for case in cases if case["case_id"] not in covered]:
        logger.info(
            f"{model_id}: {len(left_out)} of {len(cases)} {subset} cases have no image "
            f"in {reference}, and are left out"
        )
        cases = [case for case in cases if case["case_id"] in covered]
    files = {path for case in cases for path in case.get("images") or []}
    files |= {f"{reference}/{case['case_id']}.png" for case in cases}
    files |= {f"{reference}/index.json", f"{reference}/provenance.json"}
    # The reference run's text-to-image scores, where it has them.
    files |= {f"{reference}/{name}" for name in REFERENCE_SCORES}
    # A model whose prompts are upsampled first (FLUX 3) has each case's
    # caption, upsampled once on the GPU and frozen.
    files |= {frozen_caption_path(case) for case in cases if case.get("caption_key")}
    root = snapshot(model_id, ["cases/*", *sorted(files)])
    for case in cases:
        if case.get("caption_key"):
            path = root / frozen_caption_path(case)
            if path.is_file():
                case["frozen_prompt"] = json.loads(path.read_text())["caption"]
            else:
                logger.warning(f"{case['case_id']}: no frozen caption at {path}")
    return root, cases


def frozen_caption_path(case: dict) -> str:
    return f"captions/frozen/{case['caption_key']}.json"
