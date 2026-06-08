#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Pre-pull SWE-bench Docker images for the agentic eval pipeline.

This repo runs SWE-bench in two phases, each with its own image registry:

1. **SWE-agent** (patch generation) — Docker Hub via
   ``SimpleBatchInstance.from_swe_bench``::

       docker.io/swebench/sweb.eval.x86_64.<instance_id>:latest

   where ``__`` in the instance id is replaced with ``_1776_``.

   SWE-agent also builds a wrapper image from ``python:3.11.9-slim-bookworm``.

2. **SWE-bench harness** (scoring via ``swebench.harness.run_evaluation``) —
   GHCR via ``TestSpec``::

       ghcr.io/epoch-research/swe-bench.base.x86_64:latest
       ghcr.io/epoch-research/swe-bench.env.x86_64.<hash>:latest
       ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>:latest

   Instance ids keep ``__`` (no ``_1776_`` substitution).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from tqdm import tqdm

logger = logging.getLogger(__name__)

Target = Literal["sweagent", "harness", "all"]

# Same mapping as SWE-agent ``SWEBenchInstances._get_dataset_path``.
DATASET_BY_SUBSET = {
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "full": "princeton-nlp/SWE-Bench",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
}

# swerex glibc_dockerfile hardcodes this builder base image.
SWEREX_BUILDER_IMAGE = "python:3.11.9-slim-bookworm"


@dataclass(frozen=True)
class PullResult:
    image: str
    ok: bool
    skipped: bool = False
    error: str = ""


def _import_make_test_spec() -> Callable[[dict[str, Any]], Any]:
    """Import SWE-bench harness ``make_test_spec`` from the venv package.

    Running this file as ``python evals/agentic/pull_swebench_images.py`` puts
    ``evals/agentic`` on ``sys.path``, where our ``swebench.py`` wrapper shadows
    the installed ``swebench`` package from SWE-bench.
    """
    script_dir = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == script_dir:
        sys.path.pop(0)
    elif script_dir in sys.path:
        sys.path.remove(script_dir)

    loaded = sys.modules.get("swebench")
    if loaded is not None and not hasattr(loaded, "__path__"):
        for name in list(sys.modules):
            if name == "swebench" or name.startswith("swebench."):
                del sys.modules[name]

    from swebench.harness.test_spec import make_test_spec

    return make_test_spec


def sweagent_image_name(instance_id: str, image_name: str | None = None) -> str:
    """Match SWE-agent ``SimpleBatchInstance.from_swe_bench`` image naming."""
    if image_name:
        return image_name
    id_docker_compatible = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


def harness_image_names(rows: list[dict]) -> list[str]:
    """Match ``swebench.harness.test_spec.TestSpec`` image keys (+ :latest for pull)."""
    make_test_spec = _import_make_test_spec()

    images: list[str] = []
    seen: set[str] = set()
    for row in rows:
        spec = make_test_spec(row)
        for key in (spec.base_image_key, spec.env_image_key, spec.instance_image_key):
            tagged = f"{key}:latest"
            if tagged not in seen:
                seen.add(tagged)
                images.append(tagged)
    return images


def _image_exists(image: str, runtime: str = "docker") -> bool:
    result = subprocess.run(
        [runtime, "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _pull_image(image: str, runtime: str = "docker", skip_existing: bool = True) -> PullResult:
    if skip_existing and _image_exists(image, runtime):
        logger.debug("Skipping (already local): %s", image)
        return PullResult(image=image, ok=True, skipped=True)

    logger.debug("Pulling %s", image)
    result = subprocess.run(
        [runtime, "pull", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return PullResult(image=image, ok=True)

    error = (result.stderr or result.stdout or "unknown error").strip()
    logger.debug("Failed to pull %s: %s", image, error)
    return PullResult(image=image, ok=False, error=error)


def _load_rows(
    subset: str,
    split: str,
    dataset_path: str | None,
    instance_ids: list[str],
    n_tasks: int | None,
) -> list[dict]:
    if instance_ids:
        from datasets import load_dataset

        path = dataset_path or DATASET_BY_SUBSET[subset]
        dataset = load_dataset(path, split=split)
        by_id = {row["instance_id"]: row for row in dataset}
        missing = [iid for iid in instance_ids if iid not in by_id]
        if missing:
            raise ValueError(f"Unknown instance id(s) for {path}: {missing}")
        return [by_id[iid] for iid in instance_ids]

    from datasets import load_dataset

    path = dataset_path or DATASET_BY_SUBSET[subset]
    dataset = load_dataset(path, split=split)
    rows = list(dataset)
    if n_tasks is not None:
        rows = rows[:n_tasks]
    return rows


def _sweagent_images_for_rows(rows: list[dict]) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()
    for row in rows:
        image = sweagent_image_name(row["instance_id"], row.get("image_name"))
        if image not in seen:
            seen.add(image)
            images.append(image)
    return images


def _collect_images(
    rows: list[dict],
    target: Target,
    include_builder_image: bool,
) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()

    def add_many(items: list[str]) -> None:
        for image in items:
            if image not in seen:
                seen.add(image)
                images.append(image)

    if target in ("sweagent", "all"):
        if include_builder_image:
            add_many([SWEREX_BUILDER_IMAGE])
        add_many(_sweagent_images_for_rows(rows))
    if target in ("harness", "all"):
        add_many(harness_image_names(rows))
    return images


def pull_swebench_images(
    *,
    subset: str = "verified",
    split: str = "test",
    dataset_path: str | None = None,
    instance_ids: list[str] | None = None,
    n_tasks: int | None = None,
    target: Target = "all",
    include_builder_image: bool = True,
    max_workers: int = 2,
    skip_existing: bool = True,
    runtime: str = "docker",
    show_progress: bool = True,
) -> int:
    rows = _load_rows(
        subset=subset,
        split=split,
        dataset_path=dataset_path,
        instance_ids=instance_ids or [],
        n_tasks=n_tasks,
    )
    images = _collect_images(rows, target, include_builder_image)

    logger.info(
        "Pulling %d image(s) target=%s subset=%s split=%s (%d instances, max_workers=%d)",
        len(images),
        target,
        subset,
        split,
        len(rows),
        max_workers,
    )

    results: list[PullResult] = []
    pulled = skipped = 0
    failed: list[PullResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_pull_image, image, runtime, skip_existing): image
            for image in images
        }
        progress = tqdm(
            total=len(futures),
            desc="Pulling images",
            unit="image",
            disable=not show_progress,
        )
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.ok and not result.skipped:
                    pulled += 1
                elif result.skipped:
                    skipped += 1
                else:
                    failed.append(result)
                progress.set_postfix(
                    pulled=pulled,
                    skipped=skipped,
                    failed=len(failed),
                    refresh=False,
                )
                progress.update(1)
        finally:
            progress.close()

    logger.info(
        "Done: %d pulled, %d skipped, %d failed (of %d total)",
        pulled,
        skipped,
        len(failed),
        len(results),
    )
    for result in failed:
        logger.error("  %s -> %s", result.image, result.error)

    return 1 if failed else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Pre-pull SWE-bench Docker images for SWE-agent and/or the harness"
    )
    parser.add_argument(
        "--target",
        choices=["sweagent", "harness", "all"],
        default="all",
        help="Which phase to pre-pull for (default: all)",
    )
    parser.add_argument(
        "--subset",
        choices=sorted(DATASET_BY_SUBSET),
        default="verified",
        help="SWE-bench subset (default: verified)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--dataset-path",
        help="Override Hugging Face dataset path (e.g. SWE-bench/SWE-bench_Verified)",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        dest="instance_ids",
        help="Pull only specific instance id(s); repeat flag for multiple",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        help="Pull images for only the first N instances in the split",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel docker pull workers (default: 2; keep low to avoid rate limits)",
    )
    parser.add_argument(
        "--no-builder-image",
        action="store_true",
        help=f"Skip pulling {SWEREX_BUILDER_IMAGE} used by swerex wrapper builds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pull even if the image is already present locally",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar",
    )
    parser.add_argument(
        "--runtime",
        default="docker",
        help="Container runtime binary (default: docker)",
    )
    args = parser.parse_args()

    return pull_swebench_images(
        subset=args.subset,
        split=args.split,
        dataset_path=args.dataset_path,
        instance_ids=args.instance_ids,
        n_tasks=args.n_tasks,
        target=args.target,
        include_builder_image=not args.no_builder_image,
        max_workers=args.max_workers,
        skip_existing=not args.force,
        runtime=args.runtime,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    sys.exit(main())
