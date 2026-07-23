#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Pre-pull Docker images for the configured agentic eval pipelines.

The agent implementation is inferred from ``evals/eval_config.py``; callers do
not select it independently.  SWE-bench uses two image registries:

1. **mini-swe-agent / SWE-agent** (patch generation) — Docker Hub via
   ``SimpleBatchInstance.from_swe_bench``::

       docker.io/swebench/sweb.eval.x86_64.<instance_id>:latest

   where ``__`` in the instance id is replaced with ``_1776_``.

   The full SWE-agent backend also builds a wrapper image from
   ``python:3.11.9-slim-bookworm``. mini-swe-agent does not.

2. **SWE-bench harness** (scoring via ``swebench.harness.run_evaluation``) —
   GHCR via ``TestSpec``::

       ghcr.io/epoch-research/swe-bench.base.x86_64:latest
       ghcr.io/epoch-research/swe-bench.env.x86_64.<hash>:latest
       ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>:latest

   Instance ids keep ``__`` (no ``_1776_`` substitution).

Harbor-backed evals are resolved from their configured datasets:

* Terminal Bench 2.0 and 2.1 tasks normally declare prebuilt ``alexgshaw/*``
  images. Their task packages (including ``instruction.md``) are cached
  separately from Docker.
* Tau3-bench tasks are built locally, so this script pre-pulls the base images
  referenced by their Dockerfiles (currently ``python:3.12-slim``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from tqdm import tqdm

logger = logging.getLogger(__name__)

Target = Literal["sweagent", "harness", "all"]
Benchmark = Literal["swe-bench", "terminal-bench-2", "tau3-bench"]

BENCHMARK_TASK_NAMES: dict[Benchmark, tuple[str, ...]] = {
    "swe-bench": ("swe_bench_verified",),
    "terminal-bench-2": ("terminal_bench_2", "terminal_bench_2_1"),
    "tau3-bench": ("tau3_bench_banking",),
}

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


@dataclass(frozen=True)
class HarborBenchmarkConfig:
    dataset: str
    agent: str
    agent_import_path: str | None
    task_names: tuple[str, ...]
    exclude_task_names: tuple[str, ...]
    n_tasks: int | None


def _load_eval_config_list() -> list[Any]:
    """Load the source configs, including repos not present in ``MODEL_SPECS``."""
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from evals.eval_config import _eval_config_list

    return _eval_config_list


def _infer_sweagent_backend() -> str:
    task_names = BENCHMARK_TASK_NAMES["swe-bench"]
    backends = {
        task.swebench_eval_config.agent_backend
        for eval_config in _load_eval_config_list()
        for task in eval_config.tasks
        if task.task_name in task_names and task.swebench_eval_config is not None
    }
    if len(backends) != 1:
        raise ValueError(
            f"Expected one configured backend for {task_names}, found {sorted(backends)}"
        )
    return backends.pop()


def _infer_harbor_task_config(task_name: str) -> HarborBenchmarkConfig:
    """Infer one task's dataset and agent metadata from every matching config."""
    configs = [
        task.agentic_eval_config
        for eval_config in _load_eval_config_list()
        for task in eval_config.tasks
        if task.task_name == task_name and task.agentic_eval_config is not None
    ]
    if not configs:
        raise ValueError(f"No agentic eval config found for {task_name}")

    signatures = {
        (
            config.dataset,
            config.agent,
            config.agent_import_path,
            tuple(config.task_names),
            tuple(config.exclude_task_names),
        )
        for config in configs
    }
    if len(signatures) != 1:
        raise ValueError(
            f"Conflicting dataset/agent configs found for {task_name}: {signatures}"
        )

    dataset, agent, import_path, task_names, exclude_task_names = signatures.pop()
    limits = [config.n_tasks for config in configs]
    n_tasks = None if any(limit is None for limit in limits) else max(limits)
    return HarborBenchmarkConfig(
        dataset=dataset,
        agent=agent,
        agent_import_path=import_path,
        task_names=task_names,
        exclude_task_names=exclude_task_names,
        n_tasks=n_tasks,
    )


def _infer_harbor_benchmark_configs(
    benchmark: Benchmark,
) -> list[HarborBenchmarkConfig]:
    """Resolve every configured dataset version belonging to a benchmark."""
    return [
        _infer_harbor_task_config(task_name)
        for task_name in BENCHMARK_TASK_NAMES[benchmark]
    ]


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


async def _download_harbor_task_dirs_async(
    config: HarborBenchmarkConfig,
    n_tasks: int | None,
) -> list[Path]:
    """Resolve and cache exactly the Harbor tasks selected by the eval config."""
    try:
        from harbor.models.job.config import DatasetConfig
        from harbor.tasks.client import TaskClient
    except ImportError as exc:
        raise RuntimeError(
            "Harbor is required for Terminal Bench and Tau3 image discovery. "
            "Run this script with the EVALS_AGENTIC venv."
        ) from exc

    dataset = DatasetConfig(
        name=config.dataset,
        task_names=list(config.task_names) or None,
        exclude_task_names=list(config.exclude_task_names) or None,
        n_tasks=n_tasks if n_tasks is not None else config.n_tasks,
    )
    task_configs = await dataset.get_task_configs()
    task_ids = [task_config.get_task_id() for task_config in task_configs]
    if not task_ids:
        return []

    result = await TaskClient().download_tasks(task_ids=task_ids)
    return [download.path for download in result.results]


def _download_harbor_task_dirs(
    config: HarborBenchmarkConfig,
    n_tasks: int | None,
) -> list[Path]:
    return asyncio.run(_download_harbor_task_dirs_async(config, n_tasks))


_FROM_RE = re.compile(
    r"^\s*FROM\s+(?:--platform(?:=\S+|\s+\S+)\s+)?(?P<image>\S+)"
    r"(?:\s+AS\s+(?P<stage>\S+))?",
    re.IGNORECASE,
)


def _dockerfile_base_images(path: Path) -> list[str]:
    """Return external images referenced by a Dockerfile's FROM instructions."""
    if not path.is_file():
        logger.warning("Dockerfile not found: %s", path)
        return []

    images: list[str] = []
    stages: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _FROM_RE.match(line)
        if match is None:
            continue
        image = match.group("image")
        if (
            image.lower() == "scratch"
            or image.lower() in stages
            or "$" in image
        ):
            if "$" in image:
                logger.warning("Cannot resolve dynamic FROM image %r in %s", image, path)
        elif image not in images:
            images.append(image)

        if stage := match.group("stage"):
            stages.add(stage.lower())
    return images


def _compose_images(environment_dir: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to inspect Harbor docker-compose files."
        ) from exc

    compose_path = next(
        (
            path
            for path in (
                environment_dir / "docker-compose.yaml",
                environment_dir / "docker-compose.yml",
            )
            if path.is_file()
        ),
        None,
    )
    if compose_path is None:
        return []

    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8")) or {}
    services = compose.get("services", {})
    images: list[str] = []
    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue

        image = service.get("image")
        if isinstance(image, str):
            if "$" in image:
                logger.warning(
                    "Cannot resolve dynamic compose image %r for service %s in %s",
                    image,
                    service_name,
                    compose_path,
                )
            elif image not in images:
                images.append(image)

        build = service.get("build")
        if not build:
            continue
        if isinstance(build, str):
            context = build
            dockerfile = "Dockerfile"
        elif isinstance(build, dict):
            context = build.get("context", ".")
            dockerfile = build.get("dockerfile", "Dockerfile")
        else:
            continue

        dockerfile_path = environment_dir / context / dockerfile
        for base_image in _dockerfile_base_images(dockerfile_path.resolve()):
            if base_image not in images:
                images.append(base_image)
    return images


def _harbor_images_for_task_dir(task_dir: Path) -> list[str]:
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        logger.warning("Harbor task is missing task.toml: %s", task_dir)
        return []

    config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    environment_config = config.get("environment", {})
    environment_dir = task_dir / "environment"
    images: list[str] = []

    prebuilt = environment_config.get("docker_image")
    if isinstance(prebuilt, str) and prebuilt:
        images.append(prebuilt)
    else:
        images.extend(_dockerfile_base_images(environment_dir / "Dockerfile"))

    for image in _compose_images(environment_dir):
        if image not in images:
            images.append(image)
    return images


def harbor_image_names(
    config: HarborBenchmarkConfig,
    n_tasks: int | None = None,
) -> list[str]:
    task_dirs = _download_harbor_task_dirs(config, n_tasks)
    images: list[str] = []
    seen: set[str] = set()
    for task_dir in task_dirs:
        for image in _harbor_images_for_task_dir(task_dir):
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


def _pull_images(
    images: Iterable[str],
    *,
    max_workers: int,
    skip_existing: bool,
    runtime: str,
    show_progress: bool,
) -> int:
    unique_images = list(dict.fromkeys(images))
    logger.info(
        "Pulling %d image(s) with %d worker(s)",
        len(unique_images),
        max_workers,
    )

    results: list[PullResult] = []
    pulled = skipped = 0
    failed: list[PullResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_pull_image, image, runtime, skip_existing): image
            for image in unique_images
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
                    logger.info("PULLED %s", result.image)
                elif result.skipped:
                    skipped += 1
                    logger.info("LOCAL  %s", result.image)
                else:
                    failed.append(result)
                    logger.error("FAILED %s -> %s", result.image, result.error)
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
    return 1 if failed else 0


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
        "Resolved SWE-bench target=%s subset=%s split=%s (%d instances)",
        target,
        subset,
        split,
        len(rows),
    )
    return _pull_images(
        images,
        max_workers=max_workers,
        skip_existing=skip_existing,
        runtime=runtime,
        show_progress=show_progress,
    )


def pull_agentic_images(
    *,
    benchmarks: Iterable[Benchmark],
    subset: str = "verified",
    split: str = "test",
    dataset_path: str | None = None,
    instance_ids: list[str] | None = None,
    n_tasks: int | None = None,
    swe_target: Target = "all",
    include_builder_image: bool = True,
    max_workers: int = 2,
    skip_existing: bool = True,
    runtime: str = "docker",
    show_progress: bool = True,
) -> int:
    images: list[str] = []
    selected = list(dict.fromkeys(benchmarks))

    if "swe-bench" in selected:
        backend = _infer_sweagent_backend()
        rows = _load_rows(
            subset=subset,
            split=split,
            dataset_path=dataset_path,
            instance_ids=instance_ids or [],
            n_tasks=n_tasks,
        )
        needs_builder = include_builder_image and backend == "swe-agent"
        images.extend(_collect_images(rows, swe_target, needs_builder))
        logger.info(
            "Resolved SWE-bench agent=%s (%d instances, target=%s)",
            backend,
            len(rows),
            swe_target,
        )

    for benchmark in ("terminal-bench-2", "tau3-bench"):
        if benchmark not in selected:
            continue
        for config in _infer_harbor_benchmark_configs(benchmark):
            benchmark_images = harbor_image_names(config, n_tasks=n_tasks)
            images.extend(benchmark_images)
            logger.info(
                "Resolved %s dataset=%s agent=%s (%d unique image(s))",
                benchmark,
                config.dataset,
                config.agent_import_path or config.agent,
                len(benchmark_images),
            )

    return _pull_images(
        images,
        max_workers=max_workers,
        skip_existing=skip_existing,
        runtime=runtime,
        show_progress=show_progress,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Pre-pull Docker images for configured SWE-bench, Terminal Bench "
            "2.0/2.1, and Tau3-bench evals"
        )
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["swe-bench", "terminal-bench-2", "tau3-bench", "all"],
        help=(
            "Benchmark to prepare; repeat for multiple benchmarks "
            "(default: all configured benchmark families)"
        ),
    )
    parser.add_argument(
        "--swe-target",
        "--target",
        dest="swe_target",
        choices=["sweagent", "harness", "all"],
        default="all",
        help="Which SWE-bench phase to pre-pull for (default: all)",
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
        help="Pull images for only the first N selected tasks per dataset",
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

    requested_benchmarks = args.benchmark or ["all"]
    benchmarks: list[Benchmark]
    if "all" in requested_benchmarks:
        benchmarks = ["swe-bench", "terminal-bench-2", "tau3-bench"]
    else:
        benchmarks = requested_benchmarks

    return pull_agentic_images(
        benchmarks=benchmarks,
        subset=args.subset,
        split=args.split,
        dataset_path=args.dataset_path,
        instance_ids=args.instance_ids,
        n_tasks=args.n_tasks,
        swe_target=args.swe_target,
        include_builder_image=not args.no_builder_image,
        max_workers=args.max_workers,
        skip_existing=not args.force,
        runtime=args.runtime,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    sys.exit(main())
