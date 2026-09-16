#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cache the GLM-5.3 SWE-bench tasks and pull their Docker base images."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def base_image(dockerfile: Path) -> str:
    images = []
    for line in dockerfile.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0].upper() == "FROM":
            images.append(words[1])
    if len(images) != 1 or "$" in images[0] or images[0].startswith("--"):
        raise ValueError(f"Expected one literal Docker base image: {dockerfile}")
    return images[0]


def pull_image(image: str) -> dict:
    inspect = ["docker", "image", "inspect", image]
    cached = subprocess.run(inspect, capture_output=True).returncode == 0
    if not cached:
        subprocess.run(["docker", "pull", image], check=True, timeout=600)
    info = json.loads(subprocess.check_output(inspect))[0]
    return {"image": image, "id": info["Id"], "digests": info.get("RepoDigests", []), "cached": cached}


async def prepare(args) -> None:
    from harbor.registry.client.factory import RegistryClientFactory
    from harbor.tasks.client import TaskClient
    from reference_config.evals.eval_config import _eval_config_map
    from workflows.workflow_types import EvalLimitMode

    task = next(
        t for t in _eval_config_map["zai-org/GLM-5.3"].tasks
        if t.task_name == "swe_bench_verified"
    )
    config = task.agentic_eval_config
    mode = EvalLimitMode.from_string(args.limit_samples_mode) if args.limit_samples_mode else None
    names = config.task_names_map.get(mode, config.task_names)
    count = task.limit_samples_map.get(mode, config.n_tasks)
    metadata = await RegistryClientFactory.create().get_dataset_metadata(config.dataset)
    ids = metadata.task_ids
    if names:
        ids = [t for t in ids if any(fnmatch.fnmatch(t.path.name, name) for name in names)]
    if len(ids) != count:
        raise ValueError(f"Expected {count} configured tasks, resolved {len(ids)}")
    downloaded = await TaskClient().download_tasks(task_ids=ids)
    images = sorted({base_image(path / "environment/Dockerfile") for path in downloaded.paths})
    print(f"Preparing {len(ids)} tasks and {len(images)} base images", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        image_info = list(pool.map(pull_image, images))
    manifest = {
        "dataset": config.dataset,
        "limit_samples_mode": args.limit_samples_mode,
        "tasks": [
            {"source": task_id.model_dump(mode="json"), "cache_path": str(path)}
            for task_id, path in zip(ids, downloaded.paths)
        ],
        "images": image_info,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Prepared manifest: {args.manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-samples-mode", choices=["smoke-test", "ci-nightly"])
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.max_workers < 1 or args.manifest.exists():
        parser.error("Use positive --max-workers and a new --manifest path")
    subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["docker", "compose", "version"], check=True)
    asyncio.run(prepare(args))


if __name__ == "__main__":
    main()
