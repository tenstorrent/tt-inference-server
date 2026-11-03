# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Benchmarks for the image models the generations API serves (Qwen-Image-Edit-2511, Ideogram 4, FLUX.2 klein 4B).

Each configuration a model's GPU requirement was measured at is run in turn
and graded against its own entry in ``model_performance_reference.json``:

- Qwen-Image-Edit-2511: an edit with 1, 2 and 3 input images, at 1:1
  (1024x1024); the entries are keyed by ``num_images``.
- Ideogram 4: one image at 2048x2048 with the default (20 steps) and quality
  (48 steps) presets; the entries are keyed by ``preset``.
- FLUX.2 klein 4B: its release shapes, 1024x1024 and 2048x2048, with no input
  image and with one; the entries are keyed by ``num_images``.

A request is one job: ``POST /v1/generations/{model}``, then ``GET
/v1/jobs/{id}`` until it is final. Its time runs from the submit to the poll
that sees it completed, reported as ``ttft`` (the whole image, as the other
image benchmarks report it), and ``tput_user`` is images per second. Jobs run
one at a time and are deleted once done. One block holds every configuration,
its fields suffixed with the configuration's name.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from report_module.schema import Block
from workflow_module.context_helpers import get_num_calls

from .._test_common import MetricSpec, TestStatus, block_id, run_tiered_check
from .._test_common.generation_api import (
    FINAL_STATUSES,
    JOB_PATH,
    auth_headers,
    data_uri,
)
from .._test_common.image_generation_api import IDEOGRAM_CAPTION, model_for
from .._test_common.target_check import summary_from_tiered
from ..context import MediaContext

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGES = _PROJECT_ROOT / "test_fixtures" / "datasets" / "imagenet_subset"
POLL_INTERVAL_SECONDS = 0.25
JOB_TIMEOUT_SECONDS = 30 * 60
SEED = 42
QWEN_PROMPT = "Put the subjects of these images together on a sunny beach."
T2I_PROMPT = (
    "A lighthouse on a cliff at golden hour, waves breaking on the rocks below."
)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    # The target entry's keys (model_performance_reference.json).
    match: dict[str, Any]
    shape: tuple[int, int]
    request: dict[str, Any] = field(default_factory=dict)
    num_images: int = 0


def _fixture_images(count: int) -> list[str]:
    paths = sorted(FIXTURE_IMAGES.glob("*.jpg"))[:count]
    if len(paths) < count:
        raise FileNotFoundError(f"need {count} fixture images in {FIXTURE_IMAGES}")
    return [data_uri(p.read_bytes()) for p in paths]


BENCHMARKS: dict[str, list[BenchmarkConfig]] = {
    "tt-qwen-image-edit-2511": [
        BenchmarkConfig(
            name=f"{n}_input{'s' if n > 1 else ''}",
            match={"num_images": n},
            shape=(1024, 1024),
            request={"prompt": QWEN_PROMPT, "aspect_ratio": "1:1"},
            num_images=n,
        )
        for n in (1, 2, 3)
    ],
    # Its release shapes: 1:1 at 1K and 2K, with no reference or one.
    "tt-flux.2-klein-4b": [
        BenchmarkConfig(
            name=f"{side}_{n}_input{'s' if n != 1 else ''}",
            match={"num_images": n},
            shape=(side, side),
            request={
                "prompt": QWEN_PROMPT if n else T2I_PROMPT,
                "aspect_ratio": "1:1",
                "size": size,
            },
            num_images=n,
        )
        for side, size in ((1024, "1K"), (2048, "2K"))
        for n in (0, 1)
    ],
    "tt-ideogram-4": [
        BenchmarkConfig(
            name=preset,
            match={"preset": preset},
            shape=(2048, 2048),
            request={
                "structured_prompt": IDEOGRAM_CAPTION,
                "aspect_ratio": "1:1",
                "size": "2K",
                "rendering_speed": speed,
            },
        )
        for speed, preset in (
            ("default", "V4_DEFAULT_20"),
            ("quality", "V4_QUALITY_48"),
        )
    ],
}


def _run_job(base_url: str, submit_path: str, request: dict) -> tuple[bool, float]:
    """One job, submitted and polled to its end: whether it completed, and its seconds."""
    headers = auth_headers()
    started = time.monotonic()
    response = requests.post(
        f"{base_url}{submit_path}", json=request, headers=headers, timeout=60
    )
    if response.status_code != 202:
        logger.error(f"submit returned {response.status_code}: {response.text[:300]}")
        return False, 0.0
    job_id = response.json()["id"]
    url = f"{base_url}{JOB_PATH.format(job_id=job_id)}"
    status = None
    while time.monotonic() - started < JOB_TIMEOUT_SECONDS:
        status = requests.get(url, headers=headers, timeout=30).json().get("status")
        if status in FINAL_STATUSES:
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    elapsed = time.monotonic() - started
    if status in FINAL_STATUSES:
        requests.delete(url, headers=headers, timeout=30)
    if status != "completed":
        logger.error(f"job {job_id} ended {status} after {elapsed:.1f} s")
        return False, elapsed
    return True, elapsed


def _run_config(
    ctx: MediaContext, config: BenchmarkConfig, num_calls: int, submit_path: str
) -> dict:
    """``num_calls`` jobs of one configuration, its record and its target checks."""
    request = {**config.request, "seed": SEED, "format": "png"}
    if config.num_images:
        request["images"] = _fixture_images(config.num_images)
    results = []
    for i in range(num_calls):
        ok, seconds = _run_job(ctx.base_url, submit_path, request)
        logger.info(
            f"{config.name} {i + 1}/{num_calls}: {'ok' if ok else 'failed'} in {seconds:.2f} s"
        )
        results.append((ok, seconds))
    times = [s for ok, s in results if ok]
    ttft = sum(times) / len(times) if times else 0.0
    tput_user = 1 / ttft if ttft else 0.0
    checks, check = run_tiered_check(
        ctx,
        [
            MetricSpec(
                "TTFT",
                ttft * 1000 if ttft else None,
                "ttft_ms",
                lower_is_better=True,
                field_name=f"ttft_{config.name}",
            ),
            MetricSpec(
                "tput_user",
                tput_user,
                "tput_user",
                lower_is_better=False,
                field_name=f"tput_user_{config.name}",
            ),
        ],
        config.shape,
        config.match,
    )
    record = {
        "config": config.name,
        "shape": f"{config.shape[0]}x{config.shape[1]}",
        "num_requests": num_calls,
        "num_successful": len(times),
        "ttft": ttft,
        "tput_user": tput_user,
        "target_check": check,
    }
    return {"record": record, "checks": checks}


def run_generation_image_benchmark(ctx: MediaContext, runner: str) -> Block:
    """Every configuration of ``runner``'s model, each against its own targets, in one block."""
    model = model_for(ctx.model_spec.hf_model_repo)
    num_calls = get_num_calls(ctx)
    records, target_checks = [], {}
    for config in BENCHMARKS[runner]:
        result = _run_config(ctx, config, num_calls, model.submit_path)
        records.append(result["record"])
        for tier, fields in result["checks"].items():
            target_checks.setdefault(tier, {}).update(fields)
    num_total = sum(r["num_requests"] for r in records)
    num_successful = sum(r["num_successful"] for r in records)
    data: dict[str, Any] = {
        "records": records,
        "num_requests": num_total,
        "num_successful": num_successful,
        "target_check": summary_from_tiered(target_checks),
        "target_checks": target_checks,
    }
    if num_successful != num_total:
        logger.error(
            f"only {num_successful}/{num_total} jobs completed; marking the benchmark FAILED"
        )
        data["status"] = TestStatus.FAIL.value
    return Block(
        kind="benchmarks",
        task_type="image",
        title="Image Benchmark",
        id=block_id(ctx) or None,
        targets={"num_prompts": num_total, "configs": [r["config"] for r in records]},
        data=data,
    )


def serves_generation_benchmark(runner: str) -> bool:
    return runner in BENCHMARKS


__all__ = [
    "BENCHMARKS",
    "run_generation_image_benchmark",
    "serves_generation_benchmark",
]
