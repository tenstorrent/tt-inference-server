# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Z-Image-Turbo combined eval orchestrator.

This is the public entrypoint used by ``image_eval_tests.IMAGE_EVAL_DISPATCH``
for the ``tt-z-image-turbo`` runner. It chains every Z-Image-Turbo PCC
sub-eval serially and merges their results into a single response dict
that downstream report code can consume without knowing about sub-evals.

Response shape
--------------
The merged dict matches the per-sub-eval dict for backward compatibility.
    {
      "success": <bool>,                         # True iff every sub-eval PASSed
      "eval_results": {
        "accuracy_check": <int AccuracyResult>,  # worst across sub-evals
        "pcc_results":   [...],                  # flat list, each entry tagged "subtest"
        "pcc_min":       <float|None>,           # min across all sub-eval PCCs
        "pcc_mean":      <float|None>,           # mean across all sub-eval PCCs
        "pcc_tolerance": <float|None>,           # only when all sub-evals share one
        "test_type":     "z_image_turbo_combined",
        "subtests_run":  ["image_level_e2e", "dit_isolated", ...],
        "subtests_summary": { <name>: { pcc_min, pcc_mean, pcc_tolerance,
                                         accuracy_check, total_time_s,
                                         num_generation_failures }, ... },
        "num_generation_failures": <int>,        # summed across sub-evals
        "total_time_s": <float>,                 # summed across sub-evals
      },
    }
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from ...context import MediaContext
from ..image_generation_eval_test import AccuracyResult
from .dit_eval import run_dit_eval
from .pcc_eval import run_pcc_eval
from .text_encoder_eval import run_text_encoder_eval
from .vae_eval import run_vae_eval

logger = logging.getLogger(__name__)


# Each entry: (subtest_name, async_callable). Order = run order.
_SubEvalFn = Callable[[MediaContext, Optional[str]], Awaitable[dict]]
_SUB_EVALS: list[tuple[str, _SubEvalFn]] = [
    ("image_level_e2e", run_pcc_eval),
    ("text_encoder_isolated", run_text_encoder_eval),
    ("dit_isolated", run_dit_eval),
    ("vae_isolated", run_vae_eval),
]


def _merge_overall(current: AccuracyResult, sub: AccuracyResult) -> AccuracyResult:
    """Aggregate accuracy state across sub-evals."""
    if sub == AccuracyResult.FAIL:
        return AccuracyResult.FAIL
    if sub == AccuracyResult.BASELINE and current != AccuracyResult.FAIL:
        return AccuracyResult.BASELINE
    return current


async def run_z_image_turbo_eval(
    ctx: MediaContext, runner: Optional[str] = None
) -> dict:
    """Combined Z-Image-Turbo eval: runs every sub-eval and merges results.

    This is the function that ``image_eval_tests.IMAGE_EVAL_DISPATCH`` invokes
    for the ``tt-z-image-turbo`` runner. Each sub-eval owns its own goldens
    and threshold file (so they cannot collide), and contributes its
    pcc_results to a single flat list tagged with the sub-eval name.
    """
    logger.info(
        "Running Z-Image-Turbo combined eval: %d sub-evals (%s).",
        len(_SUB_EVALS),
        ", ".join(name for name, _ in _SUB_EVALS),
    )

    aggregated_pcc_results: list[dict] = []
    subtests_summary: dict[str, dict] = {}
    total_time = 0.0
    total_generation_failures = 0
    overall = AccuracyResult.PASS

    for name, fn in _SUB_EVALS:
        logger.info(f"=== Sub-eval start: {name} ===")
        result = await fn(ctx, runner)
        eval_data = result.get("eval_results", {})

        for entry in eval_data.get("pcc_results", []):
            aggregated_pcc_results.append({"subtest": name, **entry})

        sub_status = AccuracyResult(
            eval_data.get("accuracy_check", int(AccuracyResult.FAIL))
        )
        overall = _merge_overall(overall, sub_status)

        sub_total_time = eval_data.get("total_time_s") or 0.0
        sub_failed = eval_data.get("num_generation_failures") or 0
        total_time += sub_total_time
        total_generation_failures += sub_failed

        subtests_summary[name] = {
            "accuracy_check": int(sub_status),
            "pcc_min": eval_data.get("pcc_min"),
            "pcc_mean": eval_data.get("pcc_mean"),
            "pcc_tolerance": eval_data.get("pcc_tolerance"),
            "total_time_s": sub_total_time,
            "num_generation_failures": sub_failed,
        }

        logger.info(
            "=== Sub-eval done: %s — status=%s, pcc_min=%s, pcc_mean=%s ===",
            name,
            sub_status.name,
            f"{eval_data.get('pcc_min'):.6f}"
            if eval_data.get("pcc_min") is not None
            else "n/a",
            f"{eval_data.get('pcc_mean'):.6f}"
            if eval_data.get("pcc_mean") is not None
            else "n/a",
        )

    pcc_values = [r["pcc"] for r in aggregated_pcc_results if r.get("pcc") is not None]
    overall_pcc_min = min(pcc_values) if pcc_values else None
    overall_pcc_mean = (sum(pcc_values) / len(pcc_values)) if pcc_values else None

    # Surface a top-level pcc_tolerance only when every sub-eval reports the
    # same value; otherwise leave it None so consumers don't get misled.
    tolerances = {
        s["pcc_tolerance"]
        for s in subtests_summary.values()
        if s.get("pcc_tolerance") is not None
    }
    common_tolerance = next(iter(tolerances)) if len(tolerances) == 1 else None

    logger.info(
        "Z-Image-Turbo combined eval summary: %d sub-evals, %d generation "
        "failures total, PCC min=%s, mean=%s, result=%s",
        len(_SUB_EVALS),
        total_generation_failures,
        f"{overall_pcc_min:.6f}" if overall_pcc_min is not None else "n/a",
        f"{overall_pcc_mean:.6f}" if overall_pcc_mean is not None else "n/a",
        overall.name,
    )

    return {
        "success": overall == AccuracyResult.PASS,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": aggregated_pcc_results,
            "pcc_min": overall_pcc_min,
            "pcc_mean": overall_pcc_mean,
            "pcc_tolerance": common_tolerance,
            "test_type": "z_image_turbo_combined",
            "subtests_run": [name for name, _ in _SUB_EVALS],
            "subtests_summary": subtests_summary,
            "num_generation_failures": total_generation_failures,
            "total_time_s": total_time,
        },
    }
