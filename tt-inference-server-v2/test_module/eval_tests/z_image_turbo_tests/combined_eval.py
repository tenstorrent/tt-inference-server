# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Z-Image-Turbo combined eval -- runs all sub-evals in-process.

Creates a single mesh device, runs PCC tests for each component
(text encoder, DiT, VAE) and a pipeline perf test, then merges
results into the dict format that IMAGE_EVAL_DISPATCH expects.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...context import MediaContext
from ..image_generation_eval_test import AccuracyResult
from ._common import open_mesh
from .dit_eval import run_dit_pcc
from .pipeline_perf_eval import run_pipeline_perf
from .text_encoder_eval import run_text_encoder_pcc
from .vae_eval import run_vae_pcc

logger = logging.getLogger(__name__)

_SUB_EVALS = [
    ("text_encoder_pcc", run_text_encoder_pcc),
    ("dit_pcc", run_dit_pcc),
    ("vae_pcc", run_vae_pcc),
    ("pipeline_perf", run_pipeline_perf),
]


async def run_z_image_turbo_eval(
    ctx: MediaContext, runner: Optional[str] = None
) -> dict:
    logger.info(
        "Z-Image-Turbo combined eval: %d sub-evals (%s).",
        len(_SUB_EVALS),
        ", ".join(name for name, _ in _SUB_EVALS),
    )

    aggregated_pcc_results: list[dict] = []
    subtests_summary: dict[str, dict] = {}
    overall = AccuracyResult.PASS

    with open_mesh() as mesh_device:
        for name, fn in _SUB_EVALS:
            logger.info("=== Sub-eval start: %s ===", name)
            result = fn(mesh_device)
            eval_data = result.get("eval_results", {})

            for entry in eval_data.get("pcc_results", []):
                aggregated_pcc_results.append(
                    {"subtest": name, **entry}
                )

            sub_status = AccuracyResult(
                eval_data.get(
                    "accuracy_check", int(AccuracyResult.FAIL)
                )
            )
            if sub_status == AccuracyResult.FAIL:
                overall = AccuracyResult.FAIL

            subtests_summary[name] = {
                "accuracy_check": int(sub_status),
                "pcc_min": eval_data.get("pcc_min"),
                "pcc_mean": eval_data.get("pcc_mean"),
                "test_type": eval_data.get("test_type"),
            }

            # Surface perf data when available
            if "avg_ms" in eval_data:
                subtests_summary[name].update({
                    "warmup_ms": eval_data.get("warmup_ms"),
                    "avg_ms": eval_data.get("avg_ms"),
                    "min_ms": eval_data.get("min_ms"),
                    "max_ms": eval_data.get("max_ms"),
                })

            logger.info(
                "=== Sub-eval done: %s — %s ===",
                name, sub_status.name,
            )

    pcc_values = [
        r["pcc"] for r in aggregated_pcc_results
        if r.get("pcc") is not None
    ]
    pcc_min = min(pcc_values) if pcc_values else None
    pcc_mean = (
        sum(pcc_values) / len(pcc_values) if pcc_values else None
    )

    logger.info(
        "Combined eval done: PCC min=%s mean=%s result=%s",
        f"{pcc_min:.6f}" if pcc_min is not None else "n/a",
        f"{pcc_mean:.6f}" if pcc_mean is not None else "n/a",
        overall.name,
    )

    return {
        "success": overall == AccuracyResult.PASS,
        "eval_results": {
            "accuracy_check": int(overall),
            "pcc_results": aggregated_pcc_results,
            "pcc_min": pcc_min,
            "pcc_mean": pcc_mean,
            "test_type": "z_image_turbo_combined",
            "subtests_run": [name for name, _ in _SUB_EVALS],
            "subtests_summary": subtests_summary,
        },
    }
