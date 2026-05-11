#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Temporary v2 orchestrator: drive v2 evals/benchmarks against an
already-running inference server and emit a single combined report.

Unlike the top-level run.py, this script does NOT manage uv,
docker, secrets, or the server lifecycle, and it dispatches via the
v2 ``test_module`` (not the v1 evals/ + benchmarking/ subprocess
venvs). Start the inference server separately, then point this at it
via --service-port. Examples:

    # benchmarks only
    python tt-inference-server-v2/run.py \
        --model stable-diffusion-xl-base-1.0 --workflow benchmarks \
        --device n150 --service-port 8000

    # full release-style chain (eval → benchmark → one combined report)
    python tt-inference-server-v2/run.py \
        --model stable-diffusion-xl-base-1.0 --workflow release \
        --device n150 --service-port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from workflows.model_spec import MODEL_SPECS, get_runtime_model_spec  # noqa: E402
from workflows.workflow_types import DeviceTypes  # noqa: E402

from report_module import ReportGenerator  # noqa: E402
from test_module import MediaContext, MediaTaskType, run_media_task  # noqa: E402
from workflow_module import get_default_accumulator  # noqa: E402

logger = logging.getLogger("tt_v2_run")


# Single-task workflows map straight to a MediaTaskType. "release" is a
# meta-workflow handled separately (eval → benchmark → one report).
WORKFLOW_TO_TASKS: dict[str, list[MediaTaskType]] = {
    "benchmarks": [MediaTaskType.BENCHMARK],
    "evals": [MediaTaskType.EVALUATION],
    "spec_tests": [MediaTaskType.SPEC_TESTS],
    "release": [
        MediaTaskType.EVALUATION,
        MediaTaskType.BENCHMARK,
        MediaTaskType.SPEC_TESTS,
    ],
}


def parse_args() -> argparse.Namespace:
    valid_models = sorted({spec.model_name for spec in MODEL_SPECS.values()})
    valid_devices = sorted({d.name.lower() for d in DeviceTypes})

    parser = argparse.ArgumentParser(
        description=(
            "Drive v2 evals/benchmarks against a running inference server "
            "and emit a combined report."
        ),
        epilog="Available models:\n  " + "\n  ".join(valid_models),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=valid_models)
    parser.add_argument(
        "--workflow", required=True, choices=sorted(WORKFLOW_TO_TASKS)
    )
    parser.add_argument("--device", required=True, choices=valid_devices)
    parser.add_argument("--service-port", type=int, default=8000)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help=(
            "Override the per-runner image benchmark prompt count. Patches "
            "SDXL_BENCHMARK_NUM_PROMPTS / SDXL_SD35_BENCHMARK_NUM_PROMPTS in "
            "test_module.benchmark_tests.image_benchmark_tests so a smoke "
            "run doesn't take an hour."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_V2_ROOT / "output",
        help="Where to write the rendered report (markdown + json).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def _resolve_eval_config(model_name: str):
    """Return the v1 EvalConfig for ``model_name`` if available, else None.

    v2's image runners read ``ctx.all_params.tasks[0]`` for task_name /
    score / tolerance — we reuse the v1 config rather than redefining the
    same metadata in v2.
    """
    try:
        from evals.eval_config import EVAL_CONFIGS
    except Exception as e:
        logger.warning("Could not import v1 EVAL_CONFIGS (%s); evals will fail.", e)
        return None
    cfg = EVAL_CONFIGS.get(model_name)
    if cfg is None:
        logger.warning(
            "No EvalConfig registered for model=%r; eval task metadata will be empty.",
            model_name,
        )
    return cfg


def build_context(args: argparse.Namespace) -> MediaContext:
    model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
    # _test_common.target_check.load_targets() reads ctx.model_spec.cli_args["device"]
    # to look up performance targets — populate it explicitly since we skip the
    # full RuntimeConfig pipeline.
    model_spec.cli_args["device"] = args.device
    # eval_tests/image_eval_tests._run_image_generation_eval reads
    # is_sdxl_num_prompts_enabled(ctx), which pulls sdxl_num_prompts off
    # cli_args (default 100, clamped to [2, 5000]). Plumb --num-prompts
    # through so the eval phase respects the smoke-run cap too.
    if args.num_prompts is not None:
        model_spec.cli_args["sdxl_num_prompts"] = max(2, args.num_prompts)

    device = DeviceTypes.from_string(args.device)

    output_path = args.output_dir / f"{args.model}_{args.device}_{args.workflow}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Image evals access ctx.all_params.tasks[0]; benchmarks only consume
    # all_params via get_num_calls(), which returns 2 (hardcoded "evals"
    # path) when all_params has a .tasks attr. SDXL-trace overrides that
    # to SDXL_BENCHMARK_NUM_PROMPTS at dispatch time anyway, so reusing
    # the EvalConfig here is safe for benchmark-only runs too.
    eval_cfg = _resolve_eval_config(args.model)
    all_params = eval_cfg if eval_cfg is not None else []

    return MediaContext(
        all_params=all_params,
        model_spec=model_spec,
        device=device,
        output_path=str(output_path),
        service_port=args.service_port,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ctx = build_context(args)
    task_types: List[MediaTaskType] = WORKFLOW_TO_TASKS[args.workflow]

    if args.num_prompts is not None:
        from test_module import dispatch as _dispatch
        from test_module.benchmark_tests import image_benchmark_tests as _ibt

        _ibt.SDXL_BENCHMARK_NUM_PROMPTS = args.num_prompts
        _ibt.SDXL_SD35_BENCHMARK_NUM_PROMPTS = args.num_prompts
        # Spec test cases (e.g. ImageGenerationEvalsTest) read num_prompts
        # from targets["request"]["num_prompts"] in the suite JSON; the
        # cap clamps that value at dispatch time without touching the
        # source-of-truth suite files.
        _dispatch.SPEC_TESTS_NUM_PROMPTS_CAP = args.num_prompts
        logger.info("Overriding image benchmark + spec_tests prompt count to %d", args.num_prompts)

    logger.info(
        "Workflow=%s -> tasks=%s for model=%s device=%s port=%d output=%s",
        args.workflow,
        [t.value for t in task_types],
        ctx.model_spec.model_name,
        ctx.device.name,
        ctx.service_port,
        ctx.output_path,
    )

    accumulator = get_default_accumulator()
    accumulator.clear()
    failures = 0
    for task_type in task_types:
        logger.info("=== Dispatching %s ===", task_type.value)
        started = time.time()
        exit_code, block = run_media_task(ctx, task_type)
        elapsed = time.time() - started
        logger.info(
            "%s finished in %.1fs (exit=%d, block=%s)",
            task_type.value,
            elapsed,
            exit_code,
            block.kind if block else None,
        )
        if exit_code != 0 or block is None:
            logger.error("%s did not produce a Block.", task_type.value)
            failures += 1
            # For release we keep going so a partial report still lands.

    if not accumulator.blocks:
        logger.error("Accumulator is empty — no blocks to render.")
        return 1

    schema = accumulator.build_schema()
    result = ReportGenerator().generate(schema, ctx.output_path)
    logger.info("Wrote markdown: %s", result.markdown_path)
    logger.info("Wrote json:     %s", result.json_path)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
