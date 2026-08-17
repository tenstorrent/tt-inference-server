#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Temporary v2 orchestrator - needed for running tests against the inference server.

Usage:
    python tt-inference-server-v2/run.py \
        --model stable-diffusion-xl-base-1.0 --workflow benchmarks \
        --device n150 --service-port 8000

    python tt-inference-server-v2/run.py \
        --model stable-diffusion-xl-base-1.0 --workflow release \
        --device n150 --service-port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from workflows.model_spec import MODEL_SPECS, get_runtime_model_spec  # noqa: E402
from workflows.runtime_config import RuntimeConfig  # noqa: E402
from workflows.workflow_types import DeviceTypes  # noqa: E402

from test_module import MediaContext  # noqa: E402
from workflow_module import (  # noqa: E402
    OrchestratorMetadata,
    WorkflowResult,
    get_default_accumulator,
    get_workflow_class,
)

logger = logging.getLogger("tt_v2_run")

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def parse_args() -> argparse.Namespace:
    from workflow_module import WORKFLOW_REGISTRY

    valid_models = sorted({spec.model_name for spec in MODEL_SPECS.values()})
    valid_devices = sorted({d.name.lower() for d in DeviceTypes})
    valid_workflows = sorted(WORKFLOW_REGISTRY)

    parser = argparse.ArgumentParser(
        description=(
            "Drive v2 evals/benchmarks against a running inference server "
            "and emit a combined report."
        ),
        epilog="Available models:\n  " + "\n  ".join(valid_models),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=valid_models)
    parser.add_argument("--workflow", required=True, choices=valid_workflows)
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
        default=None,
        help=(
            "Where to write the rendered report (markdown + json). "
            "Defaults to <repo>/workflow_logs/reports_output/<workflow>/."
        ),
    )
    parser.add_argument(
        "--docker-server",
        action="store_true",
        help=(
            "Record server_mode=docker in the report metadata. Fallback "
            "when --runtime-model-spec-json is not supplied."
        ),
    )
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        default=None,
        help="Path to the runtime model spec JSON written by the server launcher.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            _REPO_ROOT / "workflow_logs" / "reports_output" / args.workflow
        )
    return args


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
    model_spec.cli_args["device"] = args.device
    if args.num_prompts is not None:
        model_spec.cli_args["sdxl_num_prompts"] = max(2, args.num_prompts)

    device = DeviceTypes.from_string(args.device)

    output_path = args.output_dir / f"{args.model}_{args.device}_{args.workflow}"
    output_path.mkdir(parents=True, exist_ok=True)

    eval_cfg = _resolve_eval_config(args.model)
    all_params = eval_cfg if eval_cfg is not None else []

    return MediaContext(
        all_params=all_params,
        model_spec=model_spec,
        device=device,
        output_path=str(output_path),
        service_port=args.service_port,
        spec_tests_num_prompts_cap=args.num_prompts,
    )


def _load_runtime_config(path: Optional[str]) -> Optional[RuntimeConfig]:
    """Best-effort load; returns ``None`` on missing/malformed input."""
    if not path:
        return None
    try:
        return RuntimeConfig.from_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.warning(
            "Could not load runtime_model_spec_json=%r (%s); "
            "falling back to CLI flags for server_mode.",
            path,
            e,
        )
        return None


def _resolve_server_mode(
    args: argparse.Namespace, runtime_config: Optional[RuntimeConfig]
) -> str:
    """Spec JSON wins over ``--docker-server`` when present."""
    if runtime_config is not None:
        return "docker" if runtime_config.docker_server else "API"
    return "docker" if args.docker_server else "API"


def _capture_run_command(argv: Optional[Sequence[str]] = None) -> str:
    """Paste-runnable reproduction of the orchestrator invocation."""
    parts = list(sys.argv if argv is None else argv)
    return "python " + shlex.join(parts)


def _build_orchestrator_metadata(args: argparse.Namespace) -> OrchestratorMetadata:
    runtime_config = _load_runtime_config(args.runtime_model_spec_json)
    return OrchestratorMetadata(
        server_mode=_resolve_server_mode(args, runtime_config),
        run_command=_capture_run_command(),
        runtime_model_spec_json=args.runtime_model_spec_json,
    )


def _apply_num_prompts_override(num_prompts: Optional[int]) -> None:
    if num_prompts is None:
        return
    from test_module.benchmark_tests import image_benchmark_tests as _ibt

    _ibt.SDXL_BENCHMARK_NUM_PROMPTS = num_prompts
    _ibt.SDXL_SD35_BENCHMARK_NUM_PROMPTS = num_prompts
    logger.info(
        "Overriding image benchmark + spec_tests prompt count to %d", num_prompts
    )


def _log_workflow_summary(result: WorkflowResult) -> None:
    logger.info(
        "Workflow %s finished: rc=%d (%d task(s))",
        result.workflow_name,
        result.return_code,
        len(result.task_outcomes),
    )
    for outcome in result.task_outcomes:
        logger.info(
            "  %s task=%s rc=%d elapsed=%.1fs block=%s",
            "✓" if outcome.succeeded else "✘",
            outcome.task_type,
            outcome.exit_code,
            outcome.elapsed_seconds,
            outcome.block_kind,
        )
    if result.error:
        logger.error("Workflow error: %s", result.error)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=_LOG_LEVELS[args.log_level],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ctx = build_context(args)
    _apply_num_prompts_override(args.num_prompts)

    get_default_accumulator().clear()
    workflow_cls = get_workflow_class(args.workflow)
    workflow = workflow_cls(
        ctx,
        orchestrator_metadata=_build_orchestrator_metadata(args),
    )
    result = workflow.run()
    _log_workflow_summary(result)
    return result.return_code


if __name__ == "__main__":
    sys.exit(main())
