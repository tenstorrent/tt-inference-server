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

Prefix-caching benchmark (LLM-only, --workflow benchmarks):
    python tt-inference-server-v2/run.py \
        --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
        --prefix-cache --prefix-cache-preset ci --service-port 8000 \
        --jwt-secret "$JWT_SECRET"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
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
    PrefixCacheOptions,
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
        default=_V2_ROOT / "output",
        help="Where to write the rendered report (markdown + json).",
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

    # ----- Prefix-caching benchmark (LLM-only) -----------------------
    # When --prefix-cache is set, BenchmarksWorkflow swaps its default
    # task list for the AIPerf prefix-cache sweep. Validated below to
    # require --workflow benchmarks.
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help=(
            "Switch the benchmarks workflow to the AIPerf prefix-caching "
            "scenario sweep (shared_system, prefix_pool, multi_turn, "
            "baseline, mooncake_trace). Captures vLLM "
            "prefix_cache_hits/queries via Prometheus and reports "
            "P50/P95/P99 for TTFT/TPOT/ITL/E2EL alongside cache hit-rate. "
            "Requires --workflow benchmarks."
        ),
    )
    parser.add_argument(
        "--prefix-cache-preset",
        type=str,
        choices=["ci", "full"],
        default="full",
        help=(
            "Preset for --prefix-cache (default: full). 'ci' is a short "
            "regression-friendly sweep, 'full' is the comprehensive serving "
            "validation sweep."
        ),
    )
    parser.add_argument(
        "--prefix-cache-scenarios",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of prefix-cache scenarios to run "
            "(any of: shared_system, prefix_pool, multi_turn, baseline, "
            "mooncake_trace). When unset, every scenario from the preset "
            "is run."
        ),
    )
    parser.add_argument(
        "--prefix-cache-arrival",
        type=str,
        choices=["constant", "poisson", "gamma"],
        default=None,
        help=(
            "Override the arrival pattern for every prefix-cache run. For "
            "bursty/clustered traffic use 'gamma' and set "
            "arrival_smoothness < 1.0 in the manifest. Default is the "
            "per-scenario value from the preset."
        ),
    )
    parser.add_argument(
        "--prefix-cache-request-rate",
        type=float,
        default=None,
        help="Override the target request rate (req/s) for every prefix-cache run.",
    )
    parser.add_argument(
        "--prefix-cache-scenarios-json",
        type=str,
        default=None,
        help=(
            "Path to a custom prefix-cache scenarios JSON file. See "
            "llm_module/prefix_cache/manifest.json for the schema."
        ),
    )
    parser.add_argument(
        "--prefix-cache-trace",
        type=str,
        default=None,
        help=(
            "Path to a mooncake-format JSONL trace file. When set, every "
            "mooncake_trace scenario in the preset uses this trace instead "
            "of the manifest default. See "
            "https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md"
        ),
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        default=None,
        help=(
            "JWT secret for prefix-cache runs that hit an inference server "
            "behind JWT auth. Mints a 'debug-test' token internally and sets "
            "OPENAI_API_KEY for AIPerf. Reads $JWT_SECRET when omitted."
        ),
    )

    args = parser.parse_args()
    if args.prefix_cache and args.workflow != "benchmarks":
        parser.error(
            "--prefix-cache currently requires --workflow benchmarks "
            f"(got --workflow {args.workflow})."
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


def _mint_jwt_if_secret(jwt_secret_arg: Optional[str]) -> str:
    """Mint a ``debug-test`` JWT and export it as ``OPENAI_API_KEY``.

    Looks at the ``--jwt-secret`` arg first, then ``$JWT_SECRET``. When
    no secret is supplied, returns the empty string (auth disabled).
    Matches the v1 prefix-cache behaviour so the same servers work in
    both entry points.
    """
    secret = jwt_secret_arg or os.getenv("JWT_SECRET", "")
    if not secret:
        return ""
    try:
        import jwt as _jwt
    except ImportError:
        logger.warning(
            "PyJWT is not installed; --jwt-secret was supplied but no token "
            "will be minted. Install pyjwt to enable JWT-protected servers."
        )
        return ""
    payload = {
        "team_id": "tenstorrent",
        "token_id": "debug-test",
        "exp": int(_dt.datetime.now(_dt.timezone.utc).timestamp()) + 24 * 3600,
    }
    encoded = _jwt.encode(payload, secret, algorithm="HS256")
    os.environ["OPENAI_API_KEY"] = encoded
    logger.info("Minted debug-test JWT and exported as OPENAI_API_KEY.")
    return encoded


def _build_prefix_cache_options(
    args: argparse.Namespace,
) -> Optional[PrefixCacheOptions]:
    if not args.prefix_cache:
        return None
    return PrefixCacheOptions(
        preset=args.prefix_cache_preset,
        scenarios=args.prefix_cache_scenarios,
        arrival_pattern=args.prefix_cache_arrival,
        request_rate=args.prefix_cache_request_rate,
        scenarios_json=args.prefix_cache_scenarios_json,
        trace_path=args.prefix_cache_trace,
        auth_token=_mint_jwt_if_secret(args.jwt_secret),
    )


def _build_orchestrator_metadata(args: argparse.Namespace) -> OrchestratorMetadata:
    runtime_config = _load_runtime_config(args.runtime_model_spec_json)
    return OrchestratorMetadata(
        server_mode=_resolve_server_mode(args, runtime_config),
        run_command=_capture_run_command(),
        runtime_model_spec_json=args.runtime_model_spec_json,
        prefix_cache=_build_prefix_cache_options(args),
    )


def _maybe_reexec_in_prefix_cache_venv(args: argparse.Namespace) -> None:
    """For ``--prefix-cache``: materialize ``V2_PREFIX_CACHE`` and re-exec.

    The AIPerf driver shells out to ``sys.executable -m aiperf``, so the
    interpreter running this script must have aiperf + pyjwt installed.
    Rather than make the user manage a venv by hand we materialize the
    declared ``WorkflowVenvType.V2_PREFIX_CACHE`` venv on demand and
    ``os.execv`` into it, preserving argv verbatim. Idempotent: when we
    are already running inside the venv this is a no-op.
    """
    if not args.prefix_cache:
        return

    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    venv_config = VENV_CONFIGS[WorkflowVenvType.V2_PREFIX_CACHE]
    venv_python = venv_config.venv_python
    if Path(sys.executable).resolve() == venv_python.resolve():
        logger.info("Already inside V2_PREFIX_CACHE venv (%s).", sys.executable)
        return

    logger.info("Ensuring V2_PREFIX_CACHE venv at %s ...", venv_config.venv_path)
    model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
    venv_config.setup(model_spec=model_spec)

    logger.info("Re-executing inside V2_PREFIX_CACHE venv: %s", venv_python)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


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

    _maybe_reexec_in_prefix_cache_venv(args)

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
