#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""v2 CLI entry point — drives a workflow against a running inference server.

Usage:
    python tt-inference-server-v2/run.py \
        --model stable-diffusion-xl-base-1.0 --workflow release \
        --device n150 --service-port 8000

Prefix-caching benchmark (LLM-only, --workflow benchmarks):
    This entry point has no import-time side effects, so it must run inside the
    dedicated ``V2_PREFIX_CACHE`` venv. Use the thin launcher
    ``run_prefix_cache.py`` (which selects/creates that venv and re-execs this
    script) rather than invoking run.py directly:

        python tt-inference-server-v2/run_prefix_cache.py \
            --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
            --prefix-cache --prefix-cache-preset ci --service-port 8000 \
            --jwt-secret "$JWT_SECRET"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from workflows.model_spec import MODEL_SPECS  # noqa: E402
from workflows.workflow_types import DeviceTypes  # noqa: E402

from workflow_module import (  # noqa: E402
    Command,
    CommandFactory,
    CommandResult,
)

logger = logging.getLogger("tt_v2_runner")

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class WorkflowRunner:
    """Thin executor of pre-built commands.

    Iterates the command list, calls ``execute()`` on each, and collects
    results. Has no knowledge of what any command actually does.
    """

    def __init__(self, commands: Sequence[Command]) -> None:
        self.commands: List[Command] = list(commands)
        self.results: List[CommandResult] = []

    def run(self) -> int:
        for cmd in self.commands:
            logger.info("→ command=%s", cmd.name)
            result = cmd.execute()
            self.results.append(result)
            if not result.succeeded:
                logger.error(
                    "❌ command=%s rc=%d error=%s",
                    cmd.name,
                    result.return_code,
                    result.error,
                )
                return result.return_code
            logger.info("✅ command=%s rc=0", cmd.name)
        return 0


def parse_args() -> argparse.Namespace:
    from workflow_module import WORKFLOW_REGISTRY

    valid_models = sorted({spec.model_name for spec in MODEL_SPECS.values()})
    valid_devices = sorted({d.name.lower() for d in DeviceTypes})
    valid_workflows = sorted(WORKFLOW_REGISTRY)

    parser = argparse.ArgumentParser(
        description=(
            "Standalone CLI for the v2 WorkflowRunner — drives a v2 workflow "
            "against an already-running inference server. For full server "
            "bring-up + workflow runs, invoke through v1 /run.py instead."
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
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the workflow N times, keeping each run's report under "
            "<output>/<model>_<device>_<workflow>/run_NN/, then write an "
            "aggregated benchmark summary (mean/median/stdev/percentiles + "
            "acceptance on the means) into .../summary/. Default 1 (no summary)."
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

    # ----- Prefix-caching benchmark (LLM-only) -----------------------
    # When --prefix-cache is set, BenchmarksWorkflow swaps its default
    # media-task dispatch for the AIPerf prefix-cache scenario sweep (wired
    # through CommandFactory -> OrchestratorMetadata.prefix_cache). Validated
    # below to require --workflow benchmarks. Run via run_prefix_cache.py so
    # the V2_PREFIX_CACHE venv is in place before these heavy deps are needed.
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help=(
            "Switch the benchmarks workflow to the AIPerf prefix-caching "
            "scenario sweep (shared_system, prefix_pool, multi_turn, "
            "baseline, mooncake_trace). Captures vLLM "
            "prefix_cache_hits/queries via Prometheus and reports "
            "P50/P95/P99 for TTFT/TPOT/ITL/E2EL alongside cache hit-rate. "
            "Requires --workflow benchmarks. Launch through run_prefix_cache.py."
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
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.prefix_cache and args.workflow != "benchmarks":
        parser.error(
            "--prefix-cache currently requires --workflow benchmarks "
            f"(got --workflow {args.workflow})."
        )
    if args.output_dir is None:
        args.output_dir = (
            _REPO_ROOT / "workflow_logs" / "reports_output" / args.workflow
        )
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=_LOG_LEVELS[args.log_level],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    commands = CommandFactory.build(args)
    runner = WorkflowRunner(commands)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
