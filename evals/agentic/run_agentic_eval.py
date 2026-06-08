#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def _add_swebench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--sweagent-subset", default="verified")
    parser.add_argument(
        "--agent-backend",
        choices=["swe-agent", "mini-swe-agent"],
        default="mini-swe-agent",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sweagent-config", default="config/default.yaml")
    parser.add_argument("--mini-config", default="swebench.yaml")
    parser.add_argument("--mini-model-class", default="litellm")
    parser.add_argument("--mini-last-n-observations", type=int, default=15)
    parser.add_argument("--mini-environment-class", default="docker")
    parser.add_argument("--n-concurrent-trials", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-input-tokens", type=int, default=200 * 1024)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--completion-kwargs-json", default="{}")
    parser.add_argument("--swebench-timeout-sec", type=int)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--random-delay-multiplier", type=float, default=0.3)
    parser.add_argument(
        "--instance-id", action="append", default=[], dest="instance_ids"
    )
    parser.add_argument(
        "--score-existing-predictions",
        action="store_true",
        help="Skip patch generation and score output-dir/predictions.jsonl",
    )


def _add_terminal_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--n-concurrent-trials", type=int, default=1)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--environment-type", default="docker")
    parser.add_argument("--agent-kwargs-json", default="{}")
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--override-cpus", type=int)
    parser.add_argument("--override-memory-mb", type=int)
    parser.add_argument("--timeout-multiplier", type=float)
    parser.add_argument("--agent-timeout-sec", type=float)
    parser.add_argument(
        "--include-task-name", action="append", default=[], dest="task_names"
    )
    parser.add_argument(
        "--exclude-task-name", action="append", default=[], dest="exclude_task_names"
    )
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yes", action=argparse.BooleanOptionalAction, default=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run agentic evaluations")
    subparsers = parser.add_subparsers(dest="backend", required=True)

    _add_swebench_args(subparsers.add_parser("swebench"))
    _add_terminal_bench_args(subparsers.add_parser("terminal-bench"))

    args = parser.parse_args()

    if args.backend == "swebench":
        from evals.agentic.swebench import SWEbenchRunConfig, run

        config = SWEbenchRunConfig(
            task_name=args.task_name,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            sweagent_subset=args.sweagent_subset,
            agent_backend=args.agent_backend,
            model_name=args.model_name,
            api_base=args.api_base,
            output_dir=args.output_dir,
            sweagent_config=args.sweagent_config,
            mini_config=args.mini_config,
            mini_model_class=args.mini_model_class,
            mini_last_n_observations=args.mini_last_n_observations,
            mini_environment_class=args.mini_environment_class,
            n_concurrent_trials=args.n_concurrent_trials,
            max_workers=args.max_workers,
            n_tasks=args.n_tasks,
            temperature=args.temperature,
            top_p=args.top_p,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            completion_kwargs=json.loads(args.completion_kwargs_json),
            swebench_timeout_sec=args.swebench_timeout_sec,
            shuffle=args.shuffle,
            random_delay_multiplier=args.random_delay_multiplier,
            score_existing_predictions=args.score_existing_predictions,
            instance_ids=args.instance_ids,
        )
        return run(config)

    if args.backend == "terminal-bench":
        from evals.agentic.terminal_bench import TerminalBenchRunConfig, run

        config = TerminalBenchRunConfig(
            task_name=args.task_name,
            dataset=args.dataset,
            agent=args.agent,
            model_name=args.model_name,
            jobs_dir=args.jobs_dir,
            api_base=args.api_base,
            n_concurrent_trials=args.n_concurrent_trials,
            n_attempts=args.n_attempts,
            environment_type=args.environment_type,
            agent_kwargs=json.loads(args.agent_kwargs_json),
            n_tasks=args.n_tasks,
            override_cpus=args.override_cpus,
            override_memory_mb=args.override_memory_mb,
            timeout_multiplier=args.timeout_multiplier,
            agent_timeout_sec=args.agent_timeout_sec,
            task_names=args.task_names,
            exclude_task_names=args.exclude_task_names,
            quiet=args.quiet,
            yes=args.yes,
        )
        return run(config)

    return 1


if __name__ == "__main__":
    sys.exit(main())
