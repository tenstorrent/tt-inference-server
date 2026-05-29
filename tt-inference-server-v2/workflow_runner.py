#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


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


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=_LOG_LEVELS[args.log_level],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    commands = CommandFactory.build_isolated(args)
    runner = WorkflowRunner(commands)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
