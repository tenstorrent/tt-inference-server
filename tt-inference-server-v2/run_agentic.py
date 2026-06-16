#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for v2 agentic evals.

Selects/creates the dedicated ``EVALS_AGENTIC`` virtual environment and
re-execs ``run.py`` inside it, then forwards every CLI argument verbatim.

Usage (all flags are passed straight through to run.py):
    python tt-inference-server-v2/run_agentic.py \
        --model Qwen3.6-27B --workflow agentic --device gpu \
        --service-port 8000 --runtime-model-spec-json /tmp/qwen36_agentic_nightly.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("tt_v2_agentic_launcher")


def _parse_launcher_args(argv: list[str]) -> argparse.Namespace:
    """Parse only the flags needed to choose/setup the agentic venv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--device", required=True)
    args, _ = parser.parse_known_args(argv)
    if args.workflow != "agentic":
        parser.error(
            "run_agentic.py requires --workflow agentic "
            f"(got --workflow {args.workflow})."
        )
    return args


def _ensure_agentic_venv(args: argparse.Namespace) -> Path:
    """Materialize ``EVALS_AGENTIC`` and return its interpreter path."""
    from workflows.model_spec import get_runtime_model_spec
    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
    venv_config = VENV_CONFIGS[WorkflowVenvType.EVALS_AGENTIC]
    setup_completed = venv_config.setup(model_spec=model_spec)
    assert setup_completed, "Failed to setup venv: EVALS_AGENTIC"
    return venv_config.venv_python


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_py = _V2_ROOT / "run.py"
    args = _parse_launcher_args(sys.argv[1:])
    venv_python = _ensure_agentic_venv(args)

    logger.info("Launching run.py inside EVALS_AGENTIC venv: %s", venv_python)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(venv_python), [str(venv_python), str(run_py), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
