#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for agentic evals.

Selects/creates the dedicated ``EVALS_AGENTIC`` virtual environment and
re-execs ``run_workflows.py`` inside it, then forwards every CLI argument
verbatim.

Usage (all flags are passed straight through to run_workflows.py):
    python launchers/run_agentic.py \
        --model Qwen3.6-27B --workflow agentic --device gpu \
        --service-port 8000 --runtime-model-spec-json /tmp/qwen36_agentic_nightly.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from _launcher_common import setup_venv_and_exec

logger = logging.getLogger("tt_agentic_launcher")


def _parse_launcher_args(argv: list[str]) -> argparse.Namespace:
    """Parse only the flags needed to choose/setup the agentic venv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--requirements-json", default=None)
    args, _ = parser.parse_known_args(argv)
    if args.workflow != "agentic":
        parser.error(
            "run_agentic.py requires --workflow agentic "
            f"(got --workflow {args.workflow})."
        )
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Unlike the top-level run.py entry point, an external-endpoint launch plan
    # invokes this launcher directly.  Establish the uv bootstrap precondition
    # before VenvConfig.setup() tries to execute UV_EXEC from the repo-local
    # bootstrap venv.  This is intentionally idempotent for warm CI checkouts.
    from workflows.bootstrap_uv import bootstrap_uv
    from workflow_module.model_catalog import get_model_spec_provider
    from workflows.workflow_types import WorkflowVenvType

    bootstrap_uv()
    args = _parse_launcher_args(sys.argv[1:])
    if args.requirements_json:
        # This is a separate process from run.py, so the document has to be
        # loaded again for the provider to synthesize an off-catalog spec.
        from workflow_module.requirements_schema import load_requirements
        from workflows.requirements_cli import register_requirements_providers

        register_requirements_providers(load_requirements(args.requirements_json))
    # EVALS_AGENTIC setup depends on the model, so resolve the spec first.
    model_spec = get_model_spec_provider().resolve(model=args.model, device=args.device)
    return setup_venv_and_exec(
        WorkflowVenvType.EVALS_AGENTIC,
        logger,
        "agentic evals",
        model_spec=model_spec,
    )


if __name__ == "__main__":
    sys.exit(main())
