#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the agentic trace-replay benchmark.

Selects/creates the dedicated ``AGENTIC_TRACES`` virtual environment -- which
includes cloning InferenceX at the ModelSpec's pinned revision -- and re-execs
``run_workflows.py`` inside it, forwarding every CLI argument verbatim.

Usage (all flags are passed straight through to run_workflows.py):
    python launchers/run_agentic_traces.py \
        --model Kimi-K2.7-Code --workflow agentic_traces --device super_cluster \
        --agentic-traces-mode ci --service-port 8000 \
        --runtime-model-spec-json /tmp/kimi_agentic_traces.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from _launcher_common import setup_venv_and_exec

logger = logging.getLogger("tt_agentic_traces_launcher")


def _parse_launcher_args(argv: list[str]) -> argparse.Namespace:
    """Parse only the flags needed to choose/setup the agentic-traces venv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--runtime-model-spec-json", default=None)
    args, _ = parser.parse_known_args(argv)
    if args.workflow != "agentic_traces":
        parser.error(
            "run_agentic_traces.py requires --workflow agentic_traces "
            f"(got --workflow {args.workflow})."
        )
    return args


def _resolve_model_spec(args: argparse.Namespace):
    """Resolve the spec whose ``model_id`` keys the agentic-traces config.

    Prefers the runtime spec JSON that ``run.py`` already resolved: re-resolving
    from the catalog by (model, device) alone falls back to whichever
    device_model_spec has ``default_impl=True``, which would silently pick a
    different ``model_id`` -- and therefore a different pinned InferenceX ref --
    for a model with more than one impl on the same device.
    """
    from workflows.model_spec import ModelSpec, get_runtime_model_spec

    if args.runtime_model_spec_json:
        try:
            return ModelSpec.from_json(args.runtime_model_spec_json)
        except Exception as e:  # noqa: BLE001 - fall back to catalog resolution
            logger.warning(
                "Could not load model_spec from %s (%s); falling back to catalog "
                "resolution by (model, device).",
                args.runtime_model_spec_json,
                e,
            )
    model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
    return model_spec


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from workflows.workflow_types import WorkflowVenvType

    args = _parse_launcher_args(sys.argv[1:])
    # AGENTIC_TRACES setup checks out the ref pinned for this ModelSpec, so the
    # spec has to be resolved before the venv is materialized.
    model_spec = _resolve_model_spec(args)
    return setup_venv_and_exec(
        WorkflowVenvType.AGENTIC_TRACES,
        logger,
        "agentic traces",
        model_spec=model_spec,
    )


if __name__ == "__main__":
    sys.exit(main())
