# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from typing import List, Optional, Sequence

from workflows.model_spec import get_runtime_model_spec
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_types import DeviceTypes

from test_module import MediaContext

from .commands import Command, WorkflowCommand
from .execution import OrchestratorMetadata

logger = logging.getLogger(__name__)


class CommandFactory:
    """Builds the list of commands a WorkflowRunner will execute.

    Currently emits a single ``WorkflowCommand``. A future ``ServerCommand``
    will be prepended when inline server bring-up is requested.
    """

    @staticmethod
    def build(args: argparse.Namespace) -> List[Command]:
        ctx = _build_context(args)
        metadata = _build_orchestrator_metadata(args)
        commands: List[Command] = [
            WorkflowCommand(
                ctx=ctx,
                workflow_name=args.workflow,
                orchestrator_metadata=metadata,
                num_prompts=args.num_prompts,
            ),
        ]
        return commands


def _build_context(args: argparse.Namespace) -> MediaContext:
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


def _resolve_eval_config(model_name: str):
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


def _build_orchestrator_metadata(args: argparse.Namespace) -> OrchestratorMetadata:
    runtime_config = _load_runtime_config(args.runtime_model_spec_json)
    return OrchestratorMetadata(
        server_mode=_resolve_server_mode(args, runtime_config),
        run_command=_capture_run_command(),
        runtime_model_spec_json=args.runtime_model_spec_json,
    )


def _load_runtime_config(path: Optional[str]) -> Optional[RuntimeConfig]:
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
    if runtime_config is not None:
        return "docker" if runtime_config.docker_server else "API"
    return "docker" if args.docker_server else "API"


def _capture_run_command(argv: Optional[Sequence[str]] = None) -> str:
    parts = list(sys.argv if argv is None else argv)
    return "python " + shlex.join(parts)


__all__ = ["CommandFactory"]
