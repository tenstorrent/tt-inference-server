# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from workflows.model_spec import get_runtime_model_spec
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_types import DeviceTypes

from test_module import MediaContext

from .commands import Command, SummaryCommand, WorkflowCommand
from .execution import OrchestratorMetadata, PrefixCacheOptions

logger = logging.getLogger(__name__)


class CommandFactory:
    """Builds the list of commands a WorkflowRunner will execute.

    Currently emits a single ``WorkflowCommand``. A future ``ServerCommand``
    will be prepended when inline server bring-up is requested.
    """

    @staticmethod
    def build(args: argparse.Namespace) -> List[Command]:
        metadata = _build_orchestrator_metadata(args)
        repeat = max(1, int(getattr(args, "repeat", 1) or 1))
        if repeat == 1:
            return [_workflow_command(args, _build_context(args), metadata)]
        return _build_repeated_commands(args, metadata, repeat)


def _workflow_command(
    args: argparse.Namespace,
    ctx: MediaContext,
    metadata: OrchestratorMetadata,
    *,
    continue_on_failure: bool = False,
) -> WorkflowCommand:
    return WorkflowCommand(
        ctx=ctx,
        workflow_name=args.workflow,
        orchestrator_metadata=metadata,
        num_prompts=args.num_prompts,
        continue_on_failure=continue_on_failure,
    )


def _build_repeated_commands(
    args: argparse.Namespace,
    metadata: OrchestratorMetadata,
    repeat: int,
) -> List[Command]:
    """N per-run workflows into ``run_NN/`` subfolders + a final summary."""
    leaf = f"{args.model}_{args.device}_{args.workflow}"
    container = Path(args.output_dir) / leaf
    commands: List[Command] = []
    for run_index in range(1, repeat + 1):
        run_output = container / f"run_{run_index:02d}" / leaf
        ctx = _build_context(args, output_path=run_output)
        commands.append(
            _workflow_command(args, ctx, metadata, continue_on_failure=True)
        )
    commands.append(
        SummaryCommand(
            container_dir=container,
            summary_output_dir=container / "summary",
        )
    )
    return commands


def _build_context(
    args: argparse.Namespace, output_path: Optional[Path] = None
) -> MediaContext:
    model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
    model_spec.cli_args["device"] = args.device
    if args.num_prompts is not None:
        model_spec.cli_args["sdxl_num_prompts"] = max(2, args.num_prompts)

    device = DeviceTypes.from_string(args.device)

    if output_path is None:
        output_path = args.output_dir / f"{args.model}_{args.device}_{args.workflow}"
    output_path = Path(output_path)
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
        prefix_cache=_build_prefix_cache_options(args),
    )


def _build_prefix_cache_options(
    args: argparse.Namespace,
) -> Optional[PrefixCacheOptions]:
    """Translate the ``--prefix-cache*`` CLI flags into ``PrefixCacheOptions``.

    Returns ``None`` (the default) for every non-prefix-cache run, leaving
    ``BenchmarksWorkflow`` on its normal media-task dispatch. The flags are
    only present when ``run.py`` registered them, so ``getattr`` guards keep
    this safe for the image-model entry path that never adds them.
    """
    if not getattr(args, "prefix_cache", False):
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


def _mint_jwt_if_secret(jwt_secret_arg: Optional[str]) -> str:
    """Mint a ``debug-test`` JWT and export it as ``OPENAI_API_KEY``.

    Looks at the ``--jwt-secret`` arg first, then ``$JWT_SECRET``. When no
    secret is supplied, returns the empty string (auth disabled). Matches the
    inference server's expected debug token for JWT auth.
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
