# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional

from workflows.model_spec import get_runtime_model_spec
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_types import DeviceTypes, InferenceEngine

from utils.url_helpers import resolve_deploy_url

from test_module import MediaContext

from .commands import Command, SummaryCommand, WorkflowCommand
from .execution import (
    LLMBenchOptions,
    LLMEvalOptions,
    OrchestratorMetadata,
    PrefixCacheOptions,
    ServingBenchOptions,
    SpecDecodeOptions,
)

# Workflows whose LLM path runs the standard-eval / perf-benchmark child.
_LLM_BENCH_WORKFLOWS = frozenset({"benchmarks", "release"})
_LLM_EVAL_WORKFLOWS = frozenset({"evals", "release"})

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
    runtime_config = _load_runtime_config(args.runtime_model_spec_json)

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
        runtime_config=runtime_config,
        server_url=_resolve_server_url(args, runtime_config),
    )


def _resolve_server_url(
    args: argparse.Namespace, runtime_config: Optional[RuntimeConfig]
) -> str:
    """Pick the inference-server URL to target an already-running server.

    Prefers the explicit ``--server-url`` CLI flag, then delegates to the
    shared :func:`resolve_deploy_url` (``RuntimeConfig.server_url`` propagated
    through the v2 bridge, then the ``DEPLOY_URL`` env var, then the localhost
    default). This routes v2 through the same single source of truth as every
    v1 workflow rather than re-deriving the precedence here.
    """
    explicit = getattr(args, "server_url", None)
    if explicit:
        return explicit
    return resolve_deploy_url(runtime_config)


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
    server_mode = _resolve_server_mode(args, runtime_config)
    return OrchestratorMetadata(
        server_mode=server_mode,
        run_command=_resolve_run_command(),
        runtime_model_spec_json=args.runtime_model_spec_json,
        prefix_cache=_build_prefix_cache_options(args),
        spec_decode=_build_spec_decode_options(args),
        serving_bench=_build_serving_bench_options(args),
        llm_bench=_build_llm_bench_options(args),
        llm_eval=_build_llm_eval_options(args),
    )


def _build_llm_bench_options(args: argparse.Namespace) -> Optional[LLMBenchOptions]:
    """Translate ``--tools`` into ``LLMBenchOptions`` for the LLM perf benchmark.

    Built for ``--workflow benchmarks`` and ``--workflow release`` (whose
    benchmark child runs the same sweep). The prefix-cache / spec-decode
    variants have their own options and are excluded.
    """
    if getattr(args, "workflow", None) not in _LLM_BENCH_WORKFLOWS:
        return None
    if getattr(args, "prefix_cache", False):
        return None
    if getattr(args, "spec_decode", False):
        return None
    return LLMBenchOptions(
        tools=getattr(args, "tools", None) or "vllm",
        auth_token=_resolve_auth_token(args),
        venv_python=_release_bench_venv_python(args),
    )


def _release_bench_venv_python(args: argparse.Namespace) -> Optional[str]:
    """Tool-venv interpreter for the release benchmark child.

    A standalone benchmarks run is already inside the tool venv (run_llm_bench.py
    re-execs there), so its driver uses ``sys.executable`` — return ``None``.
    A release run executes in the V2_RUN_SCRIPT venv, so pin the default
    perf-tool venv (V2_LLM_VLLM); the v2 bridge provisions it before run.py.
    """
    if getattr(args, "workflow", None) != "release":
        return None
    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    return str(VENV_CONFIGS[WorkflowVenvType.V2_LLM_VLLM].venv_python)


def _build_llm_eval_options(args: argparse.Namespace) -> Optional[LLMEvalOptions]:
    """Bearer-token plumbing for the LLM standard-eval child (evals/release)."""
    if getattr(args, "workflow", None) not in _LLM_EVAL_WORKFLOWS:
        return None
    return LLMEvalOptions(
        auth_token=_resolve_auth_token(args),
    )


def _build_serving_bench_options(
    args: argparse.Namespace,
) -> Optional[ServingBenchOptions]:
    if getattr(args, "workflow", None) != "serving_bench":
        return None
    return ServingBenchOptions(suites=getattr(args, "serving_bench_suites", None))


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
        auth_token=_resolve_auth_token(args),
    )


def _build_spec_decode_options(
    args: argparse.Namespace,
) -> Optional[SpecDecodeOptions]:
    """Translate the ``--spec-decode*`` CLI flags into ``SpecDecodeOptions``.

    Returns ``None`` (the default) for every non-spec-decode run, leaving
    ``BenchmarksWorkflow`` on its normal media-task dispatch. The flags are
    only present when ``run.py`` registered them, so ``getattr`` guards keep
    this safe for the image-model entry path that never adds them.
    """
    if not getattr(args, "spec_decode", False):
        return None
    return SpecDecodeOptions(
        preset=args.spec_decode_preset,
        warmup_requests=args.spec_decode_warmup_requests,
        auth_token=_resolve_auth_token(args),
    )


def _resolve_auth_token(args: argparse.Namespace) -> str:
    """Resolve the bearer token the eval/benchmark clients send.

    Forge/media servers (tt-media-server) check a *literal* ``Bearer $API_KEY``
    (default ``your-secret-key``, see ``security/api_key_checker.py``) and do
    NOT decode JWTs; only the vLLM/tt-metal server validates a JWT. Mirrors the
    engine branch in v1 ``evals/run_evals.py`` — a JWT sent to a forge/media
    server 401s. Falls back to the JWT path when the spec can't be resolved.
    """
    try:
        spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
        engine = getattr(spec.inference_engine, "value", spec.inference_engine)
    except Exception:  # pragma: no cover - defensive
        engine = None
    if engine in (InferenceEngine.FORGE.value, InferenceEngine.MEDIA.value):
        return os.getenv("VLLM_API_KEY") or os.getenv("API_KEY") or "your-secret-key"
    return _mint_jwt_if_secret(getattr(args, "jwt_secret", None))


def _mint_jwt_if_secret(jwt_secret_arg: Optional[str]) -> str:
    """Mint a ``debug-test`` JWT and export it as ``OPENAI_API_KEY``.

    Looks at the ``--jwt-secret`` arg first, then ``$JWT_SECRET``. When no
    secret is supplied, returns the empty string (auth disabled). Matches the
    inference server's expected debug token for JWT auth. Used for the vLLM
    (tt-metal) server; forge/media servers go through the literal-key branch
    in :func:`_resolve_auth_token`.
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
    # The tt-metal/vLLM server does NOT decode this JWT at request time. In
    # run_vllm_api_server.py -> handle_secrets() it precomputes a STATIC key
    # VLLM_API_KEY = get_encoded_api_key(JWT_SECRET), i.e.
    #   jwt.encode({"team_id": "tenstorrent", "token_id": "debug-test"}, secret,
    #              algorithm="HS256")
    # (no exp claim), and vLLM's AuthenticationMiddleware compares the incoming
    # bearer to that string by exact sha256/compare_digest byte-equality. So the
    # token minted here MUST be byte-identical to the server's key — any extra
    # claim (e.g. "exp") changes the encoded string and 401s every request.
    payload = {
        "team_id": "tenstorrent",
        "token_id": "debug-test",
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


_V1_RUN_COMMAND_ENV = "TT_V1_RUN_COMMAND"


def _resolve_run_command() -> str:
    """Resolve the ``run_command`` recorded in the report metadata."""
    propagated = os.environ.get(_V1_RUN_COMMAND_ENV)
    if propagated:
        return propagated
    return "python " + shlex.join(sys.argv)


__all__ = ["CommandFactory"]
