# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import jwt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.auth_probe import assert_probe_ok, probe_bearer
from workflows.auth_manifest import load_bearer_or_fallback
from utils.media_clients.base_strategy_interface import BaseMediaStrategy
from utils.media_clients.media_client_factory import MediaClientFactory, MediaTaskType

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from evals.eval_config import EVAL_CONFIGS, EvalConfig, EvalTask
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import run_command
from workflows.workflow_config import (
    WORKFLOW_EVALS_CONFIG,
)
from workflows.workflow_types import (
    DeviceTypes,
    EvalLimitMode,
    ModelType,
    WorkflowVenvType,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)
SMOKE_TEST_EVAL_LIMIT = 3

# fmt: off
IMAGE_RESOLUTIONS = [
    (512, 512),
    (512, 1024),
    (1024, 512),
    (1024, 1024)
    ]
# fmt: on

EVAL_TASK_TYPES = [
    ModelType.IMAGE,
    ModelType.CNN,
    ModelType.AUDIO,
    ModelType.EMBEDDING,
    ModelType.TEXT_TO_SPEECH,
    ModelType.VIDEO,
]


def _get_limit_mode(runtime_config: Optional[RuntimeConfig]) -> Optional[EvalLimitMode]:
    if runtime_config is None or not runtime_config.limit_samples_mode:
        return None
    return EvalLimitMode.from_string(runtime_config.limit_samples_mode)


def _select_eval_config(
    eval_config: EvalConfig, runtime_config: Optional[RuntimeConfig]
) -> EvalConfig:
    eval_samples_value = (
        getattr(runtime_config, "eval_samples", None)
        if runtime_config is not None
        else None
    )
    if eval_samples_value and eval_config.tasks:
        mapping = _parse_eval_samples_mapping(eval_samples_value)
        if mapping:
            requested = set(mapping.keys())
            filtered = [t for t in eval_config.tasks if t.task_name in requested]
            if not filtered:
                available = sorted({t.task_name for t in eval_config.tasks})
                raise ValueError(
                    "--eval-samples specified task(s) "
                    f"{sorted(requested)} but none match this model's eval "
                    f"tasks {available}."
                )
            unknown = requested - {t.task_name for t in filtered}
            if unknown:
                logger.warning(
                    "--eval-samples references task(s) not configured for this "
                    "model: %s",
                    sorted(unknown),
                )
            logger.info(
                "--eval-samples filtering eval tasks down to: %s",
                [t.task_name for t in filtered],
            )
            return EvalConfig(hf_model_repo=eval_config.hf_model_repo, tasks=filtered)

    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode != EvalLimitMode.SMOKE_TEST or not eval_config.tasks:
        return eval_config

    selected_task = eval_config.tasks[0]
    logger.info(
        "Smoke-test mode enabled; running only first eval task: %s",
        selected_task.task_name,
    )
    return EvalConfig(hf_model_repo=eval_config.hf_model_repo, tasks=[selected_task])


def _resolve_eval_limit(
    task: EvalTask, runtime_config: Optional[RuntimeConfig]
) -> Optional[object]:
    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode is None:
        return None
    if limit_mode == EvalLimitMode.SMOKE_TEST:
        return SMOKE_TEST_EVAL_LIMIT
    return task.limit_samples_map.get(limit_mode)


def _parse_eval_samples_mapping(value: Optional[str]) -> Optional[dict]:
    """Parse the --eval-samples value into a dict.

    Accepts either a JSON string of shape ``{"task_name": [int, ...], ...}``
    or a path to a JSON file containing the same shape.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        path = Path(value)
        if not path.is_file():
            raise ValueError(
                f"--eval-samples value is not valid JSON and not an existing file: {value}"
            )
        parsed = json.loads(path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(
            "--eval-samples must decode to a JSON object mapping task_name -> "
            f"[int, ...]; got {type(parsed).__name__}"
        )
    return parsed


def _resolve_eval_samples(
    task: EvalTask, runtime_config: Optional[RuntimeConfig]
) -> Optional[str]:
    """Resolve --eval-samples to a per-task JSON string for lm-eval's --samples.

    Returns ``None`` when eval_samples is unset, when the current task has no
    entry in the user-supplied mapping, or when the task uses a vision/audio
    (lmms-eval) backend that does not support the --samples flag.
    """
    eval_samples_value = (
        getattr(runtime_config, "eval_samples", None)
        if runtime_config is not None
        else None
    )
    if not eval_samples_value:
        return None
    if task.workflow_venv_type in (
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ):
        logger.warning(
            "--eval-samples is not supported for vision/audio evals; "
            "ignoring for task %s",
            task.task_name,
        )
        return None
    mapping = _parse_eval_samples_mapping(eval_samples_value)
    if mapping is None:
        return None
    indices = mapping.get(task.task_name)
    if indices is None:
        logger.info(
            "--eval-samples has no entry for task %s; skipping --samples for this task",
            task.task_name,
        )
        return None
    logger.info(
        "Filtering task %s to %d doc_id(s) via --samples",
        task.task_name,
        len(indices),
    )
    return json.dumps({task.task_name: indices})


def _check_media_server_health(model_spec, device, output_path, service_port):
    """
    Check if media server is healthy using DeviceLivenessTest.

    Args:
        model_spec: Model specification
        device: Device type
        output_path: Output path for logs
        service_port: Service port number

    Returns:
        tuple[bool, str]: (is_healthy, runner_in_use)

    Raises:
        RuntimeError: If media server is not healthy after all retry attempts
    """

    # Create a minimal strategy instance just for health check
    class HealthCheckStrategy(BaseMediaStrategy):
        def run_eval(self):
            pass

        def run_benchmark(self, num_calls):
            pass

    health_checker = HealthCheckStrategy(
        all_params=None,
        model_spec=model_spec,
        device=device,
        output_path=output_path,
        service_port=service_port,
    )

    is_healthy, runner_in_use = health_checker.get_health()
    if not is_healthy:
        raise RuntimeError("❌ Media server is not healthy. Aborting evaluations.")

    logger.info(f"✅ Media server is healthy. Runner in use: {runner_in_use}")
    return is_healthy, runner_in_use


def _setup_openai_api_key(args, logger):
    """Setup OPENAI_API_KEY environment variable based on JWT secret or API key.
    Args:
        args: Parsed command line arguments
        logger: Logger instance
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        api_key = "your-secret-key"
        logger.warning(
            "API_KEY is not set. Using a default key for media server auth. "
            "Set API_KEY in .env or as an environment variable."
        )
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("OPENAI_API_KEY environment variable set.")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM evals")
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for evaluation output",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on",
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name",
        required=False,
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HF_TOKEN",
        default=os.getenv("HF_TOKEN", ""),
    )
    ret_args = parser.parse_args()
    return ret_args


def build_eval_command(
    task: EvalTask,
    model_spec,
    device,
    output_path,
    service_port,
    runtime_config=None,
) -> List[str]:
    """
    Build the command for lm_eval by templating command-line arguments using properties
    from the given evaluation task and model configuration.
    """
    # Audio models use tt-media-server which has endpoints at /audio (not /v1/audio)
    # Other models use vLLM which has endpoints at /v1
    if task.workflow_venv_type == WorkflowVenvType.EVALS_AUDIO:
        base_url = f"http://127.0.0.1:{service_port}"
    else:
        base_url = f"http://127.0.0.1:{service_port}/v1"
    eval_class = task.eval_class
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    if task.use_chat_api:
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/completions"

    optional_model_args = []
    if task.max_concurrent:
        optional_model_args.append(f"num_concurrent={task.max_concurrent}")

    # lm-eval (text) expects full completions api route in base_url
    # lmms-eval (vision) expects base_url WITHOUT the endpoint path
    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_META,
        WorkflowVenvType.EVALS_VISION,
    ]:
        _base_url = base_url
    else:
        _base_url = api_url

    # Set OPENAI_API_BASE for vision and audio models
    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ]:
        os.environ["OPENAI_API_BASE"] = base_url

    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ]:
        lm_eval_exec = task_venv_config.venv_path / "bin" / "lmms-eval"
    else:
        lm_eval_exec = task_venv_config.venv_path / "bin" / "lm_eval"

    model_kwargs_list = [f"{k}={v}" for k, v in task.model_kwargs.items()]
    model_kwargs_list += optional_model_args
    model_kwargs_str = ",".join(model_kwargs_list)

    # build gen_kwargs string
    gen_kwargs_list = [f"{k}={v}" for k, v in task.gen_kwargs.items()]
    gen_kwargs_str = ",".join(gen_kwargs_list)

    # set output_dir
    # results go to {output_dir_path}/{hf_repo}/results_{timestamp}
    output_dir_path = Path(output_path) / f"eval_{model_spec.model_id}"

    # fmt: off
    if task.workflow_venv_type == WorkflowVenvType.EVALS_VISION:
        cmd = [
            str(lm_eval_exec),
            "--tasks", task.task_name,
            "--model", eval_class,
            "--model_args", (
                f"model_version={model_spec.hf_model_repo},"
                f"base_url={_base_url},"
                f"tokenizer_backend={task.tokenizer_backend},"
                f"{model_kwargs_str}"
            ),
            "--gen_kwargs", gen_kwargs_str,
            "--output_path", output_dir_path,
            "--seed", task.seed,
            "--num_fewshot", task.num_fewshot,
            "--batch_size", task.batch_size,
            "--log_samples",
            "--show_config",
        ]
    elif task.workflow_venv_type == WorkflowVenvType.EVALS_AUDIO:
        cmd = [
            str(lm_eval_exec),
            "--model", eval_class,
            "--model_args", (
                f"model={model_spec.hf_model_repo},"
                f"base_url={base_url},"
                f"{model_kwargs_str}"
            ),
            "--tasks", task.task_name,
            "--batch_size", str(task.batch_size),
            "--output_path", str(output_dir_path),
            "--log_samples",
        ]
    else:
        cmd = [
            str(lm_eval_exec),
            "--tasks", task.task_name,
            "--model", eval_class,
            "--model_args", (
                f"model={model_spec.hf_model_repo},"
                f"base_url={_base_url},"
                f"tokenizer_backend={task.tokenizer_backend},"
                f"{model_kwargs_str}"
            ),
            "--gen_kwargs", gen_kwargs_str,
            "--output_path", output_dir_path,
            "--seed", task.seed,
            "--num_fewshot", task.num_fewshot,
            "--batch_size", task.batch_size,
            "--log_samples",
            "--show_config",
        ]
    # fmt: on

    if task.include_path:
        cmd.append("--include_path")
        cmd.append(task_venv_config.venv_path / task.include_path)
        os.chdir(task_venv_config.venv_path)
    if task.apply_chat_template:
        cmd.append("--apply_chat_template")  # Flag argument (no value)

    # Add metadata parameter if specified (needed for tasks like RULER)
    if getattr(task, "custom_dataset_kwargs", None):
        cmd.append("--metadata")
        cmd.append(json.dumps(task.custom_dataset_kwargs))

    # Add safety flags for code evaluation tasks
    if task.workflow_venv_type == WorkflowVenvType.EVALS_COMMON:
        cmd.append("--trust_remote_code")
        cmd.append("--confirm_run_unsafe_code")

    samples_arg = _resolve_eval_samples(task, runtime_config)
    if samples_arg is not None:
        cmd.extend(["--samples", samples_arg])

    limit_arg = _resolve_eval_limit(task, runtime_config)
    if limit_arg is not None:
        cmd.extend(["--limit", str(limit_arg)])

    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    # runtime config loaded from JSON
    device_str = runtime_config.device
    disable_trace_capture = runtime_config.disable_trace_capture

    # Automatically control trace capture based on has_builtin_warmup
    # Only apply automatic logic if user hasn't explicitly set --disable-trace-capture
    if not disable_trace_capture and hasattr(model_spec, "has_builtin_warmup"):
        if model_spec.has_builtin_warmup:
            disable_trace_capture = True
            logger.info(
                "Model has builtin warmup (has_builtin_warmup=True), "
                "automatically disabling trace capture for evals workflow"
            )

    device = DeviceTypes.from_string(device_str)
    workflow_config = WORKFLOW_EVALS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    assert device == model_spec.device_type

    # Setup authentication based on model type
    if model_spec.model_type in EVAL_TASK_TYPES:
        _setup_openai_api_key(args, logger)
    else:
        # LLM (vLLM) auth. Mirror the server's resolution order in
        # vllm-tt-metal/src/run_vllm_api_server.py:435-449 so the eval client
        # picks the same bearer the server expects:
        #   VLLM_API_KEY (raw token) > JWT_SECRET (sign one) > existing OPENAI_API_KEY.
        # Fail loud if none are set: otherwise lm-eval sends unauthenticated
        # requests, vLLM returns 401, openai_completions.parse_generations()
        # catches KeyError('choices') and substitutes "" for every output, so
        # every task scores 0% with thousands of "Could not parse generations:
        # 'choices'" warnings (this is what bit the CI evals workflow).
        #
        # Diagnostic fingerprints: log non-reversible sha256[:8] prefixes of
        # the JWT_SECRET / bearer token so we can correlate with the server
        # log's matching fingerprint and detect cross-process divergence
        # (e.g. .env vs runner-cached env). NEVER logs raw secrets.
        import hashlib

        vllm_api_key = os.getenv("VLLM_API_KEY")
        if vllm_api_key:
            os.environ["OPENAI_API_KEY"] = vllm_api_key
            bearer_fp = hashlib.sha256(vllm_api_key.encode()).hexdigest()[:8]
            logger.info(
                f"OPENAI_API_KEY set from VLLM_API_KEY env var "
                f"(sha256[:8]={bearer_fp})."
            )
        elif args.jwt_secret:
            json_payload = json.loads(
                '{"team_id": "tenstorrent", "token_id": "debug-test"}'
            )
            encoded_jwt = jwt.encode(json_payload, args.jwt_secret, algorithm="HS256")
            os.environ["OPENAI_API_KEY"] = encoded_jwt
            secret_fp = hashlib.sha256(args.jwt_secret.encode()).hexdigest()[:8]
            bearer_fp = hashlib.sha256(encoded_jwt.encode()).hexdigest()[:8]
            logger.info(
                f"OPENAI_API_KEY set from JWT signed with JWT_SECRET "
                f"(JWT_SECRET sha256[:8]={secret_fp}, bearer "
                f"sha256[:8]={bearer_fp})."
            )
        elif os.getenv("OPENAI_API_KEY"):
            existing = os.getenv("OPENAI_API_KEY")
            bearer_fp = hashlib.sha256(existing.encode()).hexdigest()[:8]
            logger.info(
                f"OPENAI_API_KEY already set in env; using as-is "
                f"(sha256[:8]={bearer_fp}, set VLLM_API_KEY or JWT_SECRET to override)."
            )
        else:
            raise RuntimeError(
                "No bearer-token material available for the vLLM API server. "
                "The server requires JWT auth by default, so unauthenticated "
                "requests return 401 and lm-eval scores 0% on every task. "
                "Set one of: VLLM_API_KEY (raw bearer token), JWT_SECRET (to "
                "sign a token), --jwt-secret on the command line, or "
                "OPENAI_API_KEY directly. If the server was launched with "
                "--disable-vllm-api-key, set OPENAI_API_KEY to any non-empty "
                "placeholder so this guard is satisfied."
            )
    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

    # Look up the evaluation configuration for the model using EVAL_CONFIGS.
    if model_spec.model_name not in EVAL_CONFIGS:
        message = f"No evaluation tasks defined for model: {model_spec.model_name}"
        raise ValueError(message)
    eval_config = EVAL_CONFIGS[model_spec.model_name]
    eval_config = _select_eval_config(eval_config, runtime_config)

    # Set environment variable for code evaluation tasks
    # This must be set in os.environ because lm_eval modules check for it during import
    has_code_eval_tasks = any(
        task.workflow_venv_type == WorkflowVenvType.EVALS_COMMON
        for task in eval_config.tasks
    )
    if has_code_eval_tasks:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        logger.info("Set HF_ALLOW_CODE_EVAL=1 for code evaluation tasks")

    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = args.jwt_secret
    env_config.service_port = runtime_config.service_port
    env_config.vllm_model = model_spec.hf_model_repo

    if (
        model_spec.model_type in EVAL_TASK_TYPES
        and model_spec.model_type != ModelType.AUDIO
    ):
        return_code = run_media_evals(
            eval_config,
            model_spec,
            device,
            args.output_path,
            runtime_config.service_port,
        )
        return return_code

    # For AUDIO models, skip PromptClient and let lmms-eval handle server communication
    # Note: AudioClient is NOT used here
    # This runs accuracy evaluations (WER scores) via lmms-eval, not performance benchmarks.
    elif model_spec.model_type == ModelType.AUDIO:
        logger.info("Running audio evals with lmms-eval ...")

        # Check if media server is healthy before running evals
        _check_media_server_health(
            model_spec=model_spec,
            device=device,
            output_path=args.output_path,
            service_port=runtime_config.service_port,
        )

        return_codes = []
        for task in eval_config.tasks:
            logger.info(
                f"Starting workflow: {workflow_config.name} task_name: {task.task_name}"
            )
            logger.info(f"Running lm_eval for:\n {task}")
            cmd = build_eval_command(
                task,
                model_spec,
                device_str,
                args.output_path,
                runtime_config.service_port,
                runtime_config=runtime_config,
            )
            return_code = run_command(command=cmd, logger=logger, env=env_vars)
            return_codes.append(return_code)

        if all(return_code == 0 for return_code in return_codes):
            logger.info("✅ Completed evals")
            return 0
        logger.error(
            f"⛔ evals failed with return codes: {return_codes}. See logs above for details."
        )
        return 1

    # For LLM models, use PromptClient for health checks and trace capture
    else:
        prompt_client = PromptClient(
            env_config,
            model_spec=model_spec,
            runtime_config=runtime_config,
        )
        logger.info(
            "Using tensor_cache_timeout:=%ss for first-run tensor cache generation when cache monitoring is active",
            prompt_client.cache_monitor.get_tensor_cache_timeout(),
        )
        if not prompt_client.wait_for_healthy():
            logger.error("⛔️ vLLM server is not healthy. Aborting evaluations.")
            return 1

        # Preflight auth probe: /health is unauthenticated on vLLM, so a
        # healthy server does NOT prove the eval client and the server
        # agree on the bearer. We send one max_tokens=1 request and
        # assert the response has a usable 'choices' list. If it does
        # not (e.g. 401 / error JSON without 'choices'), aborting now
        # prevents the silent 0%-score failure mode where lm-eval
        # treats every "Could not parse generations: 'choices'" warning
        # as an empty completion.
        probe_base_url = f"{env_config.deploy_url}:{env_config.service_port}/v1"
        # Prefer the manifest's authoritative bearer (single source of
        # truth) and fall back to OPENAI_API_KEY in env. Same bearer
        # the in-container vLLM server validates against, by construction.
        probe_bearer_token, _bearer_src = load_bearer_or_fallback(
            getattr(runtime_config, "auth_manifest_path", None)
        )
        if probe_bearer_token is None:
            probe_bearer_token = os.environ.get("OPENAI_API_KEY")
        probe_model = getattr(env_config, "vllm_model", None) or getattr(
            model_spec, "hf_model_repo", None
        )
        probe_result = probe_bearer(
            probe_base_url,
            probe_bearer_token,
            model=probe_model,
        )
        assert_probe_ok(probe_result)

        if not disable_trace_capture:
            if "image" in model_spec.supported_modalities:
                prompt_client.capture_traces(image_resolutions=IMAGE_RESOLUTIONS)
            else:
                prompt_client.capture_traces()

        # Execute lm_eval for each task.
        logger.info("Running vLLM evals client ...")
        return_codes = []
        for task in eval_config.tasks:
            health_check = prompt_client.get_health()
            if health_check.status_code != 200:
                logger.error("⛔️ vLLM server is not healthy. Aborting evaluations.")
                return 1

            logger.info(
                f"Starting workflow: {workflow_config.name} task_name: {task.task_name}"
            )

            logger.info(f"Running lm_eval for:\n {task}")
            cmd = build_eval_command(
                task,
                model_spec,
                device_str,
                args.output_path,
                runtime_config.service_port,
                runtime_config=runtime_config,
            )
            return_code = run_command(command=cmd, logger=logger, env=env_vars)
            return_codes.append(return_code)

        if all(return_code == 0 for return_code in return_codes):
            logger.info("✅ Completed evals")
            return 0
        logger.error(
            f"⛔ evals failed with return codes: {return_codes}. See logs above for details."
        )
        return 1


def run_media_evals(all_params, model_spec, device, output_path, service_port):
    """
    Run media evals for cnn and image models only (not AUDIO models).

    AUDIO models use lmms-eval directly and do not call this function.
    This function uses ImageClient which can handle both cnn, image and audio transcription
    models via tt-media-server, but in the evals workflow it's only called for cnn and image models.
    """
    logger.info(
        f"Running media (image and cnn) benchmarks for model: {model_spec.model_name} on device: {device.name}"
    )
    return MediaClientFactory.run_media_task(
        model_spec,
        all_params,
        device,
        output_path,
        service_port,
        task_type=MediaTaskType.EVALUATION,
    )


def run_audio_evals(all_params, model_spec, device, output_path, service_port):
    """
    Run audio benchmarks for the given model and device.
    """
    logger.info(
        f"Running audio evals for model: {model_spec.model_name} on device: {device.name}"
    )
    return MediaClientFactory.run_media_task(
        model_spec,
        all_params,
        device,
        output_path,
        service_port,
        task_type=MediaTaskType.EVALUATION,
    )


if __name__ == "__main__":
    sys.exit(main())
