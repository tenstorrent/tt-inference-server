# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.media_clients.media_client_factory import MediaClientFactory, MediaTaskType
from utils.url_helpers import resolve_deploy_url

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from evals.eval_config import EVAL_CONFIGS, EvalConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_config import (
    WORKFLOW_EVALS_CONFIG,
)
from workflows.workflow_types import (
    DeviceTypes,
    EvalLimitMode,
    InferenceEngine,
    ModelType,
)

logger = logging.getLogger(__name__)
EVAL_TASK_TYPES = [
    ModelType.IMAGE,
    ModelType.CNN,
    ModelType.AUDIO,
    ModelType.TEXT_TO_SPEECH,
]


def _get_limit_mode(runtime_config: Optional[RuntimeConfig]) -> Optional[EvalLimitMode]:
    if runtime_config is None or not runtime_config.limit_samples_mode:
        return None
    return EvalLimitMode.from_string(runtime_config.limit_samples_mode)


def _select_eval_config(
    eval_config: EvalConfig, runtime_config: Optional[RuntimeConfig]
) -> EvalConfig:
    eval_samples = getattr(runtime_config, "eval_samples", None)
    if eval_samples and eval_config.tasks:
        mapping = _parse_eval_samples_mapping(eval_samples)
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


def _parse_eval_samples_mapping(value: Optional[str]) -> Optional[dict]:
    """Parse the --eval-samples value into a dict.

    Accepts either a JSON string of shape ``{"task_name": [int, ...], ...}``
    or a path to a JSON file containing the same shape.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except TypeError as exc:
        raise ValueError(
            "--eval-samples must be a JSON string or a path to a JSON file; "
            f"got {type(value).__name__}"
        ) from exc
    except json.JSONDecodeError:
        path = Path(value)
        if not path.is_file():
            raise ValueError(
                f"--eval-samples value is not valid JSON and not an existing file: {value}"
            )
        try:
            parsed = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--eval-samples file does not contain valid JSON: {path}"
            ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "--eval-samples must decode to a JSON object mapping task_name -> "
            f"[int, ...]; got {type(parsed).__name__}"
        )
    return parsed


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


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    # runtime config loaded from JSON
    device_str = runtime_config.device

    device = DeviceTypes.from_string(device_str)
    workflow_config = WORKFLOW_EVALS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    assert device == model_spec.device_type

    # Setup authentication based on model type
    if model_spec.model_type in EVAL_TASK_TYPES:
        _setup_openai_api_key(args, logger)
    elif model_spec.inference_engine in (
        InferenceEngine.MEDIA.value,
        InferenceEngine.FORGE.value,
    ):
        # Forge/media servers validate the literal API key, not JWTs.
        _setup_openai_api_key(args, logger)
        os.environ["VLLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
    # Look up the evaluation configuration for the model using EVAL_CONFIGS.
    if model_spec.model_name not in EVAL_CONFIGS:
        message = f"No evaluation tasks defined for model: {model_spec.model_name}"
        raise ValueError(message)
    eval_config = EVAL_CONFIGS[model_spec.model_name]
    eval_config = _select_eval_config(eval_config, runtime_config)

    deploy_url = resolve_deploy_url(runtime_config)

    if (
        model_spec.model_type in EVAL_TASK_TYPES
        and model_spec.model_type != ModelType.AUDIO
    ):
        return run_media_evals(
            eval_config,
            model_spec,
            device,
            args.output_path,
            runtime_config.service_port,
            deploy_url=deploy_url,
        )

    else:
        raise SystemExit(
            f"evals for {model_spec.model_name!r} ({model_spec.model_type.name}) "
            "were not routed by can_route_to_v2(); check workflows/v2_bridge.py."
        )


def run_media_evals(
    all_params, model_spec, device, output_path, service_port, deploy_url=None
):
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
        deploy_url=deploy_url,
    )


if __name__ == "__main__":
    sys.exit(main())
