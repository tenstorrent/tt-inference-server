#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
GuideLLM benchmark runner for tt-inference-server.

This script integrates GuideLLM into the benchmarks workflow and exposes
three scenario presets:
- multi_turn_chat
- custom_dataset
- omni_modal
"""

import argparse
import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jwt

# Add project root to Python path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import run_command
from workflows.workflow_types import InferenceEngine, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS = ["multi_turn_chat", "custom_dataset", "omni_modal"]
DEFAULT_OMNI_MODALITIES = ["text", "image", "video", "audio"]


@dataclass
class GuideRunConfig:
    name: str
    data: str
    profile: str
    output_dir: Path
    max_requests: Optional[int] = None
    max_seconds: Optional[int] = None
    request_type: Optional[str] = None
    data_args: Optional[str] = None
    data_column_mapper: Optional[str] = None
    data_preprocessors: Optional[str] = None
    backend_type: str = "openai_http"
    backend_kwargs: Optional[str] = None
    extra_args: Optional[List[str]] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run GuideLLM benchmarks")
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for benchmark output",
        required=True,
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument("--device", type=str, required=False)
    parser.add_argument("--model", type=str, required=False)
    return parser.parse_args()


def parse_workflow_args(workflow_args: Optional[str]) -> Dict[str, str]:
    if not workflow_args:
        return {}
    parsed: Dict[str, str] = {}
    for token in shlex.split(workflow_args):
        if "=" in token:
            key, value = token.split("=", 1)
            parsed[key.replace("-", "_")] = value
        else:
            parsed[token.replace("-", "_")] = "true"
    return parsed


def _parse_int(value: Optional[str], default_value: Optional[int]) -> Optional[int]:
    if value is None:
        return default_value
    return int(value)


def build_auth_token(jwt_secret: str) -> str:
    if not jwt_secret:
        return ""
    payload = {"team_id": "tenstorrent", "token_id": "guidellm-bench"}
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def build_backend_kwargs_json(
    auth_token: str, override_backend_kwargs: Optional[str]
) -> str:
    if override_backend_kwargs:
        return override_backend_kwargs
    kwargs = {
        "validate_backend": "/v1/models",
        "http2": False,
        "stream": False,
        "timeout": 300,
        "max_tokens": 64,
    }
    if auth_token:
        kwargs["api_key"] = auth_token
    return json.dumps(kwargs)


def create_default_multiturn_dataset(dataset_path: Path, samples: int = 50) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as dataset_file:
        for index in range(samples):
            row = {
                "system_context": (
                    "You are a helpful assistant. Keep responses concise and accurate."
                ),
                "turn_1": f"What is {index} * 3?",
                "turn_2": "Now add 10 to that result.",
            }
            dataset_file.write(json.dumps(row) + "\n")
    return dataset_path


def wait_for_server(model_spec: ModelSpec, runtime_config: RuntimeConfig, jwt_secret: str):
    env_config = EnvironmentConfig()
    env_config.jwt_secret = jwt_secret
    env_config.vllm_api_key = os.getenv("VLLM_API_KEY")
    env_config.service_port = runtime_config.service_port
    env_config.vllm_model = model_spec.hf_model_repo

    prompt_client = PromptClient(
        env_config,
        model_spec=model_spec,
        runtime_config=runtime_config,
    )
    return prompt_client.wait_for_healthy()


def build_guidellm_command(
    venv_python: Path,
    target: str,
    model_name: str,
    run_cfg: GuideRunConfig,
) -> List[str]:
    command = [
        str(venv_python),
        "-m",
        "guidellm",
        "benchmark",
        "--disable-console-interactive",
        "--target",
        target,
        "--model",
        model_name,
        "--profile",
        run_cfg.profile,
        "--backend-type",
        run_cfg.backend_type,
        "--output-dir",
        str(run_cfg.output_dir),
        "--data",
        run_cfg.data,
    ]
    if run_cfg.request_type:
        command.extend(["--request-type", run_cfg.request_type])
    if run_cfg.max_requests is not None:
        command.extend(["--max-requests", str(run_cfg.max_requests)])
    if run_cfg.max_seconds is not None:
        command.extend(["--max-seconds", str(run_cfg.max_seconds)])
    if run_cfg.data_args:
        command.extend(["--data-args", run_cfg.data_args])
    if run_cfg.data_column_mapper:
        command.extend(["--data-column-mapper", run_cfg.data_column_mapper])
    if run_cfg.data_preprocessors:
        command.extend(["--data-preprocessors", run_cfg.data_preprocessors])
    if run_cfg.backend_kwargs:
        command.extend(["--backend-kwargs", run_cfg.backend_kwargs])
    if run_cfg.extra_args:
        command.extend(run_cfg.extra_args)
    return command


def get_supported_modalities(model_spec: ModelSpec) -> List[str]:
    modalities = getattr(model_spec, "supported_modalities", None)
    if not modalities:
        return ["text"]
    return [str(modality).lower() for modality in modalities]


def build_scenario_runs(
    output_root: Path,
    workflow_params: Dict[str, str],
    auth_token: str,
    supported_modalities: List[str],
) -> List[GuideRunConfig]:
    selected_scenarios = workflow_params.get("scenarios") or workflow_params.get("scenario")
    if selected_scenarios:
        scenarios = [value.strip() for value in selected_scenarios.split(",") if value]
    else:
        scenarios = list(DEFAULT_SCENARIOS)

    backend_kwargs = build_backend_kwargs_json(
        auth_token, workflow_params.get("backend_kwargs")
    )
    runs: List[GuideRunConfig] = []

    if "multi_turn_chat" in scenarios:
        dataset = workflow_params.get("multiturn_data")
        if dataset:
            multiturn_dataset_path = Path(dataset)
        else:
            multiturn_dataset_path = output_root / "guidellm_datasets" / "multiturn.jsonl"
            create_default_multiturn_dataset(multiturn_dataset_path)
        runs.append(
            GuideRunConfig(
                name="multi_turn_chat",
                data=str(multiturn_dataset_path),
                profile=workflow_params.get("multiturn_profile", "synchronous"),
                output_dir=output_root / "guidellm_multi_turn_chat",
                max_requests=_parse_int(workflow_params.get("multiturn_max_requests"), 10),
                max_seconds=_parse_int(workflow_params.get("multiturn_max_seconds"), 120),
                request_type=workflow_params.get(
                    "multiturn_request_type", "chat_completions"
                ),
                data_column_mapper=workflow_params.get(
                    "multiturn_data_column_mapper",
                    '{"prefix_column":"system_context","text_column":"turn_1"}',
                ),
                backend_kwargs=backend_kwargs,
            )
        )

    if "custom_dataset" in scenarios:
        runs.append(
            GuideRunConfig(
                name="custom_dataset",
                data=workflow_params.get("custom_data", "mbpp"),
                profile=workflow_params.get("custom_profile", "sweep"),
                output_dir=output_root / "guidellm_custom_dataset",
                max_requests=_parse_int(workflow_params.get("custom_max_requests"), None),
                max_seconds=_parse_int(workflow_params.get("custom_max_seconds"), 60),
                data_args=workflow_params.get("custom_data_args"),
                data_column_mapper=workflow_params.get(
                    "custom_data_column_mapper", '{"text_column":"text"}'
                ),
                backend_kwargs=backend_kwargs,
            )
        )

    if "omni_modal" in scenarios:
        modalities_value = workflow_params.get("omni_modalities")
        if modalities_value:
            modalities = [value.strip().lower() for value in modalities_value.split(",")]
        else:
            modalities = list(DEFAULT_OMNI_MODALITIES)

        for modality in modalities:
            if modality != "text" and modality not in supported_modalities:
                logger.info(
                    f"Skipping omni-modal {modality} scenario: model does not advertise {modality} modality."
                )
                continue

            if modality == "text":
                runs.append(
                    GuideRunConfig(
                        name="omni_modal_text",
                        data=workflow_params.get("omni_text_data", "mbpp"),
                        profile=workflow_params.get("omni_text_profile", "synchronous"),
                        output_dir=output_root / "guidellm_omni_modal_text",
                        max_requests=_parse_int(
                            workflow_params.get("omni_text_max_requests"), 20
                        ),
                        request_type=workflow_params.get(
                            "omni_text_request_type", "chat_completions"
                        ),
                        data_column_mapper=workflow_params.get(
                            "omni_text_data_column_mapper", '{"text_column":"text"}'
                        ),
                        backend_kwargs=backend_kwargs,
                    )
                )
            elif modality == "image":
                runs.append(
                    GuideRunConfig(
                        name="omni_modal_image",
                        data=workflow_params.get("omni_image_data", "lmms-lab/MMBench_EN"),
                        profile=workflow_params.get("omni_image_profile", "synchronous"),
                        output_dir=output_root / "guidellm_omni_modal_image",
                        max_requests=_parse_int(
                            workflow_params.get("omni_image_max_requests"), 20
                        ),
                        request_type=workflow_params.get(
                            "omni_image_request_type", "chat_completions"
                        ),
                        data_args=workflow_params.get("omni_image_data_args", '{"split":"test"}'),
                        data_column_mapper=workflow_params.get(
                            "omni_image_data_column_mapper",
                            '{"image_column":"image","text_column":"question"}',
                        ),
                        data_preprocessors=workflow_params.get(
                            "omni_image_data_preprocessors", "encode_media"
                        ),
                        backend_kwargs=backend_kwargs,
                    )
                )
            elif modality == "video":
                runs.append(
                    GuideRunConfig(
                        name="omni_modal_video",
                        data=workflow_params.get("omni_video_data", "lmms-lab/Video-MME"),
                        profile=workflow_params.get("omni_video_profile", "synchronous"),
                        output_dir=output_root / "guidellm_omni_modal_video",
                        max_requests=_parse_int(
                            workflow_params.get("omni_video_max_requests"), 20
                        ),
                        request_type=workflow_params.get(
                            "omni_video_request_type", "chat_completions"
                        ),
                        data_args=workflow_params.get("omni_video_data_args", '{"split":"test"}'),
                        data_column_mapper=workflow_params.get(
                            "omni_video_data_column_mapper",
                            '{"video_column":"url","text_column":"question"}',
                        ),
                        data_preprocessors=workflow_params.get(
                            "omni_video_data_preprocessors", "encode_media"
                        ),
                        backend_kwargs=backend_kwargs,
                    )
                )
            elif modality == "audio":
                runs.append(
                    GuideRunConfig(
                        name="omni_modal_audio",
                        data=workflow_params.get(
                            "omni_audio_data", "hf-internal-testing/librispeech_asr_dummy"
                        ),
                        profile=workflow_params.get("omni_audio_profile", "synchronous"),
                        output_dir=output_root / "guidellm_omni_modal_audio",
                        max_requests=_parse_int(
                            workflow_params.get("omni_audio_max_requests"), 20
                        ),
                        request_type=workflow_params.get(
                            "omni_audio_request_type", "chat_completions"
                        ),
                        data_column_mapper=workflow_params.get(
                            "omni_audio_data_column_mapper",
                            '{"audio_column":"audio","text_column":"text"}',
                        ),
                        data_preprocessors=workflow_params.get(
                            "omni_audio_data_preprocessors", "encode_media"
                        ),
                        backend_kwargs=backend_kwargs,
                    )
                )

    extra_args = workflow_params.get("guidellm_extra_args")
    if extra_args:
        parsed_extra = shlex.split(extra_args)
        for run_cfg in runs:
            run_cfg.extra_args = parsed_extra

    return runs


def main():
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    if model_spec.inference_engine in (
        InferenceEngine.MEDIA.value,
        InferenceEngine.FORGE.value,
    ):
        os.environ["VLLM_API_KEY"] = "your-secret-key"

    if not wait_for_server(model_spec, runtime_config, args.jwt_secret):
        logger.error("vLLM server is not healthy. Aborting GuideLLM benchmarks.")
        return 1

    venv_config = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_GUIDELLM]
    auth_token = build_auth_token(args.jwt_secret)
    if auth_token:
        os.environ["OPENAI_API_KEY"] = auth_token

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    workflow_params = parse_workflow_args(runtime_config.workflow_args)
    target = workflow_params.get(
        "target", f"http://127.0.0.1:{runtime_config.service_port}/v1"
    )

    runs = build_scenario_runs(
        output_root=output_root,
        workflow_params=workflow_params,
        auth_token=auth_token,
        supported_modalities=get_supported_modalities(model_spec),
    )
    if not runs:
        logger.error("No GuideLLM runs were configured. Nothing to execute.")
        return 1

    logger.info(f"Running {len(runs)} GuideLLM scenario(s): {[r.name for r in runs]}")

    return_codes: List[int] = []
    for run_cfg in runs:
        run_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_guidellm_command(
            venv_python=venv_config.venv_python,
            target=target,
            model_name=model_spec.hf_model_repo,
            run_cfg=run_cfg,
        )
        logger.info(f"Running GuideLLM scenario: {run_cfg.name}")
        logger.info(f"GuideLLM command: {' '.join(cmd)}")
        return_code = run_command(command=cmd, logger=logger, env=os.environ.copy())
        return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("Completed GuideLLM benchmarks.")
        return 0

    logger.error(f"GuideLLM benchmark failures: return codes={return_codes}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
