# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""GuideLLM scenario presets.

Each scenario is one dataset-driven GuideLLM run (a "sweep point"):
``multi_turn_chat`` (generated JSONL), ``custom_dataset`` (mbpp), and
``omni_modal`` (text/image/video/audio over real HF datasets, gated by
the model's advertised modalities). Selection and per-scenario knobs ride
on ``--workflow-args`` exactly as in v1.

A scenario subclasses :class:`LLMRunConfig` so it flows through the
existing ``LLMPerformanceRunner`` unchanged; ``isl``/``osl`` are
data-determined (left 0) and the driver branches on the scenario fields.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import LLMRunConfig

logger = logging.getLogger(__name__)

DEFAULT_SCENARIOS = ["multi_turn_chat", "custom_dataset", "omni_modal"]
DEFAULT_OMNI_MODALITIES = ["text", "image", "video", "audio"]


@dataclass(frozen=True)
class GuideLLMScenario(LLMRunConfig):
    """One GuideLLM scenario run.

    Inherits the sweep-point fields (``isl``/``osl``/``max_concurrency``/
    ``num_prompts``) so it is a structural ``LLMRunConfig`` for the runner;
    they are placeholders here (the workload shape comes from ``data``).
    """

    name: str = ""
    data: str = ""
    profile: str = "synchronous"
    request_type: Optional[str] = None
    data_args: Optional[str] = None
    data_column_mapper: Optional[str] = None
    data_preprocessors: Optional[str] = None
    backend_type: str = "openai_http"
    backend_kwargs: Optional[str] = None
    max_requests: Optional[int] = None
    max_seconds: Optional[int] = None
    extra_args: List[str] = field(default_factory=list)


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


def get_supported_modalities(model_spec) -> List[str]:
    modalities = getattr(model_spec, "supported_modalities", None)
    if not modalities:
        return ["text"]
    return [str(modality).lower() for modality in modalities]


def build_guidellm_scenarios(
    model_spec,
    runtime_config,
    *,
    output_root: Path,
    auth_token: str = "",
) -> List[GuideLLMScenario]:
    """Build the GuideLLM scenario list for ``model_spec``.

    ``output_root`` holds the generated multi-turn dataset. Scenario
    selection and per-scenario knobs come from ``runtime_config.workflow_args``
    (v1 ``--workflow-args`` semantics).
    """
    workflow_params = parse_workflow_args(
        getattr(runtime_config, "workflow_args", None)
    )
    supported_modalities = get_supported_modalities(model_spec)

    selected_scenarios = workflow_params.get("scenarios") or workflow_params.get(
        "scenario"
    )
    if selected_scenarios:
        scenarios = [value.strip() for value in selected_scenarios.split(",") if value]
    else:
        scenarios = list(DEFAULT_SCENARIOS)

    backend_kwargs = build_backend_kwargs_json(
        auth_token, workflow_params.get("backend_kwargs")
    )
    runs: List[GuideLLMScenario] = []

    if "multi_turn_chat" in scenarios:
        dataset = workflow_params.get("multiturn_data")
        if dataset:
            multiturn_dataset_path = Path(dataset)
        else:
            multiturn_dataset_path = (
                output_root / "guidellm_datasets" / "multiturn.jsonl"
            )
            create_default_multiturn_dataset(multiturn_dataset_path)
        runs.append(
            GuideLLMScenario(
                isl=0,
                osl=0,
                max_concurrency=1,
                num_prompts=_parse_int(
                    workflow_params.get("multiturn_max_requests"), 10
                )
                or 0,
                name="multi_turn_chat",
                data=str(multiturn_dataset_path),
                profile=workflow_params.get("multiturn_profile", "synchronous"),
                max_requests=_parse_int(
                    workflow_params.get("multiturn_max_requests"), 10
                ),
                max_seconds=_parse_int(
                    workflow_params.get("multiturn_max_seconds"), 120
                ),
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
            GuideLLMScenario(
                isl=0,
                osl=0,
                max_concurrency=1,
                num_prompts=_parse_int(workflow_params.get("custom_max_requests"), 0)
                or 0,
                name="custom_dataset",
                data=workflow_params.get("custom_data", "mbpp"),
                profile=workflow_params.get("custom_profile", "sweep"),
                max_requests=_parse_int(
                    workflow_params.get("custom_max_requests"), None
                ),
                max_seconds=_parse_int(workflow_params.get("custom_max_seconds"), 60),
                data_args=workflow_params.get("custom_data_args"),
                data_column_mapper=workflow_params.get(
                    "custom_data_column_mapper", '{"text_column":"text"}'
                ),
                backend_kwargs=backend_kwargs,
            )
        )

    if "omni_modal" in scenarios:
        runs.extend(
            _build_omni_modal_scenarios(
                workflow_params, supported_modalities, backend_kwargs
            )
        )

    extra_args = workflow_params.get("guidellm_extra_args")
    if extra_args:
        parsed_extra = shlex.split(extra_args)
        runs = [_replace_extra_args(run_cfg, parsed_extra) for run_cfg in runs]

    return runs


def _replace_extra_args(
    scenario: GuideLLMScenario, extra_args: List[str]
) -> GuideLLMScenario:
    from dataclasses import replace

    return replace(scenario, extra_args=extra_args)


def _build_omni_modal_scenarios(
    workflow_params: Dict[str, str],
    supported_modalities: List[str],
    backend_kwargs: str,
) -> List[GuideLLMScenario]:
    modalities_value = workflow_params.get("omni_modalities")
    if modalities_value:
        modalities = [value.strip().lower() for value in modalities_value.split(",")]
    else:
        modalities = list(DEFAULT_OMNI_MODALITIES)

    runs: List[GuideLLMScenario] = []
    for modality in modalities:
        if modality != "text" and modality not in supported_modalities:
            logger.info(
                "Skipping omni-modal %s scenario: model does not advertise %s modality.",
                modality,
                modality,
            )
            continue

        if modality == "text":
            runs.append(
                GuideLLMScenario(
                    isl=0,
                    osl=0,
                    max_concurrency=1,
                    num_prompts=_parse_int(
                        workflow_params.get("omni_text_max_requests"), 20
                    )
                    or 0,
                    name="omni_modal_text",
                    data=workflow_params.get("omni_text_data", "mbpp"),
                    profile=workflow_params.get("omni_text_profile", "synchronous"),
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
                GuideLLMScenario(
                    isl=0,
                    osl=0,
                    max_concurrency=1,
                    num_prompts=_parse_int(
                        workflow_params.get("omni_image_max_requests"), 20
                    )
                    or 0,
                    name="omni_modal_image",
                    data=workflow_params.get("omni_image_data", "lmms-lab/MMBench_EN"),
                    profile=workflow_params.get("omni_image_profile", "synchronous"),
                    max_requests=_parse_int(
                        workflow_params.get("omni_image_max_requests"), 20
                    ),
                    request_type=workflow_params.get(
                        "omni_image_request_type", "chat_completions"
                    ),
                    data_args=workflow_params.get(
                        "omni_image_data_args", '{"split":"test"}'
                    ),
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
                GuideLLMScenario(
                    isl=0,
                    osl=0,
                    max_concurrency=1,
                    num_prompts=_parse_int(
                        workflow_params.get("omni_video_max_requests"), 20
                    )
                    or 0,
                    name="omni_modal_video",
                    data=workflow_params.get("omni_video_data", "lmms-lab/Video-MME"),
                    profile=workflow_params.get("omni_video_profile", "synchronous"),
                    max_requests=_parse_int(
                        workflow_params.get("omni_video_max_requests"), 20
                    ),
                    request_type=workflow_params.get(
                        "omni_video_request_type", "chat_completions"
                    ),
                    data_args=workflow_params.get(
                        "omni_video_data_args", '{"split":"test"}'
                    ),
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
                GuideLLMScenario(
                    isl=0,
                    osl=0,
                    max_concurrency=1,
                    num_prompts=_parse_int(
                        workflow_params.get("omni_audio_max_requests"), 20
                    )
                    or 0,
                    name="omni_modal_audio",
                    data=workflow_params.get(
                        "omni_audio_data",
                        "hf-internal-testing/librispeech_asr_dummy",
                    ),
                    profile=workflow_params.get("omni_audio_profile", "synchronous"),
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

    return runs


__all__ = [
    "GuideLLMScenario",
    "build_guidellm_scenarios",
    "DEFAULT_SCENARIOS",
    "DEFAULT_OMNI_MODALITIES",
]
