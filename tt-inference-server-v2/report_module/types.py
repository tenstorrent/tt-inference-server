# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import get_default_workflow_root_log_dir
from workflows.workflow_types import DeviceTypes

logger = logging.getLogger(__name__)

NOT_MEASURED_STR = "N/A"


@dataclass(frozen=True)
class ReportRequest:
    """Parsed from CLI or API. Fields have sensible defaults for optional omission."""

    output_path: str
    runtime_model_spec_json: str
    device: str = ""
    model: str = ""
    selected_reports: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportContext:
    """Immutable context built once from ReportRequest, shared with all strategies."""

    report_id: str
    model_spec: ModelSpec
    runtime_config: RuntimeConfig
    metadata: Dict[str, Any]
    output_path: Path
    workflow_log_dir: Path
    percentile_report: bool
    server_mode: str
    selected_sections: List[str] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model_spec.model_name

    @property
    def model_id(self) -> str:
        return self.model_spec.model_id

    @property
    def device_str(self) -> str:
        return self.metadata.get("device", "")


@dataclass(frozen=True)
class ReportResult:
    """Uniform return type from every strategy."""

    name: str
    markdown: str
    data: List[Dict[str, Any]]
    md_filename: Optional[str] = None
    display_markdown: Optional[str] = None

    @staticmethod
    def empty(name: str) -> ReportResult:
        return ReportResult(name=name, markdown="", data=[])


def build_context(request: ReportRequest) -> ReportContext:
    """Build a ReportContext from a ReportRequest.

    Loads ModelSpec and RuntimeConfig from the JSON path, resolves defaults
    for device/model, builds metadata, and returns an immutable context
    that all strategies share.
    """
    json_path = Path(request.runtime_model_spec_json)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Runtime model spec JSON not found: {request.runtime_model_spec_json}"
        )

    model_spec = ModelSpec.from_json(request.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(request.runtime_model_spec_json)

    model = request.model or runtime_config.model
    device_str = request.device or runtime_config.device
    docker_server = runtime_config.docker_server

    device = DeviceTypes.from_string(device_str)
    if device != model_spec.device_type:
        raise ValueError(
            f"Device mismatch: request device '{device_str}' does not match "
            f"model_spec device_type '{model_spec.device_type}'"
        )

    server_mode = "docker" if docker_server else "API"
    command_flag = "--docker-server" if docker_server else ""

    if model_spec.device_model_spec.default_impl:
        run_cmd = f"python run.py --model {model} --tt-device {device_str} --workflow release {command_flag}".strip()
    else:
        run_cmd = (
            f"python run.py --model {model} --tt-device {device_str} "
            f"--impl {model_spec.impl.impl_name} --workflow release {command_flag}"
        ).strip()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_id = f"{model_spec.model_id}_{run_timestamp}"

    metadata = {
        "report_id": report_id,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "runtime_model_spec_json": request.runtime_model_spec_json,
        "model_repo": model_spec.hf_model_repo,
        "model_impl": model_spec.impl.impl_name,
        "inference_engine": model_spec.inference_engine,
        "device": device_str,
        "server_mode": server_mode,
        "tt_metal_commit": model_spec.tt_metal_commit,
        "vllm_commit": model_spec.vllm_commit,
        "run_command": run_cmd,
    }

    logger.info(f"Built report context: report_id={report_id}, model={model}, device={device_str}")

    return ReportContext(
        report_id=report_id,
        model_spec=model_spec,
        runtime_config=runtime_config,
        metadata=metadata,
        output_path=Path(request.output_path),
        workflow_log_dir=get_default_workflow_root_log_dir(),
        percentile_report=runtime_config.percentile_report,
        server_mode=server_mode,
        selected_sections=list(request.selected_reports),
    )
