# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from workflows.model_spec import ModelSpec


def _short_uuid() -> str:
    """Return 8-character random UUID."""
    u = uuid.uuid4()
    return base64.urlsafe_b64encode(u.bytes)[:8].decode("utf-8")


@dataclass
class RuntimeConfig:
    """Runtime configuration capturing CLI arguments and orchestration state.

    Separated from ModelSpec to keep the static model definition clean.
    This dataclass replaces the former model_spec.cli_args pattern.
    """

    # Required — always set from CLI
    model: str
    workflow: str
    device: str

    # Model selection
    impl: Optional[str] = None
    engine: Optional[str] = None

    # Server mode
    docker_server: bool = False
    local_server: bool = False
    interactive: bool = False
    service_port: str = "8000"
    bind_host: str = "0.0.0.0"
    server_url: Optional[str] = None

    # Dev / override
    dev_mode: bool = False
    no_auth: bool = False
    print_docker_cmd: bool = False
    override_docker_image: Optional[str] = None
    override_tt_config: Optional[str] = None
    vllm_override_args: Optional[str] = None
    runtime_model_spec_json: Optional[str] = None

    # Workflow control
    tools: str = "vllm"
    disable_trace_capture: bool = False
    disable_metal_timeout: bool = False
    concurrency_sweeps: bool = False
    percentile_report: bool = False
    streaming: Optional[str] = None
    preprocessing: Optional[str] = None
    workflow_args: Optional[str] = None
    limit_samples_mode: Optional[str] = None
    sdxl_num_prompts: str = "100"

    # Device configuration
    device_id: Optional[List[int]] = None

    # Docker volume options
    host_volume: Optional[str] = None
    host_hf_cache: Optional[str] = None
    host_weights_dir: Optional[str] = None
    image_user: str = "1000"

    # Validation
    skip_system_sw_validation: bool = False

    # Misc
    tt_metal_python_venv_dir: Optional[str] = None
    tt_metal_home: Optional[str] = None
    vllm_dir: Optional[str] = None

    # Runtime state (set during execution, not from CLI)
    run_id: Optional[str] = None
    runtime_model_spec: Optional[Dict] = field(default=None, repr=False)

    @classmethod
    def from_args(
        cls,
        args,
        *,
        impl: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> RuntimeConfig:
        """Create a RuntimeConfig from an argparse Namespace.

        Optional *impl* and *engine* keyword arguments override the values
        from *args* when the model-spec resolution has already determined
        the correct values.
        """
        return cls(
            model=args.model,
            workflow=args.workflow,
            device=args.device,
            impl=impl if impl is not None else args.impl,
            engine=engine if engine is not None else args.engine,
            docker_server=args.docker_server,
            local_server=args.local_server,
            interactive=args.interactive,
            service_port=args.service_port,
            bind_host=args.bind_host,
            server_url=args.server_url,
            dev_mode=args.dev_mode,
            no_auth=args.no_auth,
            print_docker_cmd=args.print_docker_cmd,
            override_docker_image=args.override_docker_image,
            override_tt_config=args.override_tt_config,
            vllm_override_args=args.vllm_override_args,
            runtime_model_spec_json=args.runtime_model_spec_json,
            tools=args.tools,
            disable_trace_capture=args.disable_trace_capture,
            disable_metal_timeout=args.disable_metal_timeout,
            concurrency_sweeps=args.concurrency_sweeps,
            percentile_report=args.percentile_report,
            streaming=args.streaming,
            preprocessing=args.preprocessing,
            workflow_args=args.workflow_args,
            limit_samples_mode=args.limit_samples_mode,
            sdxl_num_prompts=args.sdxl_num_prompts,
            device_id=args.device_id,
            host_volume=args.host_volume,
            host_hf_cache=args.host_hf_cache,
            host_weights_dir=args.host_weights_dir,
            image_user=args.image_user,
            skip_system_sw_validation=args.skip_system_sw_validation,
            tt_metal_python_venv_dir=args.tt_metal_python_venv_dir,
            tt_metal_home=args.tt_metal_home,
            vllm_dir=args.vllm_dir,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("runtime_model_spec", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RuntimeConfig:
        """Create from a dict, ignoring unknown keys for forward compatibility."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # JSON serialisation (runtime_model_spec + runtime_config document)
    # ------------------------------------------------------------------

    def to_json(
        self, model_spec: ModelSpec, timestamp: str, model_id: str, output_dir: str
    ) -> Path:
        """Write ``{"runtime_model_spec": …, "runtime_config": …}`` to *output_dir*.

        Filename: ``runtime_model_spec_{timestamp}_{model_id}_{short-UUID}.json``

        Returns the path of the written file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        combined = {
            "runtime_model_spec": model_spec.get_serialized_dict(),
            "runtime_config": self.to_dict(),
        }

        filename = f"runtime_model_spec_{timestamp}_{model_id}_{_short_uuid()}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        return filepath

    @classmethod
    def from_json(cls, json_fpath: str) -> RuntimeConfig:
        """Load a RuntimeConfig from a JSON file.

        Supports three formats:
        - Current: ``{"runtime_model_spec": …, "runtime_config": …}``
        - Legacy v2: ``{"model_spec": …, "runtime_config": …}``
        - Legacy v1: flat ModelSpec dict with embedded ``cli_args`` dict
        """
        json_path = Path(json_fpath)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_fpath}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "runtime_config" in data:
            return cls.from_dict(data["runtime_config"])

        # Backward compat: old format had cli_args embedded in model spec
        if "cli_args" in data:
            return cls.from_dict(data["cli_args"])

        raise ValueError(
            f"No runtime_config or cli_args section found in: {json_fpath}"
        )
