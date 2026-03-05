# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Stress Tests Configuration Arguments

Consolidates arguments from multiple sources (argparse, RuntimeConfig, workflow_args)
into a single well-typed dataclass with explicit precedence rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from workflows.runtime_config import RuntimeConfig


@dataclass
class StressTestsArgs:
    """
    Consolidated configuration for stress tests execution.

    Arguments come from three sources:
    1. CLI args (argparse): project_root, output_path, jwt_secret
    2. RuntimeConfig: model, device, service_port, disable_trace_capture, etc.
    3. Parsed workflow_args: run_mode, max_context_length, endurance_mode, custom_*_values, custom_num_prompts_strategy
    """

    # From argparse
    project_root: Path
    output_path: str
    jwt_secret: str

    # From RuntimeConfig (core configuration)
    model: str
    device: str
    service_port: str = "8000"

    # From RuntimeConfig (stress tests specific)
    disable_trace_capture: bool = False
    run_mode: str = "multiple"
    max_context_length: Optional[int] = None
    endurance_mode: bool = False

    # From parsed workflow_args (single mode parameters)
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    max_concurrent: Optional[int] = None
    num_prompts: Optional[int] = None

    # From parsed workflow_args (custom parameter values for multiple mode)
    custom_isl_values: Optional[str] = None
    custom_osl_values: Optional[str] = None
    custom_concurrency_values: Optional[str] = None
    custom_num_prompts_strategy: Optional[str] = None
    only_match_concurrency: bool = False
    use_server_tokenizer: bool = False

    # Runtime additions
    model_spec: Optional[object] = None

    @classmethod
    def from_sources(
        cls, args, runtime_config: RuntimeConfig, model_spec, parsed_workflow_args: dict
    ):
        """
        Merge arguments from multiple sources with clear precedence.

        Args:
            args: argparse Namespace with project_root, output_path, jwt_secret
            runtime_config: RuntimeConfig with model, device, service_port, etc.
            model_spec: ModelSpec object
            parsed_workflow_args: dict of parsed workflow_args

        Returns:
            StressTestsArgs instance with all fields populated
        """
        return cls(
            # From argparse
            project_root=args.project_root,
            output_path=args.output_path,
            jwt_secret=args.jwt_secret,
            # From RuntimeConfig (required)
            model=runtime_config.model,
            device=runtime_config.device,
            service_port=runtime_config.service_port,
            # From RuntimeConfig (stress tests configuration)
            # Auto-disable trace capture if model has builtin warmup and user hasn't explicitly overridden
            disable_trace_capture=runtime_config.disable_trace_capture
            or getattr(model_spec, "has_builtin_warmup", False),
            run_mode=parsed_workflow_args.get("run_mode", "multiple"),
            max_context_length=parsed_workflow_args.get("max_context_length"),
            endurance_mode=parsed_workflow_args.get("endurance_mode", False),
            # From parsed workflow_args (single mode parameters)
            input_size=parsed_workflow_args.get("input_size"),
            output_size=parsed_workflow_args.get("output_size"),
            max_concurrent=parsed_workflow_args.get("max_concurrent"),
            num_prompts=parsed_workflow_args.get("num_prompts"),
            # From parsed workflow_args (multiple mode parameters)
            custom_isl_values=parsed_workflow_args.get("custom_isl_values"),
            custom_osl_values=parsed_workflow_args.get("custom_osl_values"),
            custom_concurrency_values=parsed_workflow_args.get(
                "custom_concurrency_values"
            ),
            custom_num_prompts_strategy=parsed_workflow_args.get(
                "custom_num_prompts_strategy"
            ),
            only_match_concurrency=parsed_workflow_args.get(
                "only_match_concurrency", False
            ),
            use_server_tokenizer=parsed_workflow_args.get(
                "use_server_tokenizer", False
            ),
            # Runtime
            model_spec=model_spec,
        )

    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.model:
            raise ValueError("model is required")
        if not self.device:
            raise ValueError("device is required")
