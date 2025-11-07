# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Spec Tests Configuration Arguments

Consolidates arguments from multiple sources (argparse, model_spec.cli_args, workflow_args)
into a single well-typed dataclass with explicit precedence rules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SpecTestsArgs:
    """
    Consolidated configuration for spec tests execution.
    
    Arguments come from three sources:
    1. CLI args (argparse): project_root, output_path, jwt_secret
    2. Model spec cli_args: model, device, service_port, disable_trace_capture, etc.
    3. Parsed workflow_args: custom_*_values, custom_num_prompts_strategy
    """
    
    # From argparse
    project_root: Path
    output_path: str
    jwt_secret: str
    
    # From model_spec.cli_args (core configuration)
    model: str
    device: str
    service_port: str = "8000"
    
    # From model_spec.cli_args (spec tests specific)
    disable_trace_capture: bool = False
    run_mode: str = "multiple"
    max_context_length: Optional[int] = None
    endurance_mode: bool = False
    
    # From parsed workflow_args (custom parameter values)
    custom_isl_values: Optional[str] = None
    custom_osl_values: Optional[str] = None
    custom_concurrency_values: Optional[str] = None
    custom_num_prompts_strategy: Optional[str] = None
    only_match_concurrency: bool = False
    use_server_tokenizer: bool = False
    
    # Runtime additions
    model_spec: Optional[object] = None
    
    @classmethod
    def from_sources(cls, args, cli_args: dict, model_spec, parsed_workflow_args: dict):
        """
        Merge arguments from multiple sources with clear precedence.
        
        Args:
            args: argparse Namespace with project_root, output_path, jwt_secret
            cli_args: dict from model_spec.cli_args
            model_spec: ModelSpec object
            parsed_workflow_args: dict of parsed workflow_args
            
        Returns:
            SpecTestsArgs instance with all fields populated
        """
        return cls(
            # From argparse
            project_root=args.project_root,
            output_path=args.output_path,
            jwt_secret=args.jwt_secret,
            
            # From model_spec.cli_args (required)
            model=cli_args.get("model"),
            device=cli_args.get("device"),
            service_port=cli_args.get("service_port", "8000"),
            
            # From model_spec.cli_args (spec tests configuration)
            disable_trace_capture=cli_args.get("disable_trace_capture", False),
            run_mode=cli_args.get("run_mode", "multiple"),
            max_context_length=cli_args.get("max_context_length"),
            endurance_mode=cli_args.get("endurance_mode", False),
            
            # From parsed workflow_args
            custom_isl_values=parsed_workflow_args.get("custom_isl_values"),
            custom_osl_values=parsed_workflow_args.get("custom_osl_values"),
            custom_concurrency_values=parsed_workflow_args.get("custom_concurrency_values"),
            custom_num_prompts_strategy=parsed_workflow_args.get("custom_num_prompts_strategy"),
            only_match_concurrency=parsed_workflow_args.get("only_match_concurrency", False),
            use_server_tokenizer=parsed_workflow_args.get("use_server_tokenizer", False),
            
            # Runtime
            model_spec=model_spec,
        )
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.model:
            raise ValueError("model is required")
        if not self.device:
            raise ValueError("device is required")

