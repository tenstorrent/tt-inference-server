#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
TT-NN Visualizer configuration presets for tt-inference-server.

Provides optimized TTNN_CONFIG_OVERRIDES for different debugging scenarios.
"""

import json
import os
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Instrumentation levels
InstrumentationLevel = Literal["off", "light", "full"]


def get_ttnn_config(
    level: InstrumentationLevel,
    report_name: str | None = None,
) -> dict | None:
    """
    Get TTNN_CONFIG_OVERRIDES for specified instrumentation level.
    
    Args:
        level: Instrumentation level (off, light, full)
        report_name: Custom report name (default: auto-generated)
        
    Returns:
        Configuration dict or None if level is 'off'
    """
    if level == "off":
        return None
    
    # Base configuration - common to all levels
    base_config = {
        "enable_fast_runtime_mode": False,  # Required for reporting
        "enable_logging": True,
        "report_name": report_name or "ttnn_report",
    }
    
    if level == "light":
        # Minimal overhead: Only high-level operation tracking
        return {
            **base_config,
            "enable_graph_report": False,
            "enable_detailed_buffer_report": False,  # Skip detailed buffer traces
            "enable_detailed_tensor_report": False,  # Skip tensor shape details
            "enable_comparison_mode": False,
            # Light mode focuses on operation counts and memory watermarks
        }
    
    elif level == "full":
        # Complete analysis: Full instrumentation
        return {
            **base_config,
            "enable_graph_report": True,  # Operation dependency graph
            "enable_detailed_buffer_report": True,  # Full buffer lifecycle
            "enable_detailed_tensor_report": True,  # Tensor shapes and sharding
            "enable_comparison_mode": False,
        }
    
    return None


def setup_ttnn_visualizer(
    model_spec,
    level: InstrumentationLevel = "off",
) -> Path | None:
    """
    Configure TTNN_CONFIG_OVERRIDES environment variable and return report directory.
    
    Args:
        model_spec: ModelSpec object containing run configuration
        level: Instrumentation level
        
    Returns:
        Path to reports directory if enabled, None otherwise
    """
    if level == "off":
        return None
    
    # Generate report name based on workflow and model
    workflow = getattr(model_spec.cli_args, 'workflow', 'unknown')
    report_name = f"{model_spec.model_name}_{workflow}_{level}"
    
    # Get configuration for specified level
    ttnn_config = get_ttnn_config(level, report_name)
    
    if ttnn_config is None:
        return None
    
    # Check if user already set TTNN_CONFIG_OVERRIDES
    existing_config = os.getenv("TTNN_CONFIG_OVERRIDES")
    if existing_config:
        try:
            user_config = json.loads(existing_config)
            # User config takes precedence over defaults
            ttnn_config.update(user_config)
            logger.info(f"Merged user TTNN_CONFIG_OVERRIDES with {level} preset")
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse existing TTNN_CONFIG_OVERRIDES, using {level} preset"
            )
    
    # Set environment variable
    os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps(ttnn_config)
    
    # Calculate expected report directory
    # Reports are saved to: persistent_volume/volume_id_*/ttnn_reports/
    from workflows.setup_host import SetupConfig
    setup_config = SetupConfig(model_spec)
    reports_dir = setup_config.host_model_volume_root / "ttnn_reports"
    
    logger.info(f"✅ TT-NN Visualizer enabled: level={level}, report={report_name}")
    logger.info(f"   Reports → {reports_dir}")
    logger.info(f"   Expected overhead: ~{get_overhead_estimate(level)}")
    
    return reports_dir


def get_overhead_estimate(level: InstrumentationLevel) -> str:
    """Get estimated performance overhead for instrumentation level."""
    overhead_map = {
        "off": "0%",
        "light": "5-10%",
        "full": "20-30%",
    }
    return overhead_map.get(level, "unknown")


def should_enable_for_workflow(workflow: str) -> bool:
    """
    Determine if TTNN visualizer should be recommended for a workflow.
    
    Some workflows benefit more from instrumentation than others.
    """
    # Evals and benchmarks run multiple iterations - good for profiling
    if workflow in ["evals", "benchmarks"]:
        return True
    
    # Server workflow is long-running - instrumentation overhead matters more
    if workflow == "server":
        return False
    
    return False


def get_recommendation(model_spec) -> str:
    """Get usage recommendation based on workflow and model."""
    workflow = getattr(model_spec.cli_args, 'workflow', 'unknown')
    
    if workflow == "server":
        return (
            "💡 For server workflow, use ttnn-visualizer only for debugging.\n"
            "   Add --ttnn-visualizer light for minimal overhead during live testing."
        )
    elif workflow in ["evals", "benchmarks"]:
        return (
            "💡 For evals/benchmarks, consider:\n"
            "   --ttnn-visualizer light  → Quick memory/performance check\n"
            "   --ttnn-visualizer full   → Deep analysis for optimization"
        )
    else:
        return ""

