#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceTarget:
    ttft_ms: float = None
    tput_user: float = None
    tput: float = None
    tolerance: float = 0.0


@dataclass
class PerformanceTargets:
    """Parsed performance targets from model_performance_reference.json"""

    ttft_ms: float = None
    ttft_streaming_ms: float = None
    tput_user: float = None
    tput_prefill: float = None
    e2el_ms: float = None
    tput: float = None
    rtr: float = None
    tolerance: float = 0.05
    max_concurrency: int = None
    num_eval_runs: int = None
    task_type: str = "text"

    @classmethod
    def from_device_config(cls, device_config: Dict) -> PerformanceTargets:
        """Create PerformanceTargets from device configuration dict"""
        if not device_config:
            return cls()

        # Extract from theoretical targets
        theoretical = device_config.get("targets", {}).get("theoretical", {})

        return cls(
            ttft_ms=theoretical.get("ttft_ms"),
            ttft_streaming_ms=theoretical.get("ttft_streaming_ms"),
            tput_user=theoretical.get("tput_user"),
            tput_prefill=theoretical.get("tput_prefill"),
            e2el_ms=theoretical.get("e2el_ms"),
            tput=theoretical.get("tput"),
            rtr=theoretical.get("rtr"),
            tolerance=theoretical.get("tolerance", 0.05),
            max_concurrency=device_config.get("max_concurrency"),
            num_eval_runs=device_config.get("num_eval_runs"),
            task_type=device_config.get("task_type", "text"),
        )


def get_performance_targets(
    model_name: str, device_str: str, model_type: str = None
) -> PerformanceTargets:
    """Extract device-specific performance targets for a model.

    Handles model name mapping (e.g., distil-whisper variants) and returns
    parsed performance targets in a type-safe format.

    Args:
        model_name: Name of the model
        device_str: Device string (e.g., 'galaxy', 't3k', 'n150')
        model_type: Model type (e.g., 'AUDIO', 'TEXT', 'CNN') - optional for backward compatibility

    Returns:
        PerformanceTargets object with parsed targets
    """
    device_str = device_str.lower()

    # Import here to avoid circular dependency
    from workflows.model_spec import model_performance_reference

    # Get model performance targets
    model_data = model_performance_reference.get(model_name, {})
    device_json_list = model_data.get(device_str, [])

    # Return first config if available
    if device_json_list:
        logger.info(
            f"Found performance targets for model '{model_name}' on device '{device_str}'"
        )
        return PerformanceTargets.from_device_config(device_json_list[0])

    logger.warning(
        f"No performance targets found for model '{model_name}' on device '{device_str}'"
    )
    return PerformanceTargets()


@dataclass
class BenchmarkTaskParams:
    isl: int = None
    osl: int = None
    max_concurrency: int = None
    num_prompts: int = None
    image_height: int = None
    image_width: int = None
    images_per_prompt: int = 0
    task_type: str = "text"
    theoretical_ttft_ms: float = None
    theoretical_tput_user: float = None
    targets: Dict[str, PerformanceTarget] = field(default_factory=dict)
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.10,
            "customer_complete": 0.50,
            "customer_sellable": 0.80,
        }
    )

    # has to go in here so init can read it
    num_inference_steps: int = None  # Used for CNN models

    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        for target_name, peak_perf in self.target_peak_perf.items():
            if target_name not in self.targets.keys():
                if self.theoretical_ttft_ms or self.theoretical_tput_user:
                    self.targets[target_name] = PerformanceTarget(
                        ttft_ms=self.theoretical_ttft_ms / peak_perf
                        if self.theoretical_ttft_ms
                        else None,
                        tput_user=self.theoretical_tput_user * peak_perf
                        if self.theoretical_tput_user
                        else None,
                    )


@dataclass
class BenchmarkTaskParamsCNN(BenchmarkTaskParams):
    num_eval_runs: int = 15
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.30,
            "customer_complete": 0.70,
            "customer_sellable": 0.80,
        }
    )

    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        super()._infer_data()
