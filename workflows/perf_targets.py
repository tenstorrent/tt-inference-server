#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from workflows.utils import get_repo_root_path
from workflows.workflow_types import DeviceTypes

logger = logging.getLogger(__name__)


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
    targets: Dict[str, PerfTarget] = field(default_factory=dict)
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.10,
            "customer_complete": 0.50,
            "customer_sellable": 0.80,
        }
    )
    num_inference_steps: int = None

    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        for target_name, peak_perf in self.target_peak_perf.items():
            if target_name not in self.targets and (
                self.theoretical_ttft_ms or self.theoretical_tput_user
            ):
                self.targets[target_name] = PerfTarget(
                    ttft_ms=self.theoretical_ttft_ms / peak_perf
                    if self.theoretical_ttft_ms is not None
                    else None,
                    tput_user=self.theoretical_tput_user * peak_perf
                    if self.theoretical_tput_user is not None
                    else None,
                    target_name=target_name,
                    is_derived=True,
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


def get_perf_targets_path() -> Path:
    return (
        get_repo_root_path()
        / "benchmarking"
        / "benchmark_targets"
        / "model_performance_reference.json"
    )


def load_perf_targets_json(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    filepath = Path(
        os.getenv("OVERRIDE_BENCHMARK_TARGETS", path or get_perf_targets_path())
    )
    assert filepath.exists(), f"Override benchmark file not found: {filepath}"
    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)


model_performance_reference = load_perf_targets_json()


@dataclass(frozen=True)
class PerfTarget:
    isl: int = None
    osl: int = None
    max_concurrency: int = None
    num_prompts: int = None
    task_type: str = "text"
    image_height: int = None
    image_width: int = None
    images_per_prompt: int = 0
    ttft_ms: float = None
    ttft_streaming_ms: float = None
    tput_user: float = None
    tput_prefill: float = None
    e2el_ms: float = None
    tput: float = None
    rtr: float = None
    tolerance: float = 0.05
    num_eval_runs: int = None
    num_inference_steps: int = None
    target_name: Optional[str] = None
    is_derived: bool = False
    is_summary: bool = False

    @classmethod
    def from_dict(cls, row: Dict[str, Any], is_summary: bool = False) -> "PerfTarget":
        theoretical = row.get("targets", {}).get("theoretical", {})
        return cls(
            isl=row.get("isl"),
            osl=row.get("osl"),
            max_concurrency=row.get("max_concurrency"),
            num_prompts=row.get("num_prompts"),
            task_type=row.get("task_type", "text"),
            image_height=row.get("image_height"),
            image_width=row.get("image_width"),
            images_per_prompt=row.get("images_per_prompt", 0),
            ttft_ms=theoretical.get("ttft_ms"),
            ttft_streaming_ms=theoretical.get("ttft_streaming_ms"),
            tput_user=theoretical.get("tput_user"),
            tput_prefill=theoretical.get("tput_prefill"),
            e2el_ms=theoretical.get("e2el_ms"),
            tput=theoretical.get("tput"),
            rtr=theoretical.get("rtr"),
            tolerance=theoretical.get("tolerance", 0.05),
            num_eval_runs=row.get("num_eval_runs"),
            num_inference_steps=row.get("num_inference_steps"),
            target_name=None,
            is_derived=False,
            is_summary=is_summary,
        )

    def build_threshold_targets(
        self, perf_targets_map: Dict[str, float]
    ) -> Dict[str, "PerfTarget"]:
        targets: Dict[str, PerfTarget] = {}
        latency_metrics = ("ttft_ms", "ttft_streaming_ms", "e2el_ms")
        throughput_metrics = ("tput_user", "tput_prefill", "tput", "rtr")

        for target_name, percentage in perf_targets_map.items():
            target_values = {
                "isl": self.isl,
                "osl": self.osl,
                "max_concurrency": self.max_concurrency,
                "num_prompts": self.num_prompts,
                "task_type": self.task_type,
                "image_height": self.image_height,
                "image_width": self.image_width,
                "images_per_prompt": self.images_per_prompt,
                "num_eval_runs": self.num_eval_runs,
                "num_inference_steps": self.num_inference_steps,
                "tolerance": self.tolerance,
                "target_name": target_name,
                "is_derived": True,
                "is_summary": False,
            }
            for metric_name in latency_metrics:
                metric_value = getattr(self, metric_name)
                target_values[metric_name] = (
                    metric_value / percentage if metric_value is not None else None
                )
            for metric_name in throughput_metrics:
                metric_value = getattr(self, metric_name)
                target_values[metric_name] = (
                    metric_value * percentage if metric_value is not None else None
                )
            targets[target_name] = PerfTarget(**target_values)
        return targets

    def to_benchmark_task_params(
        self, perf_targets_map: Dict[str, float]
    ) -> BenchmarkTaskParams:
        return BenchmarkTaskParams(
            isl=self.isl,
            osl=self.osl,
            max_concurrency=self.max_concurrency,
            num_prompts=self.num_prompts,
            task_type=self.task_type,
            image_height=self.image_height,
            image_width=self.image_width,
            images_per_prompt=self.images_per_prompt,
            targets=self.build_threshold_targets(perf_targets_map),
            num_inference_steps=self.num_inference_steps,
        )

    def scale_llm_for_data_parallel(self, data_parallel: int) -> "PerfTarget":
        max_concurrency = self.max_concurrency
        if max_concurrency and max_concurrency != 1:
            max_concurrency *= data_parallel
        return replace(
            self,
            max_concurrency=max_concurrency,
            tput=self.tput * data_parallel if self.tput else None,
        )

    def benchmark_identity(self) -> Dict[str, Any]:
        identity = {
            "task_type": self.task_type,
            "isl": self.isl,
            "osl": self.osl,
            "max_concurrency": self.max_concurrency,
        }
        if self.task_type == "vlm":
            identity.update(
                {
                    "image_height": self.image_height,
                    "image_width": self.image_width,
                    "images_per_prompt": self.images_per_prompt,
                }
            )
        return identity

    def summary_targets(self) -> Dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "ttft_streaming_ms": self.ttft_streaming_ms,
            "tput_user": self.tput_user,
            "tput_prefill": self.tput_prefill,
            "e2el_ms": self.e2el_ms,
            "tput": self.tput,
            "rtr": self.rtr,
            "tolerance": self.tolerance,
        }

    def matches_measurement(self, row: Dict[str, Any]) -> bool:
        row_isl = row.get("isl", row.get("input_sequence_length"))
        row_osl = row.get("osl", row.get("output_sequence_length"))
        row_concurrency = row.get("max_concurrency", row.get("max_con"))
        if (self.isl, self.osl, self.max_concurrency) != (
            row_isl,
            row_osl,
            row_concurrency,
        ):
            return False
        if self.task_type != row.get("task_type", self.task_type):
            return False
        if self.task_type == "vlm":
            return (
                self.image_height == row.get("image_height")
                and self.image_width == row.get("image_width")
                and self.images_per_prompt == row.get("images_per_prompt")
            )
        return True


@dataclass(frozen=True)
class PerfTargetSet:
    model_name: str
    device: DeviceTypes
    perf_targets: List[PerfTarget]

    def __post_init__(self):
        if not self.perf_targets:
            return
        summary_targets = [
            perf_target for perf_target in self.perf_targets if perf_target.is_summary
        ]
        if len(summary_targets) != 1:
            raise ValueError(
                "PerfTargetSet must contain exactly one summary PerfTarget"
            )

    @property
    def summary_perf_target(self) -> Optional[PerfTarget]:
        if not self.perf_targets:
            return None
        return next(
            (
                perf_target
                for perf_target in self.perf_targets
                if perf_target.is_summary
            ),
            None,
        )

    @property
    def summary_data_point(self) -> Optional[PerfTarget]:
        return self.summary_perf_target

    def to_benchmark_task_params(
        self, perf_targets_map: Dict[str, float]
    ) -> List[BenchmarkTaskParams]:
        return [
            perf_target.to_benchmark_task_params(perf_targets_map)
            for perf_target in self.perf_targets
        ]

    def scaled_for_data_parallel(self, data_parallel: int) -> "PerfTargetSet":
        return replace(
            self,
            perf_targets=[
                perf_target.scale_llm_for_data_parallel(data_parallel)
                for perf_target in self.perf_targets
            ],
        )

    def find_matching_row(
        self,
        rows: Iterable[Dict[str, Any]],
        perf_target: Optional[PerfTarget] = None,
    ) -> Optional[Dict[str, Any]]:
        target_perf_target = perf_target or self.summary_perf_target
        if target_perf_target is None:
            return None
        for row in rows:
            if target_perf_target.matches_measurement(row):
                return row
        return None


def _normalize_device(device: Union[str, DeviceTypes]) -> str:
    if isinstance(device, DeviceTypes):
        return device.name.lower()
    return str(device).lower()


def _build_perf_targets(rows: List[Dict[str, Any]]) -> List[PerfTarget]:
    return [
        PerfTarget.from_dict(row, is_summary=index == 0)
        for index, row in enumerate(rows)
    ]


def get_perf_target_map(model_name: str) -> Dict[DeviceTypes, PerfTargetSet]:
    model_data = model_performance_reference.get(model_name, {})
    perf_target_map: Dict[DeviceTypes, PerfTargetSet] = {}
    for device_str, rows in model_data.items():
        perf_target_map[DeviceTypes.from_string(device_str)] = PerfTargetSet(
            model_name=model_name,
            device=DeviceTypes.from_string(device_str),
            perf_targets=_build_perf_targets(rows),
        )
    return perf_target_map


def get_perf_target(
    model_name: str, device: Union[str, DeviceTypes]
) -> Optional[PerfTargetSet]:
    model_data = model_performance_reference.get(model_name, {})
    device_key = _normalize_device(device)
    rows = model_data.get(device_key, [])
    if not rows:
        return None
    return PerfTargetSet(
        model_name=model_name,
        device=DeviceTypes.from_string(device_key),
        perf_targets=_build_perf_targets(rows),
    )


def get_perf_target_rows(
    model_name: str, device: Union[str, DeviceTypes]
) -> List[Dict[str, Any]]:
    model_data = model_performance_reference.get(model_name, {})
    return list(model_data.get(_normalize_device(device), []))


def get_performance_targets(
    model_name: str, device_str: str, model_type: str = None
) -> PerfTarget:
    del model_type  # retained for compatibility with older callers
    perf_target_set = get_perf_target(model_name, device_str)
    if perf_target_set and perf_target_set.summary_perf_target:
        logger.info(
            f"Found performance targets for model '{model_name}' on device '{device_str}'"
        )
        return perf_target_set.summary_perf_target

    logger.warning(
        f"No performance targets found for model '{model_name}' on device '{device_str}'"
    )
    return PerfTarget()


def get_perf_reference_map(
    model_name: str, perf_targets_map: Dict[str, float]
) -> Dict[DeviceTypes, List[BenchmarkTaskParams]]:
    return {
        device: perf_target_set.to_benchmark_task_params(perf_targets_map)
        for device, perf_target_set in get_perf_target_map(model_name).items()
    }


def scale_llm_perf_targets(
    task: BenchmarkTaskParams, data_parallel: int
) -> BenchmarkTaskParams:
    scaled_targets = {
        target_name: replace(
            target,
            tput=target.tput * data_parallel if target.tput is not None else None,
            tput_prefill=target.tput_prefill * data_parallel
            if target.tput_prefill is not None
            else None,
        )
        for target_name, target in task.targets.items()
    }
    return BenchmarkTaskParams(
        isl=task.isl,
        osl=task.osl,
        max_concurrency=task.max_concurrency
        if task.max_concurrency == 1
        else task.max_concurrency * data_parallel,
        num_prompts=task.num_prompts,
        image_height=task.image_height,
        image_width=task.image_width,
        images_per_prompt=task.images_per_prompt,
        task_type=task.task_type,
        theoretical_ttft_ms=task.theoretical_ttft_ms,
        theoretical_tput_user=task.theoretical_tput_user,
        targets=scaled_targets,
        target_peak_perf=task.target_peak_perf,
        num_inference_steps=task.num_inference_steps,
    )


def get_perf_reference_for_device(
    device: DeviceTypes,
    override_tt_config: Optional[Dict[str, Any]],
    perf_reference_map: Dict[DeviceTypes, List[BenchmarkTaskParams]],
) -> List[BenchmarkTaskParams]:
    override_tt_config = override_tt_config or {}
    data_parallel = override_tt_config.get("data_parallel")

    if data_parallel:
        dp_device = device.get_data_parallel_subdevice(data_parallel)
        perf_reference = perf_reference_map.get(dp_device, [])
        if perf_reference:
            return [
                scale_llm_perf_targets(task, data_parallel) for task in perf_reference
            ]

    return perf_reference_map.get(device, [])
