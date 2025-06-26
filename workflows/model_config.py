# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from workflows.utils import (
    get_version,
    BenchmarkTaskParams,
    PerformanceTarget,
    get_model_id,
    get_repo_root_path,
)
from workflows.workflow_types import DeviceTypes

VERSION = get_version()


def read_performance_reference_json() -> Dict[DeviceTypes, List[BenchmarkTaskParams]]:
    default_filepath = (
        get_repo_root_path()
        / "benchmarking"
        / "benchmark_targets"
        / "model_performance_reference.json"
    )
    filepath = Path(os.getenv("OVERRIDE_BENCHMARK_TARGETS", default_filepath))
    assert filepath.exists(), f"Override benchmark file not found: {filepath}"
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


model_performance_reference = read_performance_reference_json()


def get_perf_reference_map(
    model_name: str, perf_targets_map: Dict[str, float]
) -> Dict[DeviceTypes, List[BenchmarkTaskParams]]:
    perf_reference_map: Dict[DeviceTypes, List[BenchmarkTaskParams]] = {}
    model_data = model_performance_reference.get(model_name, {})

    for device_str, benchmarks in model_data.items():
        device_type = DeviceTypes.from_string(device_str)

        params_list: List[BenchmarkTaskParams] = []

        for bench in benchmarks:
            # Parse performance targets under the "reference" key.
            target_dict = {}
            targets = bench.get("targets", {})
            for target_name, target_data in targets.items():
                # Create the PerformanceTarget instance.
                if target_name == "theoretical":
                    # add customer definitions: functional, complete, sellable
                    for target_key, percentage in perf_targets_map.items():
                        target_dict[target_key] = PerformanceTarget(
                            ttft_ms=target_data.get("ttft_ms") / percentage
                            if target_data.get("ttft_ms")
                            else None,
                            tput_user=target_data.get("tput_user") * percentage
                            if target_data.get("tput_user")
                            else None,
                            tput=target_data.get("tput") * percentage
                            if target_data.get("tput")
                            else None,
                        )

            # Create the BenchmarkTaskParams instance.
            benchmark_task = BenchmarkTaskParams(
                isl=bench.get("isl"),
                osl=bench.get("osl"),
                max_concurrency=bench.get("max_concurrency"),
                num_prompts=bench.get("num_prompts"),
                task_type=bench.get("task_type", "text"),
                image_height=bench.get("image_height", None),
                image_width=bench.get("image_width", None),
                images_per_prompt=bench.get("images_per_prompt", None),
                targets=target_dict,
            )
            params_list.append(benchmark_task)

        perf_reference_map[device_type] = params_list
    return perf_reference_map


@dataclass(frozen=True)
class ImplConfig:
    impl_id: str
    impl_name: str
    repo_url: str
    code_path: str


tt_transformers_impl = ImplConfig(
    impl_id="tt-transformers",
    impl_name="tt-transformers",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/tt_transformers",
)
llama3_impl = ImplConfig(
    impl_id="llama3",
    impl_name="llama3",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/llama3",
)
t3000_llama2_70b_impl = ImplConfig(
    impl_id="t3000-llama2-70b",
    impl_name="t3000-llama2-70b",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/t3000/llama2_70b",
)
llama3_subdevices_impl = ImplConfig(
    impl_id="subdevices",
    impl_name="subdevices",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/llama3_subdevices",
)


@dataclass(frozen=True)
class DeviceModelSpec:
    """
    Model-specific configuration for a specific device.
    """

    max_concurrency: int
    max_context: int
    perf_targets_map: Dict[str, float] = field(default_factory=dict)
    default_impl: bool = False
    perf_reference: List[BenchmarkTaskParams] = field(default_factory=list)
    vllm_override_args: Dict[str, str] = field(default_factory=dict)
    override_tt_config: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        """Validate that required configuration is present."""
        pass

    def _infer_data(self):
        """Infer missing data fields from other configuration values."""
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen
        # Set default concurrency and context if not provided
        if not self.max_concurrency:
            _default_max_concurrent = 32
            object.__setattr__(self, "max_concurrency", _default_max_concurrent)

        if not self.max_context:
            _default_max_context = 128 * 1024
            object.__setattr__(self, "max_context", _default_max_context)


@dataclass(frozen=True)
class ModelConfig:
    """
    Fully instantiated configuration for a specific model on a specific device.
    This is what gets used throughout the system after template expansion.
    """

    # Core identity - required fields
    model_id: str
    impl: ImplConfig
    hf_model_repo: str
    model_name: str
    device_type: DeviceTypes  # Single device, not a set

    # Version control
    tt_metal_commit: str
    vllm_commit: str

    # Device-specific configuration
    device_model_spec: DeviceModelSpec

    # Optional configuration fields
    param_count: Optional[int] = None
    min_disk_gb: Optional[int] = None
    min_ram_gb: Optional[int] = None
    repacked: int = 0
    version: str = "0.0.1"
    docker_image: Optional[str] = None
    status: str = "preview"
    code_link: Optional[str] = None
    override_tt_config: Dict[str, str] = field(default_factory=dict)
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])
    subdevice_type: Optional[DeviceTypes] = None # Used for data-parallel configurations

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        """Infer missing data fields from other configuration values."""
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen

        # Infer param count from model repo name
        if not self.param_count:
            object.__setattr__(
                self, "param_count", ModelConfig.infer_param_count(self.hf_model_repo)
            )

        # Calculate conservative disk and ram minimums based on param count
        if not self.min_disk_gb and self.param_count:
            if self.repacked:
                # 2x for raw fp16 weights hf cache (may already be present)
                # 1x for repacked quantized copy
                # 1x for tt-metal cache
                # 1x for overhead
                object.__setattr__(self, "min_disk_gb", self.param_count * 5)
            else:
                # 2x for raw fp16 weights hf cache (may already be present)
                # 2x for copy
                object.__setattr__(self, "min_disk_gb", self.param_count * 4)

        if not self.min_ram_gb and self.param_count:
            object.__setattr__(self, "min_ram_gb", self.param_count * 5)

        # Generate default docker image if not provided
        if not self.docker_image:
            # Note: default to release image, use --dev-mode at runtime to use dev images
            # TODO: Use ubuntu version to interpolate this string
            _default_docker_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64"
            _max_tag_len = 12
            _default_docker_tag = f"{VERSION}-{self.tt_metal_commit[:_max_tag_len]}-{self.vllm_commit[:_max_tag_len]}"
            object.__setattr__(
                self, "docker_image", f"{_default_docker_repo}:{_default_docker_tag}"
            )

        # Generate code link
        if not self.code_link:
            object.__setattr__(
                self,
                "code_link",
                f"{self.impl.repo_url}/tree/{self.tt_metal_commit}/{self.impl.code_path}",
            )

        if self.override_tt_config and "data_parallel" in self.override_tt_config:
            data_parallel = self.override_tt_config["data_parallel"]
            object.__setattr__(
                self,
                "subdevice_type",
                self.device_type.get_data_parallel_subdevice(data_parallel),
            )

    def validate_data(self):
        """Validate that required configuration is present."""
        assert self.hf_model_repo, "hf_model_repo must be set"
        assert self.model_name, "model_name must be set"
        assert self.model_id, "model_id must be set"

    @staticmethod
    def infer_param_count(hf_model_repo: str) -> Optional[int]:
        """
        Infers the parameter count (in billions) from the hf_model_repo string.

        Examples:
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" -> 70
            "Qwen/Qwen2.5-72B" -> 72
            "meta-llama/Llama-3.1-8B" -> 8

        Returns:
            The inferred parameter count as an int, or None if no pattern is found.
        """
        matches = re.findall(r"(\d+(?:\.\d+)?)B", hf_model_repo)
        if matches:
            # Take the last match which is typically the parameter count
            count_str = matches[-1]
            try:
                # Convert to float first (to handle potential decimals) and then to int.
                count_float = float(count_str)
                return int(count_float)
            except ValueError:
                return None
        return None


@dataclass(frozen=True)
class ModelConfigTemplate:
    """
    Template configuration that gets expanded into individual ModelConfig instances
    for each weight variant and device combination. This represents the shared configuration
    across multiple models and devices.
    """

    # Required fields
    impl: ImplConfig
    tt_metal_commit: str
    vllm_commit: str
    device_model_spec_map: Dict[DeviceTypes, DeviceModelSpec]
    weights: List[str]  # List of HF model repos to create configs for

    # Optional template fields
    repacked: int = 0
    version: str = "0.0.1"
    perf_targets_map: Dict[str, float] = field(default_factory=dict)
    docker_image: Optional[str] = None
    status: str = "preview"
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        """Validate that required configuration is present."""
        assert self.device_model_spec_map, "device_model_spec_map must be provided"
        assert self.weights, "weights must be provided"

    def _infer_data(self):
        """Infer missing data fields from other configuration values."""
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen
        # Set default performance targets if not provided
        if not self.perf_targets_map:
            # performance targets expressed as percentage of theoretical performance
            default_perf_targets_map = {
                "functional": 0.10,
                "complete": 0.50,
                "target": 1.0,
            }
            object.__setattr__(self, "perf_targets_map", default_perf_targets_map)

    def expand_to_configs(self) -> List["ModelConfig"]:
        """Expand this template into individual ModelConfig instances."""
        configs = []

        # Generate performance reference map
        main_model_name = Path(self.weights[0]).name
        perf_reference_map = get_perf_reference_map(
            main_model_name, self.perf_targets_map
        )

        for weight in self.weights:
            for device_type, device_model_spec in self.device_model_spec_map.items():
                model_name = Path(weight).name
                model_id = get_model_id(
                    self.impl.impl_id, model_name, device_type.name.lower()
                )

                # Create a new device_model_spec with performance reference data
                device_model_spec_with_perf = DeviceModelSpec(
                    max_concurrency=device_model_spec.max_concurrency,
                    max_context=device_model_spec.max_context,
                    perf_targets_map=device_model_spec.perf_targets_map,
                    default_impl=device_model_spec.default_impl,
                    perf_reference=perf_reference_map.get(device_type, []),
                    vllm_override_args=device_model_spec.vllm_override_args,
                    override_tt_config=device_model_spec.override_tt_config,
                )

                config = ModelConfig(
                    # Core identity
                    device_type=device_type,
                    impl=self.impl,
                    hf_model_repo=weight,
                    model_id=model_id,
                    model_name=model_name,
                    device_model_spec=device_model_spec_with_perf,
                    # Version control
                    tt_metal_commit=self.tt_metal_commit,
                    vllm_commit=self.vllm_commit,
                    # Template fields
                    repacked=self.repacked,
                    version=self.version,
                    docker_image=self.docker_image,
                    status=self.status,
                    override_tt_config=device_model_spec.override_tt_config,
                    supported_modalities=self.supported_modalities,
                )
                configs.append(config)
        return configs


# Model configuration templates - these get expanded into individual configs
config_templates = [
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        weights=["Qwen/Qwen3-32B"],
        device_model_spec_map={
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            )
        },
        tt_metal_commit="v0.59.0-rc39",
        vllm_commit="3accc8d",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        weights=["mistralai/Mistral-7B-Instruct-v0.3"],
        device_model_spec_map={
            DeviceTypes.N150: DeviceModelSpec(
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
            ),
            DeviceTypes.N300: DeviceModelSpec(
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
            ),
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
            ),
        },
        tt_metal_commit="v0.59.0-rc16",
        vllm_commit="dff84a3",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=["Qwen/QwQ-32B"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=llama3_impl,
        device_model_spec_map={
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=["Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"],
        tt_metal_commit="v0.56.0-rc33",
        vllm_commit="e2e0002ac7dc",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=llama3_impl,
        device_model_spec_map={
            DeviceTypes.N300: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
        tt_metal_commit="v0.56.0-rc33",
        vllm_commit="e2e0002ac7dc",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=llama3_subdevices_impl,
        device_model_spec_map={
            DeviceTypes.GALAXY: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                vllm_override_args={
                    "num_scheduler_steps": 10,
                },
                override_tt_config={
                    "dispatch_core_axis": "col",
                    "sample_on_device_mode": "all",
                    "fabric_config": "FABRIC_1D",
                    "worker_l1_size": 1344544,
                    "trace_region_size": 95693824,
                },
            ),
        },
        weights=[
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        tt_metal_commit="b6ed13f0dff9",
        vllm_commit="1954a11",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=[
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        tt_metal_commit="v0.59.0-rc14",
        vllm_commit="a869e5d",
        status="testing",
    ),
    ModelConfigTemplate(
        impl=t3000_llama2_70b_impl,
        device_model_spec_map={
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=False,
            ),
        },
        repacked=1,
        weights=[
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
        ],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.N300: DeviceModelSpec(
                max_concurrency=16,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=16,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=[
            "meta-llama/Llama-3.2-11B-Vision",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
        supported_modalities=["text", "image"],
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.N150: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.N300: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.N150: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.N300: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        },
        weights=["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.N150: DeviceModelSpec(
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceTypes.N300: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.T3K: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceTypes.GPU: DeviceModelSpec(
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=False,
            ),
        },
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.P100: DeviceModelSpec(
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceTypes.P150: DeviceModelSpec(
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
        },
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        tt_metal_commit="v0.59.0-rc3",
        vllm_commit="8a43c88",
        status="preview",
    ),
    ModelConfigTemplate(
        impl=tt_transformers_impl,
        device_model_spec_map={
            DeviceTypes.GALAXY: DeviceModelSpec(
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
                override_tt_config={
                    "data_parallel": 16,
                },
            ),
        },
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        tt_metal_commit="v0.59.0-rc26",
        vllm_commit="a869e5d",
        status="preview",
    ),
]


def get_model_config_map(
    templates: List[ModelConfigTemplate],
) -> Dict[str, ModelConfig]:
    """
    Generate final model configurations from templates.

    Args:
        templates: List of ModelConfigTemplate instances to expand

    Returns:
        Dictionary mapping model_id to ModelConfig instances
    """
    model_config_map = {}
    for template in templates:
        for config in template.expand_to_configs():
            model_config_map[config.model_id] = config
    return model_config_map


# Final model configurations generated from templates
MODEL_CONFIGS = get_model_config_map(config_templates)
