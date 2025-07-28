# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Set, Dict, List

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
ttnn_optimized_functional_whisper = ImplConfig(
    impl_id="ttnn-optimized-functional-whisper",
    impl_name="ttnn-optimized-functional-whisper",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/whisper",
)


@dataclass(frozen=True)
class ModelConfig:
    """
    All static configuration and metadata required to execute workflows for a given model.
    Note: model_name is unique from hf_model_repo so that we can have multiple
    implementations of the same model, for example from tt-metal and tt-forge.

    For details on tt-metal/TTNN implementation context limits see: https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers#implementation-notes
    """

    device_configurations: Set[DeviceTypes]
    impl: ImplConfig
    tt_metal_commit: str
    vllm_commit: str = None
    default_impl_map: Dict[DeviceTypes, bool] = field(default_factory=dict)
    hf_model_repo: str = None
    model_id: str = None
    model_name: str = None  # uses defaults based on hf_model_repo
    param_count: int = None
    min_disk_gb: int = None
    min_ram_gb: int = None
    repacked: int = 0
    version: str = "0.0.1"
    perf_targets_map: Dict[str, float] = field(default_factory=dict)
    weights: List[str] = field(default_factory=list)
    docker_image: str = None
    max_concurrency_map: Dict[DeviceTypes, int] = field(default_factory=dict)
    max_context_map: Dict[DeviceTypes, int] = field(default_factory=dict)
    status: str = "preview"  # default status for all models
    code_link: str = None
    perf_reference_map: Dict[DeviceTypes, List[BenchmarkTaskParams]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen
        if not self.hf_model_repo:
            # use first weight as default hf_model_repo
            object.__setattr__(self, "hf_model_repo", self.weights[0])

        if not self.model_name:
            # use basename of HF model ID to use same format as tt-transformers
            object.__setattr__(self, "model_name", Path(self.hf_model_repo).name)
        if not self.model_id:
            # do not set a device because model config can have many device configurations
            object.__setattr__(
                self,
                "model_id",
                get_model_id(self.impl.impl_name, self.model_name, None),
            )

        # use param count to detemine conservative disk and ram minimums
        # these are only checked during initial model setup
        if not self.param_count:
            object.__setattr__(
                self, "param_count", ModelConfig.infer_param_count(self.hf_model_repo)
            )
        if not self.min_disk_gb:
            if self.repacked:
                # 2x for raw fp16 weights hf cache (may already be present)
                # 1x for repacked quantized copy
                # 1x for tt-metal cache
                # 1x for overhead
                object.__setattr__(self, "min_disk_gb", self.param_count * 5)
            else:
                # 2x for raw fp16 weights hf cache (may already be present
                # 2x for copy
                object.__setattr__(self, "min_disk_gb", self.param_count * 4)
        if not self.min_ram_gb:
            object.__setattr__(self, "min_ram_gb", self.param_count * 5)

        if not self.docker_image:
            # Note: default to release image, use --dev-mode at runtime to use dev images
            # TODO: Use ubuntu version to interpolate this string
            _default_docker_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64"
            _max_tag_len = 128
            _default_docker_tag = f"{VERSION}-{self.tt_metal_commit[:_max_tag_len]}-{self.vllm_commit[:_max_tag_len]}"
            object.__setattr__(
                self, "docker_image", f"{_default_docker_repo}:{_default_docker_tag}"
            )

        # add GPU device for reference testing
        _device_set = self.device_configurations.copy()
        _device_set.add(DeviceTypes.GPU)
        object.__setattr__(self, "device_configurations", _device_set)

        # Fill default_impl_map for all device types if not provided
        if not self.default_impl_map:
            _default_impl_map = {}
            for device in self.device_configurations:
                _default_impl_map[device] = False
            object.__setattr__(self, "default_impl_map", _default_impl_map)

        if not self.max_concurrency_map:
            _default_max_concurrent = 32
            object.__setattr__(
                self,
                "max_concurrency_map",
                {
                    device: _default_max_concurrent
                    for device in self.device_configurations
                },
            )

        if not self.max_context_map:
            _default_max_context = 128 * 1024
            object.__setattr__(
                self,
                "max_context_map",
                {device: _default_max_context for device in self.device_configurations},
            )

        if not self.perf_targets_map:
            # performance targets expressed as percentage of theoretical performance
            default_perf_targets_map = {
                "functional": 0.10,
                "complete": 0.50,
                "target": 0.80,
            }
            object.__setattr__(self, "perf_targets_map", default_perf_targets_map)

        if not self.perf_reference_map:
            object.__setattr__(
                self,
                "perf_reference_map",
                get_perf_reference_map(self.model_name, self.perf_targets_map),
            )

        if not self.code_link:
            object.__setattr__(
                self,
                "code_link",
                f"{self.impl.repo_url}/tree/{self.tt_metal_commit}/{self.impl.code_path}",
            )

    def validate_data(self):
        assert (
            self.hf_model_repo or self.model_name or self.weights
        ), "either hf_model_repo or model_name must be set."

    @staticmethod
    def infer_param_count(hf_model_repo: str) -> int:
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


config_list = [
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.T3K},
        weights=["Qwen/QwQ-32B"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
    ),
    ModelConfig(
        impl=llama3_impl,
        default_impl_map={
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.T3K},
        weights=["Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"],
        tt_metal_commit="v0.56.0-rc33",
        vllm_commit="e2e0002ac7dc",
        status="testing",
    ),
    ModelConfig(
        impl=llama3_impl,
        default_impl_map={
            DeviceTypes.N300: True,
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.N300, DeviceTypes.T3K},
        weights=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
        tt_metal_commit="v0.56.0-rc33",
        vllm_commit="e2e0002ac7dc",
        status="testing",
    ),
    ModelConfig(
        impl=llama3_subdevices_impl,
        default_impl_map={
            DeviceTypes.GALAXY: True,
        },
        device_configurations={DeviceTypes.GALAXY},
        weights=[
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        tt_metal_commit="60e367fcc471",
        vllm_commit="8a43c881e",
        status="testing",
        max_context_map={
            DeviceTypes.GALAXY: 128 * 1024,
        },
    ),
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.T3K},
        weights=[
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
    ),
    ModelConfig(
        impl=t3000_llama2_70b_impl,
        default_impl_map={
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.T3K},
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
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.N300: True,
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.N300, DeviceTypes.T3K},
        weights=[
            "meta-llama/Llama-3.2-11B-Vision",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="testing",
        max_concurrency_map={
            DeviceTypes.N300: 16,
            DeviceTypes.T3K: 16,
        },
        max_context_map={
            DeviceTypes.N300: 128 * 1024,
            DeviceTypes.T3K: 128 * 1024,
        },
    ),
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.N150: True,
            DeviceTypes.N300: True,
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        weights=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
    ),
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.N150: True,
            DeviceTypes.N300: True,
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        weights=["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
    ),
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.N150: True,
            DeviceTypes.N300: True,
            DeviceTypes.T3K: True,
        },
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        status="ready",
        max_context_map={
            DeviceTypes.N150: 64 * 1024,
            DeviceTypes.N300: 128 * 1024,
            DeviceTypes.T3K: 128 * 1024,
            DeviceTypes.GPU: 128 * 1024,
        },
    ),
    ModelConfig(
        impl=tt_transformers_impl,
        default_impl_map={
            DeviceTypes.P100: True,
            DeviceTypes.P150: True,
        },
        device_configurations={DeviceTypes.P100, DeviceTypes.P150},
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        tt_metal_commit="v0.59.0-rc3",
        vllm_commit="8a43c88",
        status="preview",
        max_context_map={
            DeviceTypes.P100: 64 * 1024,
            DeviceTypes.P150: 64 * 1024,
        },
    ),
    ModelConfig(
        impl=ttnn_optimized_functional_whisper,
        default_impl_map={
            DeviceTypes.N150: True,
            DeviceTypes.N300: True,
        },
        device_configurations={
            DeviceTypes.N150,
            DeviceTypes.N300,
            DeviceTypes.P100,
            DeviceTypes.P150,
        },
        weights=["openai/whisper-large-v3"],
        tt_metal_commit="v0.61.0-rc1",
        vllm_commit="3dc6c31",
        param_count=1,
        status="preview",
    ),
    ModelConfig(
        impl=ttnn_optimized_functional_whisper,
        default_impl_map={
            DeviceTypes.N150: True,
            DeviceTypes.N300: True,
        },
        device_configurations={
            DeviceTypes.N150,
            DeviceTypes.N300,
            DeviceTypes.P100,
            DeviceTypes.P150,
        },
        weights=["distil-whisper/distil-large-v3"],
        tt_metal_commit="v0.61.0-rc1",
        vllm_commit="3dc6c31",
        # docker_image="ghcr.io/tenstorrent/tt-inference-server/tt-metal-whisper-distil-large-v3-dev:v0.0.1-tt-metal-07567d1618a8",
        param_count=1,
        status="preview",
    ),
]


# Generate a dictionary keyed by the model_name for each ModelConfig instance
def get_model_config_map(config_list: List[ModelConfig]) -> Dict[str, ModelConfig]:
    model_config_map = {}
    for config in config_list:
        for w in config.weights:
            for device_type in config.device_configurations:
                # make an instance for each finetune weights that can be further modified
                _model_name = Path(w).name
                _model_id = get_model_id(
                    config.impl.impl_id, _model_name, device_type.name.lower()
                )
                _model_config = replace(
                    config, model_name=_model_name, hf_model_repo=w, model_id=_model_id
                )
                model_config_map[_model_id] = _model_config
    return model_config_map


MODEL_CONFIGS = get_model_config_map(config_list)
