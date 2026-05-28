# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import json
import os
import re
import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from workflows.utils import (
    get_repo_root_path,
    get_version,
    parse_commits_from_docker_image,
    parse_image_version,
)
from workflows.utils_report import BenchmarkTaskParams, PerformanceTarget
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    ModelType,
    VersionMode,
    WorkflowType,
)

if TYPE_CHECKING:
    from workflows.runtime_config import RuntimeConfig

VERSION = get_version()
MODEL_SPECS_SCHEMA_VERSION = "0.1.0"


def generate_docker_tag(
    version: str, tt_metal_commit: str, vllm_commit: Optional[str]
) -> str:
    max_tag_len = 12
    if vllm_commit:
        return f"{version}-{tt_metal_commit[:max_tag_len]}-{vllm_commit[:max_tag_len]}"
    else:
        return f"{version}-{tt_metal_commit[:max_tag_len]}"


def generate_default_docker_link(
    version: str,
    tt_metal_commit: str,
    vllm_commit: Optional[str],
    inference_engine: str = "",
    multihost: bool = False,
) -> str:
    _default_docker_tag = generate_docker_tag(version, tt_metal_commit, vllm_commit)
    if vllm_commit is not None:
        if multihost:
            _default_docker_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-multihost-ubuntu-22.04-amd64"
        else:
            _default_docker_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64"
    elif inference_engine == "forge":
        _default_docker_repo = "ghcr.io/tenstorrent/tt-media-inference-server-forge"
    else:
        _default_docker_repo = "ghcr.io/tenstorrent/tt-media-inference-server"
    return f"{_default_docker_repo}:{_default_docker_tag}"


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


def scale_llm_perf_targets(
    task: BenchmarkTaskParams, data_parallel: int
) -> BenchmarkTaskParams:
    """Scale throughput metrics in performance targets by data_parallel factor."""
    scaled_targets = {}
    for target_name, target in task.targets.items():
        scaled_targets[target_name] = PerformanceTarget(
            ttft_ms=target.ttft_ms,
            tput_user=target.tput_user,
            tput=target.tput * data_parallel if target.tput else None,
            tolerance=target.tolerance,
        )
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


def get_perf_reference(device_model_spec, perf_reference_map):
    # Migrated to vLLM API for data parallelism
    data_parallel = device_model_spec.vllm_args.get("data_parallel_size")

    if data_parallel:
        # need to adjust perf target device for data_parallel factor
        dp_device = device_model_spec.device.get_data_parallel_subdevice(data_parallel)
        perf_reference = perf_reference_map.get(dp_device, [])
        if perf_reference:
            perf_reference = [
                scale_llm_perf_targets(task, data_parallel) for task in perf_reference
            ]
    else:
        perf_reference = perf_reference_map.get(device_model_spec.device, [])
    return perf_reference


def model_weights_to_model_name(model_weights: str) -> str:
    return Path(model_weights).name


def get_model_id(impl_name: str, model_name: str, device: str) -> str:
    # Validate that all parameters are strings
    assert isinstance(impl_name, str), (
        f"Impl name must be a string, got {type(impl_name)}"
    )
    assert isinstance(model_name, str), (
        f"Model name must be a string, got {type(model_name)}"
    )
    assert isinstance(device, str), f"Device must be a string, got {type(device)}"

    # Validate that all parameters are non-empty
    assert impl_name.strip(), "Impl name cannot be empty or whitespace-only"
    assert model_name.strip(), "Model name cannot be empty or whitespace-only"
    assert device.strip(), "Device cannot be empty or whitespace-only"

    model_id = f"id_{impl_name}_{model_name}_{device}"
    return model_id


@dataclass(frozen=True)
class ImplSpec:
    impl_id: str
    impl_name: str
    repo_url: str
    code_path: str


tt_transformers_impl = ImplSpec(
    impl_id="tt_transformers",
    impl_name="tt-transformers",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/tt_transformers",
)
llama3_70b_galaxy_impl = ImplSpec(
    impl_id="llama3_70b_galaxy",
    impl_name="llama3-70b-galaxy",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/llama3_70b_galaxy",
)
qwen3_32b_galaxy_impl = ImplSpec(
    impl_id="qwen3_32b_galaxy",
    impl_name="qwen3-32b-galaxy",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/llama3_70b_galaxy",
)
gpt_oss_impl = ImplSpec(
    impl_id="gpt_oss",
    impl_name="gpt-oss",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/gpt_oss",
)
deepseek_r1_galaxy_impl = ImplSpec(
    impl_id="deepseek_r1_galaxy",
    impl_name="deepseek-r1-galaxy",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/deepseek_v3",
)
whisper_impl = ImplSpec(
    impl_id="whisper",
    impl_name="whisper",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/whisper",
)
speecht5_impl = ImplSpec(
    impl_id="speecht5_tts",
    impl_name="speecht5-tts",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/experimental/speecht5_tts",
)
forge_vllm_plugin_impl = ImplSpec(
    impl_id="forge_vllm_plugin",
    impl_name="forge-vllm-plugin",
    repo_url="https://github.com/tenstorrent/tt-xla/tree/main",
    code_path="integrations/vllm_plugin",
)
tt_vllm_plugin_impl = ImplSpec(
    impl_id="tt_vllm_plugin",
    impl_name="tt-vllm-plugin",
    repo_url="https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin",
    code_path="tt_vllm_plugin",
)

_IMPL_REGISTRY: Dict[str, ImplSpec] = {
    "tt_transformers": tt_transformers_impl,
    "llama3_70b_galaxy": llama3_70b_galaxy_impl,
    "qwen3_32b_galaxy": qwen3_32b_galaxy_impl,
    "gpt_oss": gpt_oss_impl,
    "deepseek_r1_galaxy": deepseek_r1_galaxy_impl,
    "whisper": whisper_impl,
    "speecht5_tts": speecht5_impl,
    "forge_vllm_plugin": forge_vllm_plugin_impl,
    "tt_vllm_plugin": tt_vllm_plugin_impl,
}


@dataclass(frozen=True)
class VersionRequirement:
    """Represents a software version requirement with a specific mode."""

    specifier: str
    mode: VersionMode


@dataclass(frozen=True)
class SystemRequirements:
    """Represents system software version requirements."""

    firmware: VersionRequirement = None
    kmd: VersionRequirement = None


@dataclass(frozen=True)
class KnownIssue:
    workflow_type: WorkflowType
    reason: str
    task_name: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.workflow_type, WorkflowType):
            coerced = WorkflowType.from_string(str(self.workflow_type))
            object.__setattr__(self, "workflow_type", coerced)

    def matches(self, workflow_type: WorkflowType, task_name: Optional[str]) -> bool:
        if self.workflow_type != workflow_type:
            return False
        if self.task_name is None:
            return True
        return task_name is not None and self.task_name == task_name


@dataclass(frozen=True)
class DeviceModelSpec:
    """
    Model-specific specification for a specific device.
    """

    device: DeviceTypes
    max_concurrency: int
    max_context: int
    perf_targets_map: Dict[str, float] = field(default_factory=dict)
    default_impl: bool = False
    perf_reference: List[BenchmarkTaskParams] = field(default_factory=list)
    vllm_args: Dict[str, str] = field(default_factory=dict)
    override_tt_config: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    tensor_cache_timeout: float = 3600.0
    system_requirements: Optional[SystemRequirements] = None
    known_issues: List[KnownIssue] = field(default_factory=list)
    # When set, run_evals appends max_retries=<N> to lm-eval --model_args.
    # Default 3 × exponential backoff = hours of burn on permanent 4xx.
    eval_max_retries: Optional[int] = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        """Validate that required specification is present."""
        pass

    def _infer_data(self):
        """Infer missing data fields from other specification values."""
        max_tokens_all_users = self.max_context
        max_concurrency = self.max_concurrency
        if data_parallel_size := self.vllm_args.get("data_parallel_size"):
            assert isinstance(data_parallel_size, int)
            # vllm args need to be set per engine instance, the number of which is
            # the data_parallel_size (# of DP ranks). The variables must be computed
            # and passed to client consumers however that will make requests to the
            # DP engines without needing to know about DP rank.
            max_concurrency = max_concurrency // data_parallel_size
            max_tokens_all_users = max_tokens_all_users * data_parallel_size
        object.__setattr__(self, "max_tokens_all_users", max_tokens_all_users)
        # TODO: we should get max_num_batched_tokens from DeviceModelSpec in the future
        default_vllm_args = {
            "block_size": "64",
            "max_model_len": str(self.max_context),
            "max_num_seqs": str(max_concurrency),
            "max_num_batched_tokens": str(self.max_context),
            "max-log-len": "32",
            "seed": "9472",
            "additional_config": json.dumps({"tt": self.override_tt_config}),
        }
        merged_vllm_args = {**default_vllm_args, **self.vllm_args}
        object.__setattr__(self, "vllm_args", merged_vllm_args)

        self._infer_env_vars()

    def find_known_issue(
        self, workflow_type: WorkflowType, task_name: Optional[str] = None
    ) -> Optional[KnownIssue]:
        for issue in self.known_issues:
            if issue.matches(workflow_type, task_name):
                return issue
        return None

    def _infer_env_vars(self):
        inferred_env_vars = {}
        if self.device in [DeviceTypes.N300, DeviceTypes.T3K, DeviceTypes.GALAXY_T3K]:
            inferred_env_vars["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

        inferred_env_vars["MESH_DEVICE"] = self.device.to_mesh_device_str()

        # TODO: Remove once all model specs are uplifted to tt-metal >= 0.60.0
        if self.device.is_wormhole():
            inferred_env_vars["ARCH_NAME"] = "wormhole_b0"
        elif self.device.is_blackhole():
            inferred_env_vars["ARCH_NAME"] = "blackhole"

        merged_env_vars = {
            **inferred_env_vars,
            **self.env_vars,
        }
        object.__setattr__(self, "env_vars", merged_env_vars)


@dataclass(frozen=True)
class ModelSpec:
    """
    Fully instantiated specification for a specific model on a specific device.
    This is what gets used throughout the system after template expansion.
    """

    # Core identity - required fields (NO DEFAULTS)
    model_id: str
    impl: ImplSpec
    hf_model_repo: str
    model_name: str
    inference_engine: InferenceEngine
    device_type: DeviceTypes  # Single device, not a set
    tt_metal_commit: str
    device_model_spec: DeviceModelSpec

    # Optional specification fields (WITH DEFAULTS)
    system_requirements: Optional[SystemRequirements] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    vllm_commit: Optional[str] = None
    hf_weights_repo: Optional[str] = (
        None  # HF repo to download weights from (defaults to hf_model_repo)
    )
    param_count: Optional[int] = None
    min_disk_gb: Optional[int] = None
    min_ram_gb: Optional[int] = None
    model_type: Optional[ModelType] = ModelType.LLM
    repacked: int = 0
    version: str = VERSION
    docker_image: Optional[str] = None
    status: str = ModelStatusTypes.EXPERIMENTAL
    code_link: Optional[str] = None
    override_tt_config: Dict[str, str] = field(default_factory=dict)
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])
    subdevice_type: Optional[DeviceTypes] = (
        None  # Used for data-parallel configurations
    )
    uses_tensor_model_cache: bool = True
    has_builtin_warmup: bool = False
    metadata: Dict = field(default_factory=dict)

    # DEPRECATED - only used by tt-media-server, kept for backwards compatibility
    cli_args: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Skipped for forge/media: Forge vLLM relies on torch.compile/dynamo for its compilation pipeline; TORCHDYNAMO_DISABLE=1 breaks warmup.
        if self.inference_engine in (
            InferenceEngine.FORGE.value,
            InferenceEngine.MEDIA.value,
        ):
            default_env_vars = {}
        else:
            default_env_vars = {
                "VLLM_CONFIGURE_LOGGING": "1",
                "VLLM_RPC_TIMEOUT": "900000",
                "VLLM_TARGET_DEVICE": "tt",
                "TORCHDYNAMO_DISABLE": "1",
            }
        # order of precedence: default, env_vars, device_model_spec
        merged_env_vars = {
            **default_env_vars,
            **self.env_vars,
            **self.device_model_spec.env_vars,
        }
        object.__setattr__(self, "env_vars", merged_env_vars)

        # order of precedence: default_vllm_args, device_model_spec.vllm_args
        default_vllm_args = {
            "model": self.hf_model_repo,
        }
        merged_vllm_args = {
            **default_vllm_args,
            **self.device_model_spec.vllm_args,
        }
        object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)

        self._validate_data()
        self._infer_data()

    def _infer_data(self):
        """Infer missing data fields from other specification values."""
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen

        # Default hf_weights_repo to hf_model_repo if not set
        if not self.hf_weights_repo:
            object.__setattr__(self, "hf_weights_repo", self.hf_model_repo)

        # Infer param count from model repo name
        if not self.param_count:
            object.__setattr__(
                self, "param_count", ModelSpec.infer_param_count(self.hf_model_repo)
            )

        # Calculate conservative disk and ram minimums based on param count
        if not self.min_disk_gb and self.param_count:
            MIN_DISK_GB_AFTER_DOWNLOAD = 20
            if self.repacked:
                # 2x for raw fp16 weights hf cache (may already be present)
                # 1x for repacked quantized copy
                # 1x for tt-metal cache
                # 1x for overhead
                object.__setattr__(
                    self,
                    "min_disk_gb",
                    self.param_count * 3 + MIN_DISK_GB_AFTER_DOWNLOAD,
                )
            else:
                # 2x for raw fp16 weights hf cache (may already be present)
                # 2x for copy
                object.__setattr__(
                    self,
                    "min_disk_gb",
                    self.param_count * 2 + MIN_DISK_GB_AFTER_DOWNLOAD,
                )

        if not self.min_ram_gb and self.param_count:
            # assume fp16 equivalent weights, add 0.5x overhead buffer
            object.__setattr__(self, "min_ram_gb", self.param_count * 2.5)

        # Generate default docker image if not provided
        if not self.docker_image:
            # TODO: Use ubuntu version to interpolate this string
            _default_docker_link = generate_default_docker_link(
                self.version,
                self.tt_metal_commit,
                self.vllm_commit,
                inference_engine=self.inference_engine,
                multihost=self.device_type.is_multihost(),
            )
            object.__setattr__(self, "docker_image", _default_docker_link)

        # Generate code link
        if not self.code_link:
            object.__setattr__(
                self,
                "code_link",
                f"{self.impl.repo_url}/tree/{self.tt_metal_commit}/{self.impl.code_path}",
            )

        data_parallel = self.device_model_spec.vllm_args.get("data_parallel_size")
        if data_parallel:
            object.__setattr__(
                self,
                "subdevice_type",
                self.device_type.get_data_parallel_subdevice(data_parallel),
            )

    def _validate_data(self):
        """Validate that required specification is present."""
        assert self.hf_model_repo, "hf_model_repo must be set"
        assert self.model_name, "model_name must be set"
        assert self.model_id, "model_id must be set"
        assert self.inference_engine in [e.value for e in InferenceEngine], (
            f"inference_engine must be one of {[e.value for e in InferenceEngine]}"
        )

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
        matches = re.findall(r"(\d+(?:\.\d+)?)[Bb]", hf_model_repo)
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

    def get_serialized_dict(self) -> str:
        """
        Get the serialized representation of this model specification.
        """

        def serialize_value(obj):
            """Recursively serialize complex objects for JSON export."""
            # Handle enums first (they have __dict__ but aren't dataclasses)
            if hasattr(obj, "name") and hasattr(obj, "value"):  # Enum
                return obj.name
            elif isinstance(obj, ModelType):  # Explicit ModelType handling
                return obj.name
            elif hasattr(obj, "__dict__") and hasattr(obj, "__dataclass_fields__"):
                # Handle dataclasses by converting to dict
                result = asdict(obj)
                return {k: serialize_value(v) for k, v in result.items()}
            elif isinstance(obj, dict):
                return {k: serialize_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_value(item) for item in obj]
            else:
                return obj

        # Serialize the model spec
        spec_dict = serialize_value(self)
        return spec_dict

    def to_json(self, run_id: str, output_dir: str = ".") -> Path:
        """
        Export this model specification to a JSON file.

        Args:
            output_dir: Directory to write the JSON file (defaults to current directory)
            filename: Custom filename (defaults to "tt_model_spec_{model_id}.json")

        Returns:
            The path to the created JSON file
        """
        spec_dict = self.get_serialized_dict()

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"tt_model_spec_{run_id}.json"
        filepath = output_path / filename

        # Write JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)

        return filepath

    @classmethod
    def from_json(cls, json_fpath: str) -> "ModelSpec":
        """
        Create a ModelSpec instance from a JSON file.

        Args:
            json_fpath: Path to the JSON file to read

        Returns:
            ModelSpec instance created from the JSON data

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON data is invalid or missing required fields
        """
        json_path = Path(json_fpath)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_fpath}")

        # Read JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle combined format: {"runtime_model_spec": …, "runtime_config": …}
        if "runtime_model_spec" in data and "runtime_config" in data:
            data = data["runtime_model_spec"]

        def deserialize_enum(enum_class, value):
            """Helper to deserialize enum values."""
            if isinstance(value, str):
                # Try to get enum by name first, then by value
                try:
                    return getattr(enum_class, value)
                except AttributeError:
                    # Try by value if name lookup fails
                    for enum_member in enum_class:
                        if enum_member.value == value:
                            return enum_member
                    raise ValueError(f"Invalid {enum_class.__name__} value: {value}")
            return value

        def deserialize_dataclass_field(field_type, value):
            """Helper to deserialize nested dataclass fields."""
            if value is None:
                return None

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Extract the non-None type from Optional
                args = getattr(field_type, "__args__", ())
                if len(args) == 2 and type(None) in args:
                    field_type = args[0] if args[1] is type(None) else args[1]

            # Handle specific dataclass types
            if field_type == ImplSpec and isinstance(value, dict):
                return ImplSpec(**value)
            elif field_type == DeviceModelSpec and isinstance(value, dict):
                # Handle nested BenchmarkTaskParams in perf_reference
                perf_reference = value.get("perf_reference", [])
                if perf_reference:
                    deserialized_perf_ref = []
                    for task_data in perf_reference:
                        if isinstance(task_data, dict):
                            # Handle PerformanceTarget objects in targets
                            targets = task_data.get("targets", {})
                            deserialized_targets = {}
                            for target_name, target_data in targets.items():
                                if isinstance(target_data, dict):
                                    deserialized_targets[target_name] = (
                                        PerformanceTarget(**target_data)
                                    )
                                else:
                                    deserialized_targets[target_name] = target_data
                            task_data["targets"] = deserialized_targets
                            deserialized_perf_ref.append(
                                BenchmarkTaskParams(**task_data)
                            )
                        else:
                            deserialized_perf_ref.append(task_data)
                    value["perf_reference"] = deserialized_perf_ref
                known_issues = value.get("known_issues", [])
                if known_issues:
                    value["known_issues"] = [
                        KnownIssue(**ki) if isinstance(ki, dict) else ki
                        for ki in known_issues
                    ]
                return DeviceModelSpec(**value)
            elif field_type == SystemRequirements and isinstance(value, dict):
                for requirement_name, requirement_spec in value.items():
                    # not all system requirements are always defined
                    if requirement_spec is None:
                        continue
                    requirement_spec["mode"] = deserialize_enum(
                        VersionMode, requirement_spec["mode"]
                    )
                    version_requirement = VersionRequirement(**requirement_spec)
                    value[requirement_name] = version_requirement
                return SystemRequirements(**value)
            elif field_type == DeviceTypes:
                return deserialize_enum(DeviceTypes, value)
            elif field_type == ModelStatusTypes:
                return deserialize_enum(ModelStatusTypes, value)

            return value

        # Handle enum fields
        if "device_type" in data:
            data["device_type"] = deserialize_enum(DeviceTypes, data["device_type"])
        if "subdevice_type" in data and data["subdevice_type"] is not None:
            data["subdevice_type"] = deserialize_enum(
                DeviceTypes, data["subdevice_type"]
            )
        if "status" in data:
            data["status"] = deserialize_enum(ModelStatusTypes, data["status"])
        if "model_type" in data and data["model_type"] is not None:
            data["model_type"] = deserialize_enum(ModelType, data["model_type"])
        if "device_model_spec" in data:
            data["device_model_spec"]["device"] = deserialize_enum(
                DeviceTypes, data["device_model_spec"]["device"]
            )

        # Handle nested dataclass fields
        if "impl" in data:
            data["impl"] = deserialize_dataclass_field(ImplSpec, data["impl"])
        if "device_model_spec" in data:
            data["device_model_spec"] = deserialize_dataclass_field(
                DeviceModelSpec, data["device_model_spec"]
            )
        if "system_requirements" in data:
            system_requirements = deserialize_dataclass_field(
                SystemRequirements, data["system_requirements"]
            )
            if system_requirements is not None:
                data["system_requirements"] = system_requirements

        # Create and return the ModelSpec instance
        return cls(**data)

    def apply_overrides(self, runtime_config: RuntimeConfig):
        """Apply spec-mutating runtime overrides from RuntimeConfig.

        Only modifies fields that are part of the model specification itself
        (vllm_args, override_tt_config, docker_image, commits).  Orchestration
        and workflow state belong in RuntimeConfig, not here.
        """
        if runtime_config.override_tt_config:
            override_config_from_cli = json.loads(runtime_config.override_tt_config)

            merged_override_config = dict(self.device_model_spec.override_tt_config)

            for key, value in override_config_from_cli.items():
                if value is None:
                    merged_override_config.pop(key, None)
                else:
                    merged_override_config[key] = value

            object.__setattr__(
                self.device_model_spec,
                "override_tt_config",
                merged_override_config,
            )
            merged_vllm_args = {
                **self.device_model_spec.vllm_args,
                "additional_config": json.dumps({"tt": merged_override_config}),
            }
            object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)

        if runtime_config.vllm_override_args:
            vllm_override_args_from_cli = json.loads(runtime_config.vllm_override_args)

            merged_vllm_args = dict(self.device_model_spec.vllm_args)

            for key, value in vllm_override_args_from_cli.items():
                if value is None:
                    merged_vllm_args.pop(key, None)
                else:
                    merged_vllm_args[key] = value

            object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)

            # Mirror overridden vllm_args into env_vars so forge/media containers,
            # which read bare env vars (not vllm CLI args), pick up the override.
            VLLM_ARG_TO_ENV = {
                "max_num_seqs": "MAX_NUM_SEQS",
                "max_model_len": "MAX_MODEL_LENGTH",
            }
            overridden_env = {
                env_key: str(vllm_override_args_from_cli[vllm_key])
                for vllm_key, env_key in VLLM_ARG_TO_ENV.items()
                if vllm_override_args_from_cli.get(vllm_key) is not None
            }
            if overridden_env:
                object.__setattr__(
                    self, "env_vars", {**self.env_vars, **overridden_env}
                )

        if runtime_config.service_port:
            merged_vllm_args = {
                **self.device_model_spec.vllm_args,
                "port": runtime_config.service_port,
            }
            object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)

        if runtime_config.override_docker_image:
            object.__setattr__(
                self, "docker_image", runtime_config.override_docker_image
            )
            tt_metal_commit, vllm_commit = parse_commits_from_docker_image(
                runtime_config.override_docker_image
            )
            object.__setattr__(self, "tt_metal_commit", tt_metal_commit)
            object.__setattr__(self, "vllm_commit", vllm_commit)
            # Re-parse `version` from the override tag so the pre-0.11
            # support check (validate_runtime_args) sees the actual image
            # being run, not the template default. Unparseable override
            # tags (`:dev`, `:latest`) leave version untouched.
            parsed_version = parse_image_version(runtime_config.override_docker_image)
            if parsed_version is not None:
                object.__setattr__(
                    self, "version", ".".join(str(p) for p in parsed_version)
                )


@dataclass(frozen=True)
class ModelSpecTemplate:
    """
    Template specification that gets expanded into individual ModelSpec instances
    for each weight variant and device combination. This represents the shared specification
    across multiple models and devices.
    """

    # Required fields (NO DEFAULTS) - must come first
    weights: List[str]  # List of HF model repos to create specs for
    impl: ImplSpec
    tt_metal_commit: str
    inference_engine: InferenceEngine
    device_model_specs: List[DeviceModelSpec]

    # Optional template fields (WITH DEFAULTS) - must come after required fields
    system_requirements: Optional[SystemRequirements] = None
    vllm_commit: Optional[str] = None
    status: str = ModelStatusTypes.EXPERIMENTAL
    env_vars: Dict[str, str] = field(default_factory=dict)
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])
    repacked: int = 0
    version: str = VERSION
    perf_targets_map: Dict[str, float] = field(default_factory=dict)
    docker_image: Optional[str] = None
    model_type: Optional[ModelType] = ModelType.LLM
    min_disk_gb: Optional[int] = None
    min_ram_gb: Optional[int] = None
    uses_tensor_model_cache: bool = True
    hf_weights_repo: Optional[str] = (
        None  # HF repo to download weights from (shared across all weights)
    )
    has_builtin_warmup: bool = False
    metadata: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        self._validate_data()
        self._infer_data()

    def _validate_data(self):
        """Validate that required specification is present."""
        assert self.device_model_specs, "device_model_specs must be provided"
        assert self.weights, "weights must be provided"
        assert self.inference_engine in [engine.value for engine in InferenceEngine], (
            f"inference_engine must be a valid InferenceEngine! \
            Available: {[engine for engine in InferenceEngine]}"
        )
        # check that metadata keys are a subset of weights
        if self.metadata:
            invalid_keys = set(self.metadata.keys()) - set(self.weights)
            assert not invalid_keys, (
                f"These keys do not exist as weights: {invalid_keys}, valid weights: {self.weights}"
            )

    def _infer_data(self):
        """Infer missing data fields from other specification values."""
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

    def expand_to_specs(self) -> List["ModelSpec"]:
        """Expand this template into individual ModelSpec instances."""
        specs = []

        # Generate performance reference map
        main_model_name = model_weights_to_model_name(self.weights[0])
        perf_reference_map = get_perf_reference_map(
            main_model_name, self.perf_targets_map
        )

        for weight in self.weights:
            for device_model_spec in self.device_model_specs:
                device_type = device_model_spec.device
                model_name = Path(weight).name
                model_id = get_model_id(
                    self.impl.impl_name, model_name, device_type.name.lower()
                )

                # Perf reference for this device accounting for impl features
                # e.g. data parallelism factor
                perf_reference = get_perf_reference(
                    device_model_spec, perf_reference_map
                )

                # Create a new device_model_spec with performance reference data
                device_model_spec_with_perf = DeviceModelSpec(
                    device=device_model_spec.device,
                    max_concurrency=device_model_spec.max_concurrency,
                    max_context=device_model_spec.max_context,
                    perf_targets_map=device_model_spec.perf_targets_map,
                    default_impl=device_model_spec.default_impl,
                    perf_reference=perf_reference,
                    vllm_args=device_model_spec.vllm_args,
                    override_tt_config=device_model_spec.override_tt_config,
                    env_vars=device_model_spec.env_vars,
                    tensor_cache_timeout=device_model_spec.tensor_cache_timeout,
                    system_requirements=device_model_spec.system_requirements,
                    known_issues=device_model_spec.known_issues,
                    eval_max_retries=device_model_spec.eval_max_retries,
                )
                spec = ModelSpec(
                    # Core identity
                    device_type=device_type,
                    impl=self.impl,
                    hf_model_repo=weight,
                    model_id=model_id,
                    model_name=model_name,
                    inference_engine=self.inference_engine,
                    device_model_spec=device_model_spec_with_perf,
                    # Version control
                    system_requirements=device_model_spec.system_requirements
                    if device_model_spec.system_requirements
                    else self.system_requirements,
                    tt_metal_commit=self.tt_metal_commit,
                    vllm_commit=self.vllm_commit,
                    hf_weights_repo=self.hf_weights_repo,
                    # Template fields
                    env_vars=self.env_vars,
                    repacked=self.repacked,
                    version=self.version,
                    docker_image=self.docker_image,
                    status=self.status,
                    override_tt_config=device_model_spec.override_tt_config,
                    supported_modalities=self.supported_modalities,
                    min_disk_gb=self.min_disk_gb,
                    min_ram_gb=self.min_ram_gb,
                    model_type=self.model_type,
                    uses_tensor_model_cache=self.uses_tensor_model_cache,
                    has_builtin_warmup=self.has_builtin_warmup,
                    metadata=self.metadata.get(weight, {}),
                )

                specs.append(spec)
        return specs


# Catalog data lives in workflows/model_specs/catalog.yaml.
# spec_templates below loads from that file at import time.


def _build_system_requirements(data: Optional[Dict]) -> Optional["SystemRequirements"]:
    if data is None:
        return None
    kwargs: Dict = {}
    for key in ("firmware", "kmd"):
        if data.get(key) is not None:
            kwargs[key] = VersionRequirement(
                specifier=data[key]["specifier"],
                mode=VersionMode[data[key]["mode"]],
            )
    return SystemRequirements(**kwargs)


def _build_device_model_spec(data: Dict) -> "DeviceModelSpec":
    kwargs = dict(data)
    kwargs["device"] = DeviceTypes.from_string(kwargs["device"])
    if "system_requirements" in kwargs:
        kwargs["system_requirements"] = _build_system_requirements(
            kwargs["system_requirements"]
        )
    if "known_issues" in kwargs:
        kwargs["known_issues"] = [
            KnownIssue(
                workflow_type=WorkflowType.from_string(ki["workflow_type"]),
                reason=ki["reason"],
                task_name=ki.get("task_name"),
            )
            for ki in kwargs["known_issues"]
        ]
    return DeviceModelSpec(**kwargs)


def _build_template(data: Dict) -> "ModelSpecTemplate":
    kwargs = dict(data)
    impl_id = kwargs["impl"]
    if impl_id not in _IMPL_REGISTRY:
        raise ValueError(
            f"Unknown impl '{impl_id}'. Known impls: {sorted(_IMPL_REGISTRY)}"
        )
    kwargs["impl"] = _IMPL_REGISTRY[impl_id]
    kwargs["inference_engine"] = InferenceEngine[kwargs["inference_engine"]].value
    kwargs["device_model_specs"] = [
        _build_device_model_spec(d) for d in kwargs["device_model_specs"]
    ]
    if "system_requirements" in kwargs:
        kwargs["system_requirements"] = _build_system_requirements(
            kwargs["system_requirements"]
        )
    if "model_type" in kwargs and kwargs["model_type"] is not None:
        kwargs["model_type"] = ModelType[kwargs["model_type"]]
    if "status" in kwargs:
        kwargs["status"] = ModelStatusTypes[kwargs["status"]]
    return ModelSpecTemplate(**kwargs)


def load_templates_from_yaml(path: Path) -> List["ModelSpecTemplate"]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "templates" not in data:
        raise ValueError(f"YAML file {path} is empty or missing 'templates' key")
    return [_build_template(t) for t in data["templates"]]


_MODEL_SPECS_DIR = get_repo_root_path() / "workflows" / "model_specs"

# Catalog environments live in sibling directories under _MODEL_SPECS_DIR.
# Set MODEL_SPECS_ENV=dev to load the dev set instead of prod.
_VALID_MODEL_SPECS_ENVS = ("prod", "dev")
_MODEL_SPECS_ENV = os.getenv("MODEL_SPECS_ENV", "prod")
if _MODEL_SPECS_ENV not in _VALID_MODEL_SPECS_ENVS:
    raise ValueError(
        f"MODEL_SPECS_ENV must be one of {_VALID_MODEL_SPECS_ENVS}, "
        f"got {_MODEL_SPECS_ENV!r}"
    )

# One catalog file per model category. Load order determines spec_templates
# order, which in turn determines MODEL_SPECS dict insertion order.
_CATALOG_FILES = (
    "llm.yaml",
    "vlm.yaml",
    "video.yaml",
    "image.yaml",
    "audio_tts.yaml",
    "embedding.yaml",
    "cnn.yaml",
)

spec_templates: List["ModelSpecTemplate"] = [
    template
    for fname in _CATALOG_FILES
    for template in load_templates_from_yaml(
        _MODEL_SPECS_DIR / _MODEL_SPECS_ENV / fname
    )
]


def get_model_spec_map(
    templates: List[ModelSpecTemplate],
) -> Dict[str, ModelSpec]:
    """
    Generate final model specifications from templates.

    Args:
        templates: List of ModelSpecTemplate instances to expand

    Returns:
        Dictionary mapping model_id to ModelSpec instances
    """
    model_spec_map = {}
    for template in templates:
        for spec in template.expand_to_specs():
            model_spec_map[spec.model_id] = spec
    return model_spec_map


def export_model_specs_json(model_specs: dict, output_path: Path) -> int:
    """Export MODEL_SPECS to a nested JSON file.

    Output is wrapped with metadata and nested model specs:
    schema_version, release_version, model_specs[hf_model_repo][device_type]
    [inference_engine][impl_id].

    Args:
        model_specs: Dictionary mapping model_id to ModelSpec objects.
        output_path: Path where the JSON file should be written.

    Returns:
        Number of model specs exported.
    """
    nested_specs = {}
    num_specs = 0
    for model_id, model_spec in model_specs.items():
        hf_repo = model_spec.hf_model_repo
        device = model_spec.device_type.to_string()
        engine = model_spec.inference_engine
        impl_id = model_spec.impl.impl_id

        nested_specs.setdefault(hf_repo, {})
        nested_specs[hf_repo].setdefault(device, {})
        nested_specs[hf_repo][device].setdefault(engine, {})
        nested_specs[hf_repo][device][engine][impl_id] = (
            model_spec.get_serialized_dict()
        )
        num_specs += 1

    export_data = {
        "schema_version": MODEL_SPECS_SCHEMA_VERSION,
        "release_version": VERSION,
        "model_specs": nested_specs,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    return num_specs


# Final model specifications generated from templates
MODEL_SPECS = get_model_spec_map(spec_templates)


# Cache of additional catalog environments loaded on demand. Avoids paying the
# YAML-load cost up front for environments that may never be used in a given run.
_MODEL_SPECS_BY_ENV: Dict[str, Dict[str, ModelSpec]] = {_MODEL_SPECS_ENV: MODEL_SPECS}


def _load_model_specs_for_env(env: str) -> Dict[str, ModelSpec]:
    """Return the model_id->ModelSpec dict for a given catalog env, loading
    YAMLs from workflows/model_specs/<env>/ on first access."""
    if env not in _VALID_MODEL_SPECS_ENVS:
        raise ValueError(
            f"Unknown catalog env {env!r}; must be one of {_VALID_MODEL_SPECS_ENVS}"
        )
    cached = _MODEL_SPECS_BY_ENV.get(env)
    if cached is not None:
        return cached
    templates = [
        template
        for fname in _CATALOG_FILES
        for template in load_templates_from_yaml(_MODEL_SPECS_DIR / env / fname)
    ]
    specs_map = get_model_spec_map(templates)
    _MODEL_SPECS_BY_ENV[env] = specs_map
    return specs_map


def get_runtime_model_spec(
    model: str,
    device: str,
    engine: Optional[str] = None,
    impl: Optional[str] = None,
    env: Optional[str] = None,
) -> Tuple[ModelSpec, str, str]:
    """Select a ModelSpec from the catalog for the given env.

    When *env* is None or matches the import-time MODEL_SPECS_ENV, this reads
    from the already-loaded MODEL_SPECS dict. When *env* names a different
    catalog environment (e.g. "dev" when the host default is "prod"), the
    YAMLs under workflows/model_specs/<env>/ are loaded lazily on first call.

    Pure function -- does **not** mutate any external state.

    Returns ``(model_spec, resolved_impl, resolved_engine)`` so the caller
    can construct a fully-initialised RuntimeConfig in one step.
    """
    device_type = DeviceTypes.from_string(device)

    specs_map = (
        MODEL_SPECS
        if env is None or env == _MODEL_SPECS_ENV
        else _load_model_specs_for_env(env)
    )
    env_label = env or _MODEL_SPECS_ENV

    candidate_specs = [
        spec
        for spec in specs_map.values()
        if spec.model_name == model
        and spec.device_type == device_type
        and (not engine or spec.inference_engine == engine)
        and (not impl or spec.impl.impl_name == impl)
    ]

    if not candidate_specs:
        engine_msg = f", engine={engine}" if engine else ""
        impl_msg = f", impl={impl}" if impl else ""
        raise ValueError(
            f"Model:={model} does not support device:={device}{engine_msg}{impl_msg} "
            f"in the {env_label!r} catalog"
        )

    default_spec = next(
        (spec for spec in candidate_specs if spec.device_model_spec.default_impl),
        None,
    )
    selected_spec = default_spec or (candidate_specs[0] if (impl or engine) else None)

    if selected_spec is None:
        raise ValueError(
            f"Model:={model} does not have a default impl for "
            f"device:={device}, engine:={engine} in the {env_label!r} catalog; "
            f"you must pass --impl or --engine"
        )

    resolved_impl = selected_spec.impl.impl_name
    resolved_engine = engine if engine else selected_spec.inference_engine

    model_spec = specs_map[selected_spec.model_id]
    return model_spec, resolved_impl, resolved_engine
