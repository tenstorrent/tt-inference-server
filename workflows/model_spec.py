# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import IntEnum, auto
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict, make_dataclass
from typing import Dict, List, Optional, Union

from workflows.utils import (
    get_version,
    BenchmarkTaskParams,
    PerformanceTarget,
    get_repo_root_path,
)
from workflows.workflow_types import DeviceTypes, ModelStatusTypes, VersionMode

VERSION = get_version()


def generate_docker_tag(version: str, tt_metal_commit: str, vllm_commit: str) -> str:
    max_tag_len = 12
    if vllm_commit:
        return f"{version}-{tt_metal_commit[:max_tag_len]}-{vllm_commit[:max_tag_len]}"
    else:
        return f"{version}-{tt_metal_commit[:max_tag_len]}"


def generate_default_docker_link(
    version: str, tt_metal_commit: str, vllm_commit: str
) -> str:
    _default_docker_tag = generate_docker_tag(version, tt_metal_commit, vllm_commit)
    _default_docker_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64"
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


def model_weights_to_model_name(model_weights: str) -> str:
    return Path(model_weights).name


def get_model_id(impl_name: str, model_name: str, device: str) -> str:
    # Validate that all parameters are strings
    assert isinstance(
        impl_name, str
    ), f"Impl name must be a string, got {type(impl_name)}"
    assert isinstance(
        model_name, str
    ), f"Model name must be a string, got {type(model_name)}"
    assert isinstance(device, str), f"Device must be a string, got {type(device)}"

    # Validate that all parameters are non-empty
    assert impl_name.strip(), "Impl name cannot be empty or whitespace-only"
    assert model_name.strip(), "Model name cannot be empty or whitespace-only"
    assert device.strip(), "Device cannot be empty or whitespace-only"

    model_id = f"id_{impl_name}_{model_name}_{device}"
    return model_id


class ModelType(IntEnum):
    LLM = auto()
    CNN = auto()


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
llama3_impl = ImplSpec(
    impl_id="llama3",
    impl_name="llama3",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/llama3",
)
t3000_llama2_70b_impl = ImplSpec(
    impl_id="llama2_70b",
    impl_name="llama2-70b",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/t3000/llama2_70b",
)
llama3_70b_galaxy_impl = ImplSpec(
    impl_id="llama3_70b_galaxy",
    impl_name="llama3-70b-galaxy",
    repo_url="https://github.com/tenstorrent/tt-metal",
    code_path="models/demos/llama3_70b_galaxy",
)


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

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        """Validate that required specification is present."""
        pass

    def _infer_data(self):
        """Infer missing data fields from other specification values."""
        default_vllm_args = {
            "block_size": "64",
            "max_model_len": str(self.max_context),
            "max_num_seqs": str(self.max_concurrency),
            "max_num_batched_tokens": str(self.max_context),
            "num_scheduler_steps": "10",
            "max-log-len": "32",
            "override_tt_config": json.dumps(self.override_tt_config),
        }
        merged_vllm_args = {**default_vllm_args, **self.vllm_args}
        object.__setattr__(self, "vllm_args", merged_vllm_args)

        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen
        # Set default concurrency and context if not provided
        if not self.max_concurrency:
            _default_max_concurrent = 32
            object.__setattr__(self, "max_concurrency", _default_max_concurrent)

        if not self.max_context:
            _default_max_context = 128 * 1024
            object.__setattr__(self, "max_context", _default_max_context)

        self._infer_env_vars()

    def _infer_env_vars(self):
        inferred_env_vars = {}
        if self.device in [DeviceTypes.N300, DeviceTypes.T3K]:
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
    device_type: DeviceTypes  # Single device, not a set
    tt_metal_commit: str
    device_model_spec: DeviceModelSpec

    # Optional specification fields (WITH DEFAULTS)
    system_requirements: Optional[SystemRequirements] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    vllm_commit: Optional[str] = None
    custom_inference_server: Optional[str] = None
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
    cli_args: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        default_env_vars = {
            "VLLM_CONFIGURE_LOGGING": "1",
            "VLLM_RPC_TIMEOUT": "900000",
            "VLLM_TARGET_DEVICE": "tt",
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

        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        """Infer missing data fields from other specification values."""
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen

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
            object.__setattr__(self, "min_ram_gb", self.param_count * 4)

        # Generate default docker image if not provided
        if not self.docker_image:
            # Note: default to release image, use --dev-mode at runtime to use dev images
            # TODO: Use ubuntu version to interpolate this string
            _default_docker_link = generate_default_docker_link(
                VERSION, self.tt_metal_commit, self.vllm_commit
            )
            object.__setattr__(self, "docker_image", _default_docker_link)

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
        """Validate that required specification is present."""
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

    def apply_runtime_args(self, args):
        # handle runtime model spec overrides
        def is_mutable(obj):
            return isinstance(obj, (list, dict, set))

        def _build_field(key, value):
            typ = type(value)
            if is_mutable(value):
                return (key, typ, field(default_factory=lambda v=value: v.copy()))
            else:
                return (key, typ, value)

        fields = [_build_field(key, value) for key, value in args.__dict__.items()]
        CliArgsClass = make_dataclass("cli_args", fields)
        cli_args = CliArgsClass(**args.__dict__)
        object.__setattr__(self, "cli_args", cli_args)

        if args.override_tt_config:
            object.__setattr__(
                self.device_model_spec,
                "override_tt_config",
                json.loads(args.override_tt_config),
            )
            # Update vllm_args to include the new override_tt_config
            merged_vllm_args = {
                **self.device_model_spec.vllm_args,
                "override_tt_config": args.override_tt_config,
            }
            object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)
        if args.vllm_override_args:
            # Get existing vllm_override_args and merge with new values
            vllm_override_args = json.loads(args.vllm_override_args)
            merged_vllm_args = {
                **self.device_model_spec.vllm_args,
                **vllm_override_args,
            }
            object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)

        if args.service_port:
            # Add service port to vllm_args
            merged_vllm_args = {
                **self.device_model_spec.vllm_args,
                **{"port": args.service_port},
            }
            object.__setattr__(self.device_model_spec, "vllm_args", merged_vllm_args)

        if args.dev_mode:
            object.__setattr__(
                self, "docker_image", self.docker_image.replace("-release-", "-dev-")
            )

        if args.override_docker_image:
            object.__setattr__(self, "docker_image", args.override_docker_image)


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
    custom_inference_server: Optional[str] = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        """Validate that required specification is present."""
        assert self.device_model_specs, "device_model_specs must be provided"
        assert self.weights, "weights must be provided"

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

                # Create a new device_model_spec with performance reference data
                device_model_spec_with_perf = DeviceModelSpec(
                    device=device_model_spec.device,
                    max_concurrency=device_model_spec.max_concurrency,
                    max_context=device_model_spec.max_context,
                    perf_targets_map=device_model_spec.perf_targets_map,
                    default_impl=device_model_spec.default_impl,
                    perf_reference=perf_reference_map.get(device_type, []),
                    vllm_args=device_model_spec.vllm_args,
                    override_tt_config=device_model_spec.override_tt_config,
                    env_vars=device_model_spec.env_vars,
                )

                spec = ModelSpec(
                    # Core identity
                    device_type=device_type,
                    impl=self.impl,
                    hf_model_repo=weight,
                    model_id=model_id,
                    model_name=model_name,
                    device_model_spec=device_model_spec_with_perf,
                    # Version control
                    system_requirements=self.system_requirements,
                    tt_metal_commit=self.tt_metal_commit,
                    vllm_commit=self.vllm_commit,
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
                    custom_inference_server=self.custom_inference_server,
                )
                specs.append(spec)
        return specs


# Model specification templates - these get expanded into individual specs
spec_templates = [
    ModelSpecTemplate(
        weights=[
            "google/gemma-3-1b-it",
        ],
        impl=tt_transformers_impl,
        tt_metal_commit="dc85f59",
        vllm_commit="87fe4a4",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
                override_tt_config={
                    "l1_small_size": 768,
                    "fabric_config": "FABRIC_1D",
                },
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
    ),
    ModelSpecTemplate(
        weights=[
            "google/gemma-3-4b-it",
            "google/medgemma-4b-it",
        ],
        impl=tt_transformers_impl,
        tt_metal_commit="dc85f59",
        vllm_commit="87fe4a4",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                vllm_args={
                    "mm-processor-kwargs": json.dumps({
                        "use_fast": True,
                        "do_convert_rgb": True,
                        "do_pan_and_scan": True,
                    }),
                },
                override_tt_config={
                    "l1_small_size": 768,
                    "fabric_config": "FABRIC_1D",
                },
            ),
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                vllm_args={
                    "mm-processor-kwargs": json.dumps({
                        "use_fast": True,
                        "do_convert_rgb": True,
                        "do_pan_and_scan": True,
                    }),
                },
                override_tt_config={
                    "l1_small_size": 768,
                    "fabric_config": "FABRIC_1D",
                },
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
        supported_modalities=["text", "image"],
    ),
    ModelSpecTemplate(
        weights=[
            "google/gemma-3-27b-it",
            "google/medgemma-27b-it",
        ],
        impl=tt_transformers_impl,
        tt_metal_commit="dc85f59",
        vllm_commit="87fe4a4",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                vllm_args={
                    "mm-processor-kwargs": json.dumps({
                        "use_fast": True,
                        "do_convert_rgb": True,
                        "do_pan_and_scan": True,
                    }),
                },
                override_tt_config={
                    "l1_small_size": 768,
                    "fabric_config": "FABRIC_1D",
                },
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
        supported_modalities=["text", "image"],
    ),
    ModelSpecTemplate(
        weights=["Qwen/Qwen3-8B"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.61.1-rc1",
        vllm_commit="5cbc982",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=40960,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=40960,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=40960,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
    ),
    ModelSpecTemplate(
        weights=["Qwen/Qwen3-32B"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.59.0-rc39",
        vllm_commit="3accc8d",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            )
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
        env_vars={
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
        },
    ),
    ModelSpecTemplate(
        weights=["mistralai/Mistral-7B-Instruct-v0.3"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.62.0-rc11",
        vllm_commit="bd7dd31",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=32 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.COMPLETE,
        env_vars={
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
        },
    ),
    ModelSpecTemplate(
        weights=["Qwen/QwQ-32B"],
        impl=tt_transformers_impl,
        tt_metal_commit="834686671ea3",
        vllm_commit="44f8562",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
        env_vars={
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
        },
    ),
    ModelSpecTemplate(
        weights=["Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="834686671ea3",
        vllm_commit="44f8562",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                override_tt_config={
                    "trace_region_size": 26000000,
                },
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
        env_vars={
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            "MAX_PREFILL_CHUNK_SIZE": "16",
        },
    ),
    ModelSpecTemplate(
        weights=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.62.0-rc10",
        vllm_commit="c348d08",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.N150X4,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
        env_vars={
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
        },
    ),
    ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        impl=llama3_70b_galaxy_impl,
        tt_metal_commit="e70e16f",
        vllm_commit="4948b77",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.GALAXY,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                vllm_args={
                    "num_scheduler_steps": 1,
                },
                override_tt_config={
                    "dispatch_core_axis": "col",
                    "sample_on_device_mode": "all",
                    "fabric_config": "FABRIC_1D_RING",
                    "worker_l1_size": 1345000,
                    "trace_region_size": 192441344,
                },
            ),
        ],
        system_requirements=SystemRequirements(
            firmware=VersionRequirement(
                specifier=">=18.6.0",
                mode=VersionMode.STRICT,
            ),
            kmd=VersionRequirement(
                specifier=">=2.1.0",
                mode=VersionMode.STRICT,
            ),
        ),
        status=ModelStatusTypes.TOP_PERF,
    ),
    ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        impl=tt_transformers_impl,
        system_requirements=SystemRequirements(
            firmware=VersionRequirement(
                specifier=">=18.2.0,<=18.5.0",
                mode=VersionMode.STRICT,
            ),
            kmd=VersionRequirement(
                specifier=">=2.0.0,<=2.3.0",
                mode=VersionMode.STRICT,
            ),
        ),
        tt_metal_commit="v0.59.0-rc14",
        vllm_commit="a869e5d",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                env_vars={
                    "MAX_PREFILL_CHUNK_SIZE": "32",
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                },
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
    ),
    ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ],
        impl=tt_transformers_impl,
        system_requirements=SystemRequirements(
            firmware=VersionRequirement(
                specifier=">=18.5.0",
                mode=VersionMode.STRICT,
            ),
            kmd=VersionRequirement(
                specifier=">=2.3.0",
                mode=VersionMode.STRICT,
            ),
        ),
        tt_metal_commit="v0.59.0-rc51",
        vllm_commit="b35fe70",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.P150X4,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
    ),
    ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
        ],
        impl=t3000_llama2_70b_impl,
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=False,
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
        repacked=1,
        env_vars={
            "MAX_PREFILL_CHUNK_SIZE": "32",
        },
    ),
    ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.2-11B-Vision",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.61.1-rc1",
        vllm_commit="5cbc982",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=16,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=16,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
        supported_modalities=["text", "image"],
    ),
    ModelSpecTemplate(
        weights=[
            "meta-llama/Llama-3.2-90B-Vision",
            "meta-llama/Llama-3.2-90B-Vision-Instruct",
        ],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.61.1-rc1",
        vllm_commit="5cbc982",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
                env_vars={
                    "MAX_PREFILL_CHUNK_SIZE": 16,
                },
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
        supported_modalities=["text", "image"],
    ),
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.62.0-rc15",
        vllm_commit="b4b88c3",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        env_vars={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1},
        status=ModelStatusTypes.FUNCTIONAL,
    ),
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.57.0-rc71",
        vllm_commit="2a8debd",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
    ),
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.62.0-rc11",
        vllm_commit="bd7dd31",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.N300,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.GPU,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=False,
            ),
        ],
        status=ModelStatusTypes.FUNCTIONAL,
    ),
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.59.0-rc3",
        vllm_commit="8a43c88",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.P100,
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.P150,
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
    ),
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="v0.62.0-rc5",
        vllm_commit="3fc3263",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.P150X4,
                max_concurrency=32 * 4,
                max_context=128 * 1024,
                default_impl=True,
                override_tt_config={
                    "data_parallel": 4,
                    "sample_on_device_mode": "decode_only",
                },
            ),
        ],
        status=ModelStatusTypes.COMPLETE,
    ),
    ModelSpecTemplate(
        weights=["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="a9dfadb",
        vllm_commit="aa4ae1e",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.GALAXY,
                max_concurrency=32 * 4,
                max_context=64 * 1024,
                default_impl=True,
                override_tt_config={
                    "data_parallel": 4,
                    "sample_on_device_mode": "decode_only",
                },
                env_vars={
                    "TT_MM_THROTTLE_PERF": 5,
                },
            ),
        ],
        system_requirements=SystemRequirements(
            firmware=VersionRequirement(
                specifier=">=18.6.0",
                mode=VersionMode.STRICT,
            ),
            kmd=VersionRequirement(
                specifier=">=2.1.0",
                mode=VersionMode.STRICT,
            ),
        ),
        status=ModelStatusTypes.FUNCTIONAL,
    ),
    ModelSpecTemplate(
        weights=["Qwen/Qwen2.5-Coder-32B-Instruct"],
        impl=tt_transformers_impl,
        tt_metal_commit="6da108e",
        vllm_commit="005baf4",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=32,
                max_context=128 * 1024,
                default_impl=True,
            ),
        ],
        status=ModelStatusTypes.EXPERIMENTAL,
        env_vars={
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
        },
    ),
    ModelSpecTemplate(
        weights=["stabilityai/stable-diffusion-xl-base-1.0"],
        tt_metal_commit="v0.57.0-rc71",
        impl=tt_transformers_impl,
        min_disk_gb=15,
        min_ram_gb=6,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/tt-media-server-dev-ubuntu-22.04-amd64:v0.0.2-rc1",
        model_type=ModelType.CNN,
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=1,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=8,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.GALAXY,
                max_concurrency=32,
                max_context=64 * 1024,
                default_impl=True,
            ),
        ],
    ),
    ModelSpecTemplate(
        weights=["stabilityai/stable-diffusion-3.5-large"],
        tt_metal_commit="v0.57.0-rc71",
        impl=tt_transformers_impl,
        min_disk_gb=15,
        min_ram_gb=6,
        docker_image="http://ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64:v0.0.3-rc4",
        model_type=ModelType.CNN,
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.T3K,
                max_concurrency=1,
                max_context=64 * 1024,
                default_impl=True,
            ),
            DeviceModelSpec(
                device=DeviceTypes.GALAXY,
                max_concurrency=1,
                max_context=64 * 1024,
                default_impl=True,
            ),
        ],
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


# Final model specifications generated from templates
MODEL_SPECS = get_model_spec_map(spec_templates)


def get_runtime_model_spec(args):
    # Infer the impl from the default for given model_name if not provided
    if not args.impl:
        device_type = DeviceTypes.from_string(args.device)
        for _, model_spec in MODEL_SPECS.items():
            if (
                model_spec.model_name == args.model
                and model_spec.device_type == device_type
                and model_spec.device_model_spec.default_impl
            ):
                args.impl = model_spec.impl.impl_name
                break

    if not args.impl:
        raise ValueError(
            f"Model:={args.model} does not have a default impl, you must pass --impl"
        )

    model_id = get_model_id(args.impl, args.model, args.device)
    model_spec = MODEL_SPECS[model_id]
    model_spec.apply_runtime_args(args)

    return model_spec
