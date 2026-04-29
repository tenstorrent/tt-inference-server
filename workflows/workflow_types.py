# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from enum import Enum, IntEnum, auto
from typing import List


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
    STRESS_TESTS = auto()
    TESTS = auto()
    REPORTS = auto()
    SERVER = auto()
    RELEASE = auto()
    SPEC_TESTS = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid TaskType: {name}")


class WorkflowVenvType(IntEnum):
    SYSTEM_SOFTWARE_VALIDATION = auto()
    STRESS_TESTS_RUN_SCRIPT = auto()
    STRESS_TESTS = auto()
    EVALS_RUN_SCRIPT = auto()
    TESTS_RUN_SCRIPT = auto()
    BENCHMARKS_RUN_SCRIPT = auto()
    REPORTS_RUN_SCRIPT = auto()
    EVALS_COMMON = auto()
    EVALS_META = auto()
    EVALS_VISION = auto()
    EVALS_AUDIO = auto()
    EVALS_VIDEO = auto()
    EVALS_EMBEDDING = auto()
    BENCHMARKS_HTTP_CLIENT_VLLM_API = auto()
    BENCHMARKS_EMBEDDING = auto()
    BENCHMARKS_VIDEO = auto()
    BENCHMARKS_VLLM = auto()
    BENCHMARKS_GENAI_PERF = auto()
    BENCHMARKS_AIPERF = auto()
    HF_SETUP = auto()
    SERVER = auto()
    TT_SMI = auto()
    TT_TOPOLOGY = auto()


class BenchmarkTaskType(IntEnum):
    HTTP_CLIENT_VLLM_API = auto()
    HTTP_CLIENT_CNN_API = auto()
    HTTP_CLIENT_VIDEO_API = auto()
    GENAI_PERF = auto()
    AIPERF = auto()


class DeviceTypes(IntEnum):
    CPU = auto()
    GPU = auto()
    E150 = auto()
    N150 = auto()
    N150X4 = auto()
    N300 = auto()
    T3K = auto()
    P100 = auto()
    P150 = auto()
    P150X4 = auto()  # 4x P150 cards (1,4 mesh)
    P150X8 = auto()  # BH LoudBox - 8x P150 (2,4 mesh)
    P300 = auto()  # Single P300 card (2 dies)
    P300X2 = auto()  # 2x P300 cards = 4 chips (2,2 mesh)
    BLACKHOLE_GALAXY = auto()  # BH Galaxy - 32x P150 chips
    GALAXY = auto()
    GALAXY_T3K = auto()
    DUAL_GALAXY = auto()
    QUAD_GALAXY = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid DeviceType: {name}")

    def to_string(self) -> str:
        return self.name.upper()

    def to_mesh_device_str(self) -> str:
        mapping = {
            DeviceTypes.CPU: "CPU",
            DeviceTypes.E150: "E150",
            DeviceTypes.N150: "N150",
            DeviceTypes.P100: "P100",
            DeviceTypes.P150: "P150",
            DeviceTypes.P150X4: "P150x4",
            DeviceTypes.P150X8: "P150x8",
            DeviceTypes.P300: "P300",
            DeviceTypes.P300X2: "P300x2",
            DeviceTypes.BLACKHOLE_GALAXY: "BH-Galaxy",
            DeviceTypes.N150X4: "N150x4",
            DeviceTypes.N300: "N300",
            DeviceTypes.T3K: "T3K",
            DeviceTypes.GALAXY: "TG",
            DeviceTypes.GALAXY_T3K: "T3K",
            DeviceTypes.DUAL_GALAXY: "(8,8)",
            DeviceTypes.QUAD_GALAXY: "(8,16)",
            DeviceTypes.GPU: "GPU",
        }
        if self not in mapping:
            raise ValueError(f"Invalid DeviceType: {self}")
        return mapping[self]

    def to_product_str(self) -> str:
        mapping = {
            DeviceTypes.E150: "e150",
            DeviceTypes.N150: "n150",
            DeviceTypes.P100: "p100",
            DeviceTypes.P150: "p150",
            DeviceTypes.P150X4: "BH 4xP150",
            DeviceTypes.P150X8: "BH LoudBox",
            DeviceTypes.P300: "BH P300",
            DeviceTypes.P300X2: "BH QuietBox GE (2xP300)",
            DeviceTypes.BLACKHOLE_GALAXY: "BH Galaxy",
            DeviceTypes.N150X4: "4xn150",
            DeviceTypes.N300: "n300",
            DeviceTypes.T3K: "WH LoudBox/QuietBox",
            DeviceTypes.GALAXY: "WH Galaxy",
            DeviceTypes.GALAXY_T3K: "WH Galaxy",
            DeviceTypes.DUAL_GALAXY: "Dual WH Galaxy",
            DeviceTypes.QUAD_GALAXY: "Quad WH Galaxy",
        }
        if self not in mapping:
            raise ValueError(f"Invalid DeviceType: {self}")
        return mapping[self]

    def get_topology_requirement(self) -> bool:
        """Return the required system-level mesh topology for a given DeviceType"""
        # topology not required for Blackhole
        if self.is_blackhole():
            return

        # mesh topology only required for multi-wh configurations, excluding galaxy
        requires_mesh_topology = {DeviceTypes.N150X4, DeviceTypes.T3K}
        if self in requires_mesh_topology:
            return SystemTopology.MESH

        # TODO: for future, more advanced topology requirements

    def is_wormhole(self) -> bool:
        wormhole_devices = {
            DeviceTypes.N150,
            DeviceTypes.N300,
            DeviceTypes.N150X4,
            DeviceTypes.T3K,
            DeviceTypes.GALAXY,
            DeviceTypes.GALAXY_T3K,
        }
        return self in wormhole_devices

    def is_blackhole(self) -> bool:
        blackhole_devices = (
            DeviceTypes.P100,
            DeviceTypes.P150,
            DeviceTypes.P150X4,
            DeviceTypes.P150X8,
            DeviceTypes.P300,
            DeviceTypes.P300X2,
            DeviceTypes.BLACKHOLE_GALAXY,
        )
        return self in blackhole_devices

    def is_multihost(self) -> bool:
        """Check if this device type requires multi-host deployment."""
        return self in {DeviceTypes.DUAL_GALAXY, DeviceTypes.QUAD_GALAXY}

    def get_multihost_num_hosts(self) -> int:
        """Get expected number of hosts for multi-host device types.

        Returns:
            Number of hosts required for this device type.

        Raises:
            ValueError: If device type is not a multi-host type.
        """
        host_counts = {
            DeviceTypes.DUAL_GALAXY: 2,
            DeviceTypes.QUAD_GALAXY: 4,
        }
        if self not in host_counts:
            raise ValueError(
                f"Device type {self.name} is not a multi-host device type. "
                f"Supported: {[d.name for d in host_counts.keys()]}"
            )
        return host_counts[self]

    def get_data_parallel_subdevice(self, data_parallel: int) -> "DeviceTypes":
        data_parallel_map = {
            (DeviceTypes.GALAXY, 1): DeviceTypes.GALAXY,
            (DeviceTypes.GALAXY, 4): DeviceTypes.T3K,
            (DeviceTypes.GALAXY, 16): DeviceTypes.N300,
            (DeviceTypes.GALAXY, 32): DeviceTypes.N150,
            (DeviceTypes.T3K, 1): DeviceTypes.T3K,
            (DeviceTypes.T3K, 4): DeviceTypes.N300,
            (DeviceTypes.T3K, 8): DeviceTypes.N150,
            (DeviceTypes.GALAXY_T3K, 1): DeviceTypes.T3K,
            (DeviceTypes.GALAXY_T3K, 4): DeviceTypes.N300,
            (DeviceTypes.GALAXY_T3K, 8): DeviceTypes.N150,
            (DeviceTypes.N150X4, 1): DeviceTypes.N150X4,
            (DeviceTypes.N300, 1): DeviceTypes.N300,
            (DeviceTypes.N300, 2): DeviceTypes.N150,
            (DeviceTypes.N150, 1): DeviceTypes.N150,
            (DeviceTypes.P150X4, 4): DeviceTypes.P150,
            (DeviceTypes.P150X8, 8): DeviceTypes.P150,
            (DeviceTypes.BLACKHOLE_GALAXY, 1): DeviceTypes.BLACKHOLE_GALAXY,
            (DeviceTypes.BLACKHOLE_GALAXY, 4): DeviceTypes.P150X8,
            (DeviceTypes.BLACKHOLE_GALAXY, 8): DeviceTypes.P150X4,
            (DeviceTypes.BLACKHOLE_GALAXY, 32): DeviceTypes.P150,
        }
        if (self, data_parallel) not in data_parallel_map:
            raise ValueError(
                f"Invalid DeviceType or data_parallel: {self}, {data_parallel}"
            )
        return data_parallel_map[(self, data_parallel)]


class SystemTopology(Enum):
    """Enumerates all valid Wormhole system topologies"""

    MESH = "Mesh"
    LINEAR_TORUS = "Linear/Torus"
    ISOLATED = "Isolated or not configured"

    @classmethod
    def from_topology_string(cls, value: str):
        """Instantiates a SystemTopology from the result string from the `tt-topology -ls` command"""
        if value is None:
            raise ValueError(
                "Topology configuration value is None (tt-topology may have failed)"
            )
        value_lower = value.lower()
        for member in cls:
            if member.value is not None and member.value.lower() == value_lower:
                return member
        raise ValueError(f"Unknown topology configuration: {value}")


class ReportCheckTypes(IntEnum):
    NA = auto()
    PASS = auto()
    FAIL = auto()

    @classmethod
    def from_result(cls, result: bool):
        res_map = {
            None: ReportCheckTypes.NA,
            True: ReportCheckTypes.PASS,
            False: ReportCheckTypes.FAIL,
        }
        return res_map[result]

    @classmethod
    def to_display_string(cls, check_type: str):
        disp_map = {
            ReportCheckTypes.NA: "N/A",
            ReportCheckTypes.PASS: "PASS ✅",
            ReportCheckTypes.FAIL: "FAIL ⛔",
        }
        return disp_map[check_type]


class ModelStatusTypes(IntEnum):
    """
    EXPERIMENTAL: Model implementation is available, but is unstable or has performance issues.
    FUNCTIONAL: Model runs functionally without issue, but performance is lower than expected.
    COMPLETE: Operationally complete, performance is usable for most applications.
    TOP_PERF: Performance close to theoretical peak, nearly fully optimized.
    """

    EXPERIMENTAL = auto()
    FUNCTIONAL = auto()
    COMPLETE = auto()
    TOP_PERF = auto()

    @property
    def display_string(self) -> str:
        return {
            ModelStatusTypes.EXPERIMENTAL: "🛠️ Experimental",
            ModelStatusTypes.FUNCTIONAL: "🟡 Functional",
            ModelStatusTypes.COMPLETE: "🟢 Complete",
            ModelStatusTypes.TOP_PERF: "🚀 Top Performance",
        }[self]

    @property
    def required_target_tiers(self) -> List[str]:
        """Tiers that MUST pass for a model at this status level.

        Tiers not in this list are still computed and reported but
        treated as informational -- failures are accepted and do not
        block a release. This enables programmatic masking: e.g. an
        EXPERIMENTAL model (forge, new bring-up) can fail every
        performance benchmark and still be released.
        """
        tier_map = {
            ModelStatusTypes.EXPERIMENTAL: [],
            ModelStatusTypes.FUNCTIONAL: ["functional"],
            ModelStatusTypes.COMPLETE: ["functional", "complete"],
            ModelStatusTypes.TOP_PERF: ["functional", "complete", "target"],
        }
        return tier_map[self]


class EvalLimitMode(IntEnum):
    SMOKE_TEST = auto()
    CI_COMMIT = auto()
    CI_NIGHTLY = auto()
    CI_LONG = auto()

    @classmethod
    def from_string(cls, name: str):
        if name is None:
            return None
        try:
            return cls[name.upper().replace("-", "_")]
        except KeyError:
            raise ValueError(f"Invalid EvalLimitMode: {name}")


class VersionMode(IntEnum):
    """Defines the enforcement mode for a version requirement."""

    STRICT = auto()  # Requirement must be met, raises an error otherwise.
    SUGGESTED = auto()  # A warning is issued if the requirement is not met.


class InferenceEngine(Enum):
    VLLM = "vLLM"
    MEDIA = "media"
    FORGE = "forge"

    @property
    def display_name(self) -> str:
        return {
            InferenceEngine.VLLM: "vLLM (tt-metal integration fork)",
            InferenceEngine.MEDIA: "tt-media-server",
            InferenceEngine.FORGE: "tt-media-server (forge plugin)",
        }[self]

    @classmethod
    def from_string(cls, name: str):
        return cls[name.upper()]

    def to_string(self) -> str:
        return self.name.lower()


class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    NOACTION = "noaction"


class ModelType(IntEnum):
    LLM = auto()
    CNN = auto()
    AUDIO = auto()
    IMAGE = auto()
    EMBEDDING = auto()
    TEXT_TO_SPEECH = auto()
    VIDEO = auto()
    VLM = auto()  # Vision-Language Models (text+image-to-text)

    @property
    def display_name(self) -> str:
        display_names = {
            ModelType.LLM: "Large Language Model",
            ModelType.CNN: "Convolutional Neural Network",
            ModelType.AUDIO: "Audio",
            ModelType.IMAGE: "Image",
            ModelType.EMBEDDING: "Embedding",
            ModelType.TEXT_TO_SPEECH: "Text-to-Speech",
            ModelType.VIDEO: "Video",
            ModelType.VLM: "Vision-Language Model",
        }
        return display_names[self]

    @property
    def short_name(self) -> str:
        short_names = {
            ModelType.LLM: "LLM",
            ModelType.VLM: "VLM",
            ModelType.AUDIO: "Audio",
            ModelType.IMAGE: "Image",
            ModelType.CNN: "CNN",
            ModelType.EMBEDDING: "Embedding",
            ModelType.TEXT_TO_SPEECH: "TTS",
            ModelType.VIDEO: "Video",
        }
        return short_names[self]

    @property
    def task_type(self) -> str:
        task_types = {
            ModelType.LLM: "text",
            ModelType.VLM: "vlm",
            ModelType.AUDIO: "audio",
            ModelType.IMAGE: "image",
            ModelType.CNN: "cnn",
            ModelType.EMBEDDING: "embedding",
            ModelType.TEXT_TO_SPEECH: "text_to_speech",
            ModelType.VIDEO: "video",
        }
        return task_types[self]
