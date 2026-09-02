# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tenstorrent-specific workflow types + backward-compatible re-exports.

Generic validation-framework enums (``WorkflowType``, ``ModelType``,
``ReportCheckTypes``, ``ModelStatusTypes``, ``EvalLimitMode``,
``AgenticTracesMode``, ``BenchmarkTaskType``, ``VersionMode``) are owned by
the engine in ``workflow_module/engine_types.py`` and re-exported here so
existing ``workflows/`` callers keep working unchanged.

The enums that remain defined in this module are Tenstorrent adapter
concerns: hardware taxonomy (``DeviceTypes``, ``SystemTopology``), inference
stack (``InferenceEngine``, ``ModelSource``), and TT host venv provisioning
(``WorkflowVenvType``).
"""

from enum import Enum, IntEnum, auto

from workflow_module.engine_types import (
    AgenticTracesMode,
    BenchmarkTaskType,
    EvalLimitMode,
    ModelStatusTypes,
    ModelType,
    ReportCheckTypes,
    VersionMode,
    WorkflowType,
)

__all__ = [
    # re-exported engine-owned types
    "AgenticTracesMode",
    "BenchmarkTaskType",
    "EvalLimitMode",
    "ModelStatusTypes",
    "ModelType",
    "ReportCheckTypes",
    "VersionMode",
    "WorkflowType",
    # Tenstorrent-specific types defined here
    "WorkflowVenvType",
    "DeviceTypes",
    "SystemTopology",
    "InferenceEngine",
    "ModelSource",
]


class WorkflowVenvType(IntEnum):
    SYSTEM_SOFTWARE_VALIDATION = auto()
    STRESS_TESTS_RUN_SCRIPT = auto()
    STRESS_TESTS = auto()
    EVALS_RUN_SCRIPT = auto()
    TESTS_RUN_SCRIPT = auto()
    BENCHMARKS_RUN_SCRIPT = auto()
    REPORTS_RUN_SCRIPT = auto()
    WORKFLOW_RUN_SCRIPT = auto()
    PREFIX_CACHE = auto()
    AGENTIC_TRACES = auto()
    LLM_VLLM = auto()
    LLM_GUIDELLM = auto()
    LLM_AIPERF = auto()
    SPEC_DECODE = auto()
    EVALS_COMMON = auto()
    EVALS_META = auto()
    EVALS_VISION = auto()
    EVALS_AUDIO = auto()
    EVALS_EMBEDDING = auto()
    EVALS_AGENTIC = auto()
    BENCHMARKS_VLLM = auto()
    BENCHMARKS_VLLM_FORGE = auto()
    BENCHMARKS_GENAI_PERF = auto()
    HF_SETUP = auto()
    SERVER = auto()
    TT_SMI = auto()
    TT_TOPOLOGY = auto()


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
    SUPER_CLUSTER = auto()

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
            DeviceTypes.SUPER_CLUSTER: "Super-Cluster",
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
            DeviceTypes.P300X2: "BH QuietBox 2",
            DeviceTypes.BLACKHOLE_GALAXY: "BH Galaxy",
            DeviceTypes.N150X4: "4xn150",
            DeviceTypes.N300: "n300",
            DeviceTypes.T3K: "WH LoudBox/QuietBox",
            DeviceTypes.GALAXY: "WH Galaxy",
            DeviceTypes.GALAXY_T3K: "WH Galaxy",
            DeviceTypes.DUAL_GALAXY: "Dual WH Galaxy",
            DeviceTypes.QUAD_GALAXY: "Quad WH Galaxy",
            DeviceTypes.SUPER_CLUSTER: "BH Super-Cluster",
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
            DeviceTypes.SUPER_CLUSTER,
        )
        return self in blackhole_devices

    def is_multihost(self) -> bool:
        """Check if this device type requires multi-host deployment."""
        return self in {
            DeviceTypes.DUAL_GALAXY,
            DeviceTypes.QUAD_GALAXY,
            DeviceTypes.SUPER_CLUSTER,
        }

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
            (DeviceTypes.DUAL_GALAXY, 8): DeviceTypes.T3K,
            (DeviceTypes.QUAD_GALAXY, 16): DeviceTypes.T3K,
            (DeviceTypes.SUPER_CLUSTER, 1): DeviceTypes.SUPER_CLUSTER,
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
