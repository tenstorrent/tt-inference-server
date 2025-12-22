# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import Enum, IntEnum, auto


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
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
    LOCAL_SETUP_VALIDATION = auto()
    EVALS_RUN_SCRIPT = auto()
    TESTS_RUN_SCRIPT = auto()
    BENCHMARKS_RUN_SCRIPT = auto()
    REPORTS_RUN_SCRIPT = auto()
    EVALS_COMMON = auto()
    EVALS_META = auto()
    EVALS_VISION = auto()
    EVALS_AUDIO = auto()
    BENCHMARKS_HTTP_CLIENT_VLLM_API = auto()
    BENCHMARKS_GENAI_PERF = auto()
    BENCHMARKS_AIPERF = auto()
    HF_SETUP = auto()
    SERVER = auto()


class BenchmarkTaskType(IntEnum):
    HTTP_CLIENT_VLLM_API = auto()
    HTTP_CLIENT_CNN_API = auto()
    GENAI_PERF = auto()
    AIPERF = auto()


class DeviceTypes(IntEnum):
    CPU = auto()
    E150 = auto()
    N150 = auto()
    P100 = auto()
    P150 = auto()
    P150X4 = auto()
    P150X8 = auto()
    N150X4 = auto()
    N300 = auto()
    T3K = auto()
    GALAXY = auto()
    GALAXY_T3K = auto()
    GPU = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid DeviceType: {name}")

    def to_mesh_device_str(self) -> str:
        mapping = {
            DeviceTypes.CPU: "CPU",
            DeviceTypes.E150: "E150",
            DeviceTypes.N150: "N150",
            DeviceTypes.P100: "P100",
            DeviceTypes.P150: "P150",
            DeviceTypes.P150X4: "P150x4",
            DeviceTypes.P150X8: "P150x8",
            DeviceTypes.N150X4: "N150x4",
            DeviceTypes.N300: "N300",
            DeviceTypes.T3K: "T3K",
            DeviceTypes.GALAXY: "TG",
            DeviceTypes.GALAXY_T3K: "T3K",
            DeviceTypes.GPU: "GPU",
        }
        if self not in mapping:
            raise ValueError(f"Invalid DeviceType: {self}")
        return mapping[self]

    def to_product_str(self) -> str:
        mapping = {
            DeviceTypes.CPU: "CPU",
            DeviceTypes.E150: "e150",
            DeviceTypes.N150: "n150",
            DeviceTypes.P100: "p100",
            DeviceTypes.P150: "p150",
            DeviceTypes.P150X4: "4xp150",
            DeviceTypes.P150X8: "8xp150",
            DeviceTypes.N150X4: "4xn150",
            DeviceTypes.N300: "n300",
            DeviceTypes.T3K: "TT-LoudBox",
            DeviceTypes.GALAXY: "Tenstorrent Galaxy",
            DeviceTypes.GALAXY_T3K: "Tenstorrent Galaxy",
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
        )
        return True if self in blackhole_devices else False

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
        for member in cls:
            if member.value.lower() == value.lower():  # case-insensitive match
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

    @classmethod
    def to_display_string(cls, check_type: str):
        disp_map = {
            ModelStatusTypes.EXPERIMENTAL: "🛠️ Experimental",
            ModelStatusTypes.FUNCTIONAL: "🟡 Functional",
            ModelStatusTypes.COMPLETE: "🟢 Complete",
            ModelStatusTypes.TOP_PERF: "🚀 Top Performance",
        }
        return disp_map[check_type]


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
