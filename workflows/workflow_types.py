# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import IntEnum, auto


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
    TESTS = auto()
    REPORTS = auto()
    SERVER = auto()
    RELEASE = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid TaskType: {name}")


class WorkflowVenvType(IntEnum):
    EVALS_RUN_SCRIPT = auto()
    BENCHMARKS_RUN_SCRIPT = auto()
    REPORTS_RUN_SCRIPT = auto()
    EVALS = auto()
    EVALS_REASON = auto()
    EVALS_META = auto()
    EVALS_VISION = auto()
    BENCHMARKS_HTTP_CLIENT_VLLM_API = auto()
    SERVER = auto()


class BenchmarkTaskType(IntEnum):
    HTTP_CLIENT_VLLM_API = auto()


class DeviceTypes(IntEnum):
    CPU = auto()
    E150 = auto()
    N150 = auto()
    P100 = auto()
    P150 = auto()
    N300 = auto()
    T3K = auto()
    GALAXY = auto()
    GPU = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid DeviceType: {name}")

    @classmethod
    def to_mesh_device_str(cls, device: "DeviceTypes") -> str:
        mapping = {
            DeviceTypes.CPU: "CPU",
            DeviceTypes.E150: "E150",
            DeviceTypes.N150: "N150",
            DeviceTypes.P100: "P100",
            DeviceTypes.P150: "P150",
            DeviceTypes.N300: "N300",
            DeviceTypes.T3K: "T3K",
            DeviceTypes.GALAXY: "TG",
        }
        if device not in mapping:
            raise ValueError(f"Invalid DeviceType: {device}")
        return mapping[device]

    @classmethod
    def to_product_str(cls, device: "DeviceTypes") -> str:
        mapping = {
            DeviceTypes.CPU: "CPU",
            DeviceTypes.E150: "e150",
            DeviceTypes.N150: "n150",
            DeviceTypes.P100: "p100",
            DeviceTypes.P150: "p150",
            DeviceTypes.N300: "n300",
            DeviceTypes.T3K: "TT-LoudBox",
            DeviceTypes.GALAXY: "Tenstorrent Galaxy",
        }
        if device not in mapping:
            raise ValueError(f"Invalid DeviceType: {device}")
        return mapping[device]

    @classmethod
    def is_blackhole(cls, device: "DeviceTypes") -> bool:
        blackhole_devices = (cls.P100, cls.P150)
        return True if device in blackhole_devices else False

    def get_data_parallel_subdevice(self, data_parallel: int) -> "DeviceTypes":
        data_parallel_map = {
            (DeviceTypes.GALAXY, 1): DeviceTypes.GALAXY,
            (DeviceTypes.GALAXY, 4): DeviceTypes.T3K,
            (DeviceTypes.GALAXY, 16): DeviceTypes.N300,
            (DeviceTypes.GALAXY, 32): DeviceTypes.N150,
            (DeviceTypes.T3K, 1): DeviceTypes.T3K,
            (DeviceTypes.T3K, 4): DeviceTypes.N300,
            (DeviceTypes.T3K, 8): DeviceTypes.N150,
            (DeviceTypes.N300, 1): DeviceTypes.N300,
            (DeviceTypes.N300, 2): DeviceTypes.N150,
            (DeviceTypes.N150, 1): DeviceTypes.N150,
        }
        if (self, data_parallel) not in data_parallel_map:
            raise ValueError(f"Invalid DeviceType or data_parallel: {self}, {data_parallel}")
        return data_parallel_map[(self, data_parallel)]


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
