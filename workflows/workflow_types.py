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
    TESTS_RUN_SCRIPT = auto()
    EVALS_RUN_SCRIPT = auto()
    BENCHMARKS_RUN_SCRIPT = auto()
    REPORTS_RUN_SCRIPT = auto()
    EVALS = auto()
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
    N300 = auto()
    T3K = auto()
    GALAXY = auto()

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
            DeviceTypes.N300: "n300",
            DeviceTypes.T3K: "TT-LoudBox",
            DeviceTypes.GALAXY: "Tenstorrent Galaxy",
        }
        if device not in mapping:
            raise ValueError(f"Invalid DeviceType: {device}")
        return mapping[device]
