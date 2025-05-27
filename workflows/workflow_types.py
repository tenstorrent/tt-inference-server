# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import IntEnum, auto


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    DOCKER_EVALS = auto()
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
    DOCKER_EVALS_RUN_SCRIPT = auto()
    DOCKER_EVALS_LMMS_EVAL = auto()
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
    def arch_name(cls, device: "DeviceTypes") -> str:
        arch_name = ""
        if cls._is_blackhole(device):
            arch_name = "blackhole"
        elif cls._is_wormhole(device):
            arch_name = "wormhole_b0"
        else:
            raise ValueError("DeviceType is neither Wormhole or Blackhole")
        return arch_name

    @classmethod
    def wh_arch_yaml(cls, device: "DeviceTypes") -> str:
        wh_arch_yaml_var = ""
        if device in (cls.N150, cls.N300, cls.T3K):
            wh_arch_yaml_var = "wormhole_b0_80_arch_eth_dispatch.yaml"
        return wh_arch_yaml_var

    @classmethod
    def _is_wormhole(cls, device: "DeviceTypes") -> bool:
        wormhole_devices = (cls.N150, cls.N300, cls.T3K, cls.GALAXY)
        return True if device in wormhole_devices else False

    @classmethod
    def _is_blackhole(cls, device: "DeviceTypes") -> bool:
        blackhole_devices = (cls.P100, cls.P150)
        return True if device in blackhole_devices else False


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
