#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import subprocess
from typing import Dict, Iterator, Optional, Set, Tuple

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

DEVICE_DEFAULT_PRIORITY = (
    DeviceTypes.QUAD_GALAXY,
    DeviceTypes.DUAL_GALAXY,
    DeviceTypes.GALAXY,
    DeviceTypes.GALAXY_T3K,
    DeviceTypes.T3K,
    DeviceTypes.BLACKHOLE_GALAXY,
    DeviceTypes.P300X2,
    DeviceTypes.P300,
    DeviceTypes.P150X8,
    DeviceTypes.P150X4,
    DeviceTypes.P150,
    DeviceTypes.P100,
    DeviceTypes.N150X4,
    DeviceTypes.N300,
    DeviceTypes.N150,
    DeviceTypes.E150,
    DeviceTypes.CPU,
    DeviceTypes.GPU,
)

BOARD_TYPE_COUNT_TO_DEVICE: Dict[Tuple[Tuple[str, int], ...], DeviceTypes] = {
    (("n300", 1),): DeviceTypes.N300,
    (("n300", 4),): DeviceTypes.T3K,
    (("tt-galaxy-wh", 32),): DeviceTypes.GALAXY,
    (("n150", 1),): DeviceTypes.N150,
    (("n150", 4),): DeviceTypes.N150X4,
    (("e150", 1),): DeviceTypes.E150,
    (("p100", 1),): DeviceTypes.P100,
    (("p150", 1),): DeviceTypes.P150,
    (("p150", 4),): DeviceTypes.P150X4,
    (("p150", 8),): DeviceTypes.P150X8,
    (("p300", 1),): DeviceTypes.P300,
    (("p300", 2),): DeviceTypes.P300X2,
    (("tt-galaxy-bh", 32),): DeviceTypes.BLACKHOLE_GALAXY,
}


# tt-dra-driver boardName per board_type. Matches board_type except the Galaxy
# UBBs, which the driver names differently.
_DRA_BOARD_NAME_OVERRIDES: Dict[str, str] = {
    "tt-galaxy-wh": "galaxy-wormhole",
    "tt-galaxy-bh": "galaxy-blackhole",
}


def _dra_single_board_type_devices() -> Iterator[Tuple[str, str, int]]:
    """Yield (device_key, board_type, board_count) for each DRA-deployable device.

    Heterogeneous (multi-board-type) shapes are future work: they can't be
    expressed as a single boardName selector, so they are skipped here. Both DRA
    maps below are built from this one generator so deviceBoardCounts and
    deviceBoardNames always cover the exact same devices (no count-without-name
    drift that would render an unsatisfiable `boardName == ""` selector).
    """
    for board_tuple, device in BOARD_TYPE_COUNT_TO_DEVICE.items():
        board_types = {board_type for board_type, _ in board_tuple}
        if len(board_types) != 1:
            continue
        board_type = next(iter(board_types))
        board_count = sum(count for _, count in board_tuple)
        yield device.name.lower(), board_type, board_count


def dra_device_board_counts() -> Dict[str, int]:
    """Map device_key (e.g. "t3k") -> number of Tenstorrent boards a DRA
    ResourceClaim must request. Single source of truth for the chart's per-device
    board count (see _dra_single_board_type_devices).
    """
    return dict(
        sorted((key, count) for key, _bt, count in _dra_single_board_type_devices())
    )


# Boards that carry two ASICs; every other board type is single-chip. Matches the
# tt-dra-driver chipCount table (docs.tenstorrent.com/tt-dra-driver, single-host).
_TWO_CHIP_BOARD_TYPES = {"n300", "p300"}


def dra_device_chip_counts() -> Dict[str, int]:
    """Map device_key (e.g. "t3k") -> number of ASICs the device shape has.

    Drives the chart's hugepage sizing: tt-metal maps one 1Gi page per ASIC, so
    a device needs board_count x chips_per_board pages. Same source and device
    coverage as dra_device_board_counts (see _dra_single_board_type_devices).
    """
    return dict(
        sorted(
            (key, count * (2 if bt in _TWO_CHIP_BOARD_TYPES else 1))
            for key, bt, count in _dra_single_board_type_devices()
        )
    )


def dra_device_board_names() -> Dict[str, str]:
    """Map device_key (e.g. "t3k") -> the tt-dra-driver `boardName` a DRA
    ResourceClaim selects (via CEL). Same source and device coverage as
    dra_device_board_counts (see _dra_single_board_type_devices).
    """
    return dict(
        sorted(
            (key, _DRA_BOARD_NAME_OVERRIDES.get(bt, bt))
            for key, bt, _count in _dra_single_board_type_devices()
        )
    )


def _collect_supported_devices_for_model(
    model_name: str, engine: Optional[str] = None
) -> Set[DeviceTypes]:
    supported_devices = set()
    for _, model_spec in MODEL_SPECS.items():
        # Accept either the full HF repo id or the bare basename.
        if model_name not in (model_spec.model_name, model_spec.hf_model_repo):
            continue
        if engine and model_spec.inference_engine != engine:
            continue
        supported_devices.add(model_spec.device_type)
    return supported_devices


def _normalize_board_type(board_type: str) -> str:
    return " ".join(board_type.strip().lower().split())


def _ensure_tt_smi_venv_setup() -> None:
    venv_config = VENV_CONFIGS[WorkflowVenvType.TT_SMI]
    model_spec = next(iter(MODEL_SPECS.values()), None)
    if model_spec is None:
        raise ValueError(
            "No model specs available to setup TT_SMI virtual environment."
        )
    venv_config.setup(model_spec=model_spec)


def _get_tt_smi_board_type_counts() -> Dict[str, int]:
    _ensure_tt_smi_venv_setup()
    venv_config = VENV_CONFIGS[WorkflowVenvType.TT_SMI]
    tt_smi_executable = str(venv_config.venv_path / "bin" / "tt-smi")
    command = f"{tt_smi_executable} -s"
    tt_smi_output = subprocess.check_output(
        ["bash", "-lc", command],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    tt_smi_data = json.loads(tt_smi_output)
    board_type_counts: Dict[str, int] = {}
    board_chip_counts: Dict[str, int] = {}
    for info in tt_smi_data.get("device_info", []):
        board_info = info.get("board_info", {})
        raw_board_type = _normalize_board_type(board_info.get("board_type", ""))
        board_type = raw_board_type.split(" ", 1)[0]
        if board_type:
            if board_type in _TWO_CHIP_BOARD_TYPES:
                # tt-smi reports per-chip entries (e.g., n300 L/R), so convert chip
                # counts to board counts after collapsing L/R suffixes.
                board_chip_counts[board_type] = board_chip_counts.get(board_type, 0) + 1
            else:
                board_type_counts[board_type] = board_type_counts.get(board_type, 0) + 1

    for board_type, chip_count in board_chip_counts.items():
        if chip_count % 2 != 0:
            raise ValueError(
                f"Invalid tt-smi chip count for {board_type}: {chip_count}. "
                "Expected an even number of chips (L/R pairs)."
            )
        board_type_counts[board_type] = chip_count // 2

    return board_type_counts


def _map_board_type_counts_to_device_types(
    board_type_counts: Dict[str, int],
) -> Set[DeviceTypes]:
    normalized_counts = tuple(sorted(board_type_counts.items()))
    mapped_device = BOARD_TYPE_COUNT_TO_DEVICE.get(normalized_counts)
    if mapped_device:
        return {mapped_device}
    return set()


def infer_default_device(model_name: str, engine: Optional[str] = None) -> str:
    supported_devices = _collect_supported_devices_for_model(model_name, engine)
    if not supported_devices:
        engine_msg = f", engine={engine}" if engine else ""
        raise ValueError(
            f"No model specs found for model={model_name}{engine_msg}. "
            "Please provide --tt-device explicitly."
        )

    try:
        board_type_counts = _get_tt_smi_board_type_counts()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
    ) as exc:
        raise ValueError(
            "Unable to infer --tt-device from `tt-smi -s` output. "
            "Pass --tt-device explicitly."
        ) from exc

    available_devices = _map_board_type_counts_to_device_types(board_type_counts)
    if not available_devices:
        raise ValueError(
            "Unable to map tt-smi board counts to a supported DeviceTypes value. "
            f"Observed board counts: {board_type_counts}. Pass --tt-device explicitly."
        )

    candidate_devices = supported_devices & available_devices
    if not candidate_devices:
        supported_device_names = ", ".join(d.name.lower() for d in supported_devices)
        available_device_names = ", ".join(d.name.lower() for d in available_devices)
        raise ValueError(
            f"Detected device(s) [{available_device_names}] from tt-smi do not match "
            f"supported devices [{supported_device_names}] for model={model_name}. "
            "Pass --tt-device explicitly."
        )

    for device in DEVICE_DEFAULT_PRIORITY:
        if device in candidate_devices:
            return device.name.lower()

    supported_device_names = ", ".join(d.name.lower() for d in supported_devices)
    raise ValueError(
        f"Unable to infer --tt-device for model={model_name}. "
        f"Supported devices: {supported_device_names}"
    )
