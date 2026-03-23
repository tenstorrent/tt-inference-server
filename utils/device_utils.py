# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from typing import Optional

DEVICE_TO_MESH_STR = {
    "CPU": "CPU",
    "E150": "E150",
    "N150": "N150",
    "P100": "P100",
    "P150": "P150",
    "P150X4": "P150x4",
    "P150X8": "P150x8",
    "N150X4": "N150x4",
    "N300": "N300",
    "P300": "P300",
    "P300X2": "P300x2",
    "T3K": "T3K",
    "GALAXY": "TG",
    "GALAXY_T3K": "T3K",
    "DUAL_GALAXY": "(8,8)",
    "QUAD_GALAXY": "(8,16)",
    "GPU": "GPU",
}


def _get_value(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_mesh_device_str(device) -> str:
    if hasattr(device, "to_mesh_device_str"):
        return device.to_mesh_device_str()

    device_name = _get_value(device, "name", device)
    if device_name is None:
        raise ValueError("Device name is required to resolve a tensor cache path.")

    mesh_device_str = DEVICE_TO_MESH_STR.get(str(device_name).upper())
    if mesh_device_str is None:
        raise ValueError(f"Unknown device type: {device_name}")
    return mesh_device_str


def get_mesh_device_name(model_spec=None, device: Optional[str] = None) -> str:
    subdevice_type = _get_value(model_spec, "subdevice_type")
    if subdevice_type is not None:
        return to_mesh_device_str(subdevice_type)

    device_value = device
    if device_value is None:
        device_value = _get_value(_get_value(model_spec, "device_type"), "name")
    return to_mesh_device_str(device_value)
