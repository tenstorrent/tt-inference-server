# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.workflow_types import DeviceTypes

MULTIHOST_DEVICES = frozenset({DeviceTypes.DUAL_GALAXY, DeviceTypes.QUAD_GALAXY})


def device_key(device: DeviceTypes) -> str:
    return device.name.lower()


def is_multihost(device: DeviceTypes) -> bool:
    return device in MULTIHOST_DEVICES
