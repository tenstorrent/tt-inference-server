# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest

from workflows.helm_generator.device import device_key, is_multihost
from workflows.workflow_types import DeviceTypes


@pytest.mark.parametrize(
    "device, expected",
    [
        (DeviceTypes.GALAXY, "galaxy"),
        (DeviceTypes.GALAXY_T3K, "galaxy_t3k"),
        (DeviceTypes.T3K, "t3k"),
        (DeviceTypes.N150, "n150"),
        (DeviceTypes.N300, "n300"),
        (DeviceTypes.P300X2, "p300x2"),
        (DeviceTypes.DUAL_GALAXY, "dual_galaxy"),
    ],
)
def test_device_key_is_lowercase_name(device, expected):
    assert device_key(device) == expected


def test_is_multihost_true_for_dual_and_quad_galaxy():
    assert is_multihost(DeviceTypes.DUAL_GALAXY)
    assert is_multihost(DeviceTypes.QUAD_GALAXY)


def test_is_multihost_false_for_single_host_devices():
    assert not is_multihost(DeviceTypes.GALAXY)
    assert not is_multihost(DeviceTypes.T3K)
    assert not is_multihost(DeviceTypes.N300)
