# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for P300X2 alias support in media server constants."""

import pytest

from config.constants import DeviceTypes, ModelConfigs, ModelRunners


class TestDeviceTypesP300X2:
    """Tests for P300X2 enum entry in media server DeviceTypes."""

    def test_p300x2_enum_exists(self):
        """P300X2 must exist as a DeviceTypes enum member."""
        assert hasattr(DeviceTypes, "P300X2")

    def test_p300x2_value(self):
        """P300X2 enum value must be 'p300x2'."""
        assert DeviceTypes.P300X2.value == "p300x2"

    def test_qbge_still_exists(self):
        """QBGE enum member must still exist (backward compat)."""
        assert hasattr(DeviceTypes, "QBGE")
        assert DeviceTypes.QBGE.value == "qbge"

    def test_p300x2_and_qbge_are_distinct(self):
        """P300X2 and QBGE must be distinct enum members."""
        assert DeviceTypes.P300X2 != DeviceTypes.QBGE

    def test_device_lookup_by_value(self):
        """Must be able to look up P300X2 by its string value."""
        result = DeviceTypes("p300x2")
        assert result == DeviceTypes.P300X2


class TestModelConfigsP300X2:
    """Tests that ModelConfigs contains P300X2 entries for QBGE-supported models."""

    def test_flux_1_dev_p300x2_config_exists(self):
        """TT_FLUX_1_DEV must have a P300X2 config."""
        key = (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P300X2)
        assert key in ModelConfigs, "Missing ModelConfigs entry for TT_FLUX_1_DEV + P300X2"

    def test_flux_1_schnell_p300x2_config_exists(self):
        """TT_FLUX_1_SCHNELL must have a P300X2 config."""
        key = (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P300X2)
        assert key in ModelConfigs, "Missing ModelConfigs entry for TT_FLUX_1_SCHNELL + P300X2"

    def test_wan_2_2_p300x2_config_exists(self):
        """TT_WAN_2_2 must have a P300X2 config."""
        key = (ModelRunners.TT_WAN_2_2, DeviceTypes.P300X2)
        assert key in ModelConfigs, "Missing ModelConfigs entry for TT_WAN_2_2 + P300X2"

    def test_flux_1_dev_p300x2_mesh_shape(self):
        """TT_FLUX_1_DEV P300X2 must use 2x2 mesh (same as QBGE)."""
        config = ModelConfigs[(ModelRunners.TT_FLUX_1_DEV, DeviceTypes.P300X2)]
        assert config["device_mesh_shape"] == (2, 2)

    def test_flux_1_schnell_p300x2_mesh_shape(self):
        """TT_FLUX_1_SCHNELL P300X2 must use 2x2 mesh (same as QBGE)."""
        config = ModelConfigs[(ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.P300X2)]
        assert config["device_mesh_shape"] == (2, 2)

    def test_wan_2_2_p300x2_mesh_shape(self):
        """TT_WAN_2_2 P300X2 must use 1x4 mesh (same as QBGE)."""
        config = ModelConfigs[(ModelRunners.TT_WAN_2_2, DeviceTypes.P300X2)]
        assert config["device_mesh_shape"] == (1, 4)

    def test_qbge_configs_still_exist(self):
        """QBGE configs must still exist (backward compat)."""
        assert (ModelRunners.TT_FLUX_1_DEV, DeviceTypes.QBGE) in ModelConfigs
        assert (ModelRunners.TT_FLUX_1_SCHNELL, DeviceTypes.QBGE) in ModelConfigs
        assert (ModelRunners.TT_WAN_2_2, DeviceTypes.QBGE) in ModelConfigs
