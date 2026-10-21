# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest
from config.constants import (
    WAN22_RESOLUTION_LARGE_MESH,
    WAN22_RESOLUTION_SMALL_MESH,
    wan22_target_resolution,
)

GALAXY = (4, 8)
T3K = (2, 4)


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("WAN22_RESOLUTION", raising=False)


@pytest.mark.parametrize(
    "mesh_shape,expected",
    [(GALAXY, WAN22_RESOLUTION_LARGE_MESH), (T3K, WAN22_RESOLUTION_SMALL_MESH)],
)
def test_defaults_by_mesh_size(mesh_shape, expected):
    assert wan22_target_resolution(mesh_shape) == expected


@pytest.mark.parametrize("value", ["480p", "480P", " 480p "])
def test_480p_on_a_galaxy(monkeypatch, value):
    monkeypatch.setenv("WAN22_RESOLUTION", value)
    assert wan22_target_resolution(GALAXY) == WAN22_RESOLUTION_SMALL_MESH


def test_720p_on_a_small_mesh(monkeypatch):
    monkeypatch.setenv("WAN22_RESOLUTION", "720p")
    assert wan22_target_resolution(T3K) == WAN22_RESOLUTION_LARGE_MESH


def test_unknown_value_names_the_choices(monkeypatch):
    monkeypatch.setenv("WAN22_RESOLUTION", "1080p")
    with pytest.raises(ValueError, match=r"\['480p', '720p'\]"):
        wan22_target_resolution(GALAXY)
