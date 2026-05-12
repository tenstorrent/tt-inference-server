# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Tests for the Wan2.2 T2V/I2V shared fabric-resolution policy.

These pin the per-mesh fabric/trace-region defaults so that the original
regression (refactor that dropped FABRIC_1D_RING for the BH p150x4 (1,4)
tray and surfaced as ``topology_mapper.cpp:527 mapping_result.success``)
cannot recur silently.
"""

from unittest.mock import patch

import pytest
import ttnn  # mocked by tests/conftest.py
from tt_model_runners.dit_runners import (
    WAN22_BH_RING_MESH_SHAPES,
    WAN22_GALAXY_BH_TRACE_REGION_BYTES,
    WAN22_GALAXY_ROUTER_MAX_PAYLOAD_BYTES,
    _wan22_dit_device_params,
    _wan22_needs_ring_fabric,
)

P150X4_MESH_SHAPE = (1, 4)
T3K_MESH_SHAPE = (2, 4)
GLX_WORMHOLE_MESH_SHAPE = (4, 8)
GLX_BLACKHOLE_MESH_SHAPE = (4, 32)


def _patch_arch(*, blackhole: bool):
    """Patch the ``is_blackhole`` symbol used by the resolver."""
    return patch(
        "tt_model_runners.dit_runners.is_blackhole",
        return_value=blackhole,
    )


class TestWan22NeedsRingFabric:
    """Predicate that drives FABRIC_1D vs FABRIC_1D_RING selection."""

    def test_galaxy_mesh_always_needs_ring_on_blackhole(self):
        with _patch_arch(blackhole=True):
            assert _wan22_needs_ring_fabric(GLX_BLACKHOLE_MESH_SHAPE) is True

    def test_galaxy_mesh_always_needs_ring_on_wormhole(self):
        with _patch_arch(blackhole=False):
            assert _wan22_needs_ring_fabric(GLX_WORMHOLE_MESH_SHAPE) is True

    def test_p150x4_needs_ring_on_blackhole(self):
        # Regression guard: dropping (1,4) from the ring set silently broke
        # i2v on p150x4 with "graph could not fit" in the topology mapper.
        with _patch_arch(blackhole=True):
            assert _wan22_needs_ring_fabric(P150X4_MESH_SHAPE) is True

    def test_p150x4_does_not_need_ring_on_wormhole(self):
        # The (1,4) ring wiring is a Blackhole-tray property, not a generic
        # 4-chip property. WH 4-chip meshes should stay linear.
        with _patch_arch(blackhole=False):
            assert _wan22_needs_ring_fabric(P150X4_MESH_SHAPE) is False

    def test_t3k_does_not_need_ring(self):
        with _patch_arch(blackhole=False):
            assert _wan22_needs_ring_fabric(T3K_MESH_SHAPE) is False

    def test_p150x4_is_in_bh_ring_mesh_shapes(self):
        # Source-of-truth set documents the small BH ring boards; the
        # resolver derives behavior from this set, not from inline checks.
        assert P150X4_MESH_SHAPE in WAN22_BH_RING_MESH_SHAPES


class TestWan22DitDeviceParamsBlackhole:
    """End-to-end resolution for Blackhole meshes."""

    def test_p150x4_uses_ring_fabric_and_relaxed_init(self):
        with _patch_arch(blackhole=True):
            params = _wan22_dit_device_params(P150X4_MESH_SHAPE)

        assert params["fabric_config"] is ttnn.FabricConfig.FABRIC_1D_RING
        assert params["reliability_mode"] is ttnn.FabricReliabilityMode.RELAXED_INIT
        # Small BH meshes do not bump the trace region or router payload.
        assert "trace_region_size" not in params
        assert "fabric_router_config" not in params

    def test_galaxy_bh_uses_ring_relaxed_init_and_bumped_trace(self):
        with _patch_arch(blackhole=True):
            params = _wan22_dit_device_params(GLX_BLACKHOLE_MESH_SHAPE)

        assert params["fabric_config"] is ttnn.FabricConfig.FABRIC_1D_RING
        assert params["reliability_mode"] is ttnn.FabricReliabilityMode.RELAXED_INIT
        assert params["trace_region_size"] == WAN22_GALAXY_BH_TRACE_REGION_BYTES
        router = params["fabric_router_config"]
        assert (
            router.max_packet_payload_size_bytes
            == WAN22_GALAXY_ROUTER_MAX_PAYLOAD_BYTES
        )


class TestWan22DitDeviceParamsWormhole:
    """End-to-end resolution for Wormhole meshes."""

    def test_t3k_uses_linear_fabric_and_no_reliability_override(self):
        with _patch_arch(blackhole=False):
            params = _wan22_dit_device_params(T3K_MESH_SHAPE)

        assert params["fabric_config"] is ttnn.FabricConfig.FABRIC_1D
        assert "reliability_mode" not in params
        assert "trace_region_size" not in params
        assert "fabric_router_config" not in params

    def test_galaxy_wh_uses_ring_without_bh_specific_extras(self):
        with _patch_arch(blackhole=False):
            params = _wan22_dit_device_params(GLX_WORMHOLE_MESH_SHAPE)

        assert params["fabric_config"] is ttnn.FabricConfig.FABRIC_1D_RING
        # Trace bump and router config are gated on Blackhole, not on mesh size.
        assert "reliability_mode" not in params
        assert "trace_region_size" not in params
        assert "fabric_router_config" not in params


@pytest.mark.parametrize("mesh_shape", sorted(WAN22_BH_RING_MESH_SHAPES))
def test_every_registered_bh_ring_mesh_resolves_to_ring(mesh_shape):
    """Any shape declared in the BH ring set must resolve to FABRIC_1D_RING."""
    with _patch_arch(blackhole=True):
        params = _wan22_dit_device_params(mesh_shape)
    assert params["fabric_config"] is ttnn.FabricConfig.FABRIC_1D_RING
