#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Offline config check for SD3.5 on the BH QuietBox 2 (P300X2, 1x4 mesh).

WHY THIS EXISTS
---------------
Running SD3.5 on QB2 needs four pieces of configuration spread across two repos,
and nothing else verifies that they agree with each other:

  1. tt-metal   models/tt_dit/pipelines/stable_diffusion_35_large/
                pipeline_stable_diffusion_35_large.py  -- the (1, 4) _PRESETS entry
  2. this repo  tt-media-server/config/constants.py    -- (TT_SD3_5, P300X2) -> (1, 4)
  3. this repo  tt-media-server/tt_model_runners/dit_runners.py
                                                       -- ring fabric + 50MB trace
  4. this repo  workflows/model_specs/dev/image.yaml   -- P300X2 device block
                                                          (release metadata only,
                                                           not asserted here)

A mismatch between any two of them is a silent misconfiguration that only shows
up as a confusing runtime failure on hardware -- or worse, a slow/incorrect run.
Catching it here costs seconds; catching it after a container build costs ~13
minutes per attempt.

The tt-metal side is NOT covered by the existing SD3.5 pytest suite:
test_pipeline_sd35.py passes ``dit_parallel_config`` explicitly for its
``1x4cfg0sp0tp1`` case, which bypasses ``_PRESETS`` entirely. The media server
instead calls ``StableDiffusion3Pipeline.create_pipeline()``, which passes
neither a parallel config nor a topology, so it depends completely on the preset
table. That path had zero coverage before this script.

The topology assertion is the subtle one. Every other preset runs Linear, and
both ``default()`` and ``create_pipeline()`` default to Linear -- but a 1x4 row
needs Ring. Without the preset-level topology the server would silently build a
Linear config for a mesh the pipeline expects to be Ring-connected.

WHY THIS IS A SCRIPT AND NOT A PYTEST FILE
------------------------------------------
``tt-media-server/tests/conftest.py`` is mock-only by construction: it installs
MagicMock stubs for ``ttnn`` and ``models``, and at line 244 it dereferences
``ttnn.experimental.tensor``, an attribute that exists on the mock but NOT on the
real module. So importing real ttnn under that conftest fails during collection.
Putting these checks in ``tests/`` would therefore either fail to collect, or --
if the stubs won that race -- pass vacuously, since ``(1, 4) in MagicMock()`` is
truthy. A standalone script avoids both traps.

WHAT THIS PROVES AND DOES NOT PROVE
-----------------------------------
Proves: the four pieces of configuration resolve to consistent values, and the
QB2-specific settings do not leak into any other mesh shape.

Does NOT prove: that the 1x4 mesh actually runs. No device is opened, no fabric
is initialised, no image is generated. Those need hardware plus a build of the
matching tt-metal commit.

USAGE
-----
Needs a tt-metal checkout carrying the (1, 4) preset, and MODEL/DEVICE in the
environment -- ``dit_runners`` calls ``get_settings()`` at class-definition time,
so importing it without them raises ``KeyError`` on an unrelated runner name::

    cd <tt-inference-server>
    TTM=/path/to/tt-metal
    MODEL="stable-diffusion-3.5-large" DEVICE="p300x2" MODEL_RUNNER="tt-sd3.5" \
    PYTHONPATH=$TTM/ttnn:$TTM:$TTM/tools:$PWD/tt-media-server \
    TT_METAL_HOME=$TTM \
    LD_LIBRARY_PATH=$TTM/build_Release/lib \
    python tt-media-server/scripts/check_sd35_qb2_config.py

Exits 0 on success, 1 on any failed check, 2 if the prerequisites are missing.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

QB2_MESH = (1, 4)

# Meshes that must keep the pre-QB2 behaviour: Linear topology, no ring fabric,
# and the original 25MB trace region.
OTHER_MESHES = [(2, 2), (2, 4), (4, 8)]

_failures: list[str] = []


def check(label: str, got, want) -> bool:
    ok = got == want
    suffix = "" if ok else f"   (expected {want!r})"
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {got!r}{suffix}")
    if not ok:
        _failures.append(label)
    return ok


def _bail(msg: str) -> None:
    print(f"PREREQUISITE MISSING: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    try:
        import ttnn
    except ImportError as exc:
        _bail(f"cannot import ttnn ({exc}); is PYTHONPATH pointing at a tt-metal checkout?")

    # A MagicMock here would make every assertion below pass vacuously.
    if isinstance(sys.modules.get("ttnn"), MagicMock):
        _bail("ttnn is a MagicMock, not the real module; checks would pass vacuously")

    try:
        from models.tt_dit.pipelines.stable_diffusion_35_large import (
            pipeline_stable_diffusion_35_large as sd35,
        )
    except ImportError as exc:
        _bail(f"cannot import the SD3.5 pipeline ({exc})")

    # ---------------------------------------------------------------
    print("=" * 70)
    print("1. tt-metal: SD3.5 _PRESETS has the (1, 4) entry")
    print("=" * 70)
    preset = sd35._PRESETS.get(QB2_MESH)
    if preset is None:
        # Without this entry, default() raises KeyError on preset["cfg"] --
        # it indexes the preset unconditionally.
        print("  [FAIL] no (1, 4) entry; create_pipeline() would raise KeyError on preset['cfg']")
        _failures.append("_PRESETS (1,4) missing")
    else:
        # Only one mesh axis is wide on a 1x4 row, so sp and tp cannot each own
        # an axis the way the (2, 2) preset does: tp takes all four chips.
        check("cfg", preset["cfg"], (1, 0))
        check("sp", preset["sp"], (1, 0))
        check("tp", preset["tp"], (4, 1))
        check("num_links", preset["num_links"], 2)
        check("topology", preset.get("topology"), ttnn.Topology.Ring)

    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("2. tt-metal: the server path (create_pipeline -> default) picks Ring")
    print("=" * 70)
    # create_pipeline() passes neither topology nor dit_parallel_config, so
    # everything below comes from the preset. This is the real media-server path.
    cfg = sd35.StableDiffusion3PipelineConfig.default(mesh_shape=ttnn.MeshShape(*QB2_MESH))
    check("topology", cfg.topology, ttnn.Topology.Ring)
    check("num_links", cfg.num_links, 2)
    check("dit tp factor", cfg.dit_parallel_config.tensor_parallel.factor, 4)
    check("dit tp axis", cfg.dit_parallel_config.tensor_parallel.mesh_axis, 1)
    check("dit sp factor", cfg.dit_parallel_config.sequence_parallel.factor, 1)
    check("dit cfg factor", cfg.dit_parallel_config.cfg_parallel.factor, 1)

    print()
    print("  regression guard: every other shape stays Linear")
    for mesh in OTHER_MESHES:
        other = sd35.StableDiffusion3PipelineConfig.default(mesh_shape=ttnn.MeshShape(*mesh))
        check(f"  {mesh} topology", other.topology, ttnn.Topology.Linear)

    print()
    print("  an explicit caller argument must still win over the preset")
    forced = sd35.StableDiffusion3PipelineConfig.default(
        mesh_shape=ttnn.MeshShape(*QB2_MESH), topology=ttnn.Topology.Linear
    )
    check("  forced Linear on 1x4", forced.topology, ttnn.Topology.Linear)

    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("3. tt-inference-server: P300X2 resolves to the (1, 4) mesh")
    print("=" * 70)
    from config.constants import DeviceIds, DeviceTypes, ModelConfigs, ModelRunners

    entry = ModelConfigs.get((ModelRunners.TT_SD3_5, DeviceTypes.P300X2))
    if entry is None:
        print("  [FAIL] SD3.5 is not registered for P300X2")
        _failures.append("ModelConfigs P300X2 missing")
    else:
        # device_ids is a flat group of four, matching the other (1, 4) P300X2
        # entries (VLLMForge_GEMMA4_31B, VLLMForge_QWEN_32B, TT_Z_IMAGE_TURBO).
        # The SDXL runners' DEVICE_IDS_2X2_GROUP pairs with their (2, 1) mesh.
        check("device_mesh_shape", tuple(entry["device_mesh_shape"]), QB2_MESH)
        check("device_ids", entry["device_ids"], DeviceIds.DEVICE_IDS_4_GROUP.value)
        check("is_galaxy", entry["is_galaxy"], False)

    print()
    print("  regression guard: T3K and GALAXY untouched")
    for device, mesh in [(DeviceTypes.T3K, (2, 4)), (DeviceTypes.GALAXY, (4, 8))]:
        other = ModelConfigs.get((ModelRunners.TT_SD3_5, device))
        check(f"  {device.value} mesh", tuple(other["device_mesh_shape"]), mesh)

    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("4. tt-inference-server: runner device params gate on (1, 4)")
    print("=" * 70)
    from tt_model_runners.dit_runners import TTSD35Runner

    def params_for(mesh):
        # Constructing the runner wants a real device; the method only reads
        # self.settings.device_mesh_shape, so a stub is sufficient.
        stub = types.SimpleNamespace(settings=types.SimpleNamespace(device_mesh_shape=mesh))
        return TTSD35Runner.get_pipeline_device_params(stub)

    # The server default is FABRIC_1D (linear), which would contradict the Ring
    # topology the preset asks for. 25MB is also too small for the 4-chip trace
    # ("Creating trace buffers of size ... but only 25000000B is allocated").
    qb2 = params_for(QB2_MESH)
    check("trace_region_size", qb2.get("trace_region_size"), 50_000_000)
    check("fabric_config", qb2.get("fabric_config"), ttnn.FabricConfig.FABRIC_1D_RING)

    print()
    print("  regression guard: no other mesh gets the ring fabric or 50MB")
    # This is the containment requirement. Note TT_Z_IMAGE_TURBO also runs a
    # (1, 4) P300X2 mesh but is a different runner, so it is unaffected -- the
    # guard is the per-runner override, not the shape alone.
    for mesh in OTHER_MESHES:
        other = params_for(mesh)
        check(f"  {mesh} trace_region_size", other.get("trace_region_size"), 25_000_000)
        check(f"  {mesh} has fabric_config", "fabric_config" in other, False)

    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    if _failures:
        print(f"RESULT: {len(_failures)} CHECK(S) FAILED")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("RESULT: ALL CHECKS PASSED (no hardware touched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
