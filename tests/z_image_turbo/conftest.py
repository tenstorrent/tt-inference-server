# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

import pytest


@pytest.fixture(scope="function")
def device_params(request):
    import ttnn

    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    import ttnn

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = (1, 4)

    if isinstance(param, tuple):
        rows, cols = param
        num_requested = rows * cols
        mesh_shape = ttnn.MeshShape(rows, cols)
    else:
        num_requested = param
        mesh_shape = ttnn.MeshShape(1, param)

    if num_requested > ttnn.get_num_devices():
        avail = ttnn.get_num_devices()
        pytest.skip(f"Requested {num_requested} devices, {avail} available")

    fabric_config = device_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

    mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    yield mesh

    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)

    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
