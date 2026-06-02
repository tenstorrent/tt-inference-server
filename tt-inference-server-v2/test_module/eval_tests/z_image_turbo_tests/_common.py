# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Shared helpers for Z-Image-Turbo in-process eval tests."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

MESH_SHAPE = (1, 4)
NUM_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]

PROMPTS = [
    "Cinematic cyberpunk street, neon rain puddles, glowing signs, 8k resolution.",
    "Whimsical treehouse village, golden hour lighting, floating lanterns, digital art.",
    "Majestic snow leopard, piercing blue eyes, mountain peak, hyper-realistic fur.",
    "Vintage oil painting, stormy sea, wooden ship, dramatic lightning strikes.",
    "Astronaut sitting on moon, drinking coffee, earth in background, surreal.",
]

INFERENCE_STEPS = 9


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient over flattened tensors."""
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.norm() * b_centered.norm()
    return (num / den).item() if den > 0 else 0.0


@contextmanager
def open_mesh(mesh_shape=MESH_SHAPE):
    """Create a (rows, cols) mesh device with FABRIC_1D, yield, cleanup."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape)
    )
    mesh.enable_program_cache()
    try:
        yield mesh
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def to_device_int32(pt, mesh_device):
    import ttnn

    dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.BufferType.DRAM,
        None,
    )
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def to_device_bf16(pt, mesh_device):
    import ttnn

    dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.BufferType.DRAM,
        None,
    )
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def tt_to_torch(tt_tensor, mesh_device):
    import ttnn

    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // NUM_DEVICES].float()
