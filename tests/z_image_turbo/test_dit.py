# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""
DiT (Diffusion Transformer) PCC test -- compare one TTNN forward pass
against the PyTorch reference (HuggingFace ZImageTransformer2DModel)
using random data with a fixed seed.
"""

import pytest
import torch

import ttnn
from models.demos.z_image_turbo.tt.dit import model_pt
from models.demos.z_image_turbo.tt.dit.model_ttnn import ZImageTransformerTTNN

CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def pcc(a, b):
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.norm() * b_centered.norm()
    return (num / den).item() if den > 0 else 0.0


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_dit_pcc(mesh_device):
    mesh_device.enable_program_cache()

    torch.manual_seed(42)
    latent = torch.randn(
        LATENT_CHANNELS, 1, IMG_LATENT_H, IMG_LATENT_W, dtype=torch.bfloat16
    )
    cap_feats = torch.randn(CAP_TOKENS, 2560, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    pt_model = model_pt.load_model()
    model_pt.pad_heads(pt_model)
    pt_out = model_pt.forward(pt_model, [latent], timestep.float(), cap_feats)[0]
    del pt_model

    tt_model = ZImageTransformerTTNN(mesh_device)
    tt_model.set_cap_feats(cap_feats.unsqueeze(0))

    tt_lat = _to_device_bf16(latent, mesh_device)
    tt_ts = _to_device_bf16(timestep, mesh_device)

    # Compile run
    compile_out = tt_model._forward_impl([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    for t in compile_out:
        ttnn.deallocate(t, True)

    # Cached run
    tt_lat = _to_device_bf16(latent, mesh_device)
    tt_ts = _to_device_bf16(timestep, mesh_device)
    tt_out = tt_model._forward_impl([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    tt_result = _tt_to_torch(tt_out[0], mesh_device)

    correlation = pcc(pt_out.float(), tt_result)
    print(f"\nDiT PCC={correlation:.6f}")
    pt_f = pt_out.float()
    print(f"  PT shape={pt_f.shape}  "
          f"range=[{pt_f.min():.4f}, {pt_f.max():.4f}]")
    print(f"  TT shape={tt_result.shape}  "
          f"range=[{tt_result.min():.4f}, {tt_result.max():.4f}]")
    assert correlation > 0.998, f"DiT PCC too low: {correlation:.6f}"
