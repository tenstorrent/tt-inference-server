# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""
Text encoder PCC test -- compare TTNN forward pass against the PyTorch
reference (HuggingFace Qwen3Model) using the 5 standard evaluation prompts.
"""

import pytest
import torch

import ttnn
from models.demos.z_image_turbo.tt.text_encoder import model_pt
from models.demos.z_image_turbo.tt.text_encoder.model_ttnn import TextEncoderTTNN

CAP_TOKENS = 128
DRAM_RM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

PROMPTS = [
    "Cinematic cyberpunk street, neon rain puddles, glowing signs, 8k resolution.",
    "Whimsical treehouse village, golden hour lighting, floating lanterns, digital art.",
    "Majestic snow leopard, piercing blue eyes, mountain peak, hyper-realistic fur.",
    "Vintage oil painting, stormy sea, wooden ship, dramatic lightning strikes.",
    "Astronaut sitting on moon, drinking coffee, earth in background, surreal.",
]


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
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
@pytest.mark.parametrize(
    "prompt", PROMPTS, ids=[f"prompt_{i}" for i in range(len(PROMPTS))]
)
def test_text_encoder_pcc(mesh_device, prompt):
    mesh_device.enable_program_cache()

    input_ids = model_pt.tokenize(prompt)

    pt_model = model_pt.load_model()
    pt_result = model_pt.forward(pt_model, input_ids)
    del pt_model

    tt_model = TextEncoderTTNN(mesh_device, seq_len=CAP_TOKENS)

    # Compile run
    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_out = tt_model(tt_ids)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tt_out, True)
    ttnn.deallocate(tt_ids, True)

    # Cached run
    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_out = tt_model(tt_ids)
    ttnn.synchronize_device(mesh_device)
    tt_result = _tt_to_torch(tt_out, mesh_device)

    correlation = pcc(pt_result, tt_result)
    print(f"\nText encoder PCC={correlation:.6f} for: {prompt[:60]}...")
    print(f"  PT shape={pt_result.shape}  "
          f"range=[{pt_result.min():.4f}, {pt_result.max():.4f}]")
    print(f"  TT shape={tt_result.shape}  "
          f"range=[{tt_result.min():.4f}, {tt_result.max():.4f}]")
    assert correlation > 0.986, (
        f"Text encoder PCC too low: {correlation:.6f}"
    )
