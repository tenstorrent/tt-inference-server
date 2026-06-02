# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
"""
VAE decoder PCC test -- compare TTNN forward pass against the PyTorch
reference (diffusers AutoencoderKL decoder) using random data with a fixed seed.
"""

import pytest

from models.demos.z_image_turbo.tt.vae.model_pt import VaeDecoderPT, get_input
from models.demos.z_image_turbo.tt.vae.model_ttnn import VaeDecoderTTNN


def pcc(a, b):
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.norm() * b_centered.norm()
    return (num / den).item() if den > 0 else 0.0


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_vae_pcc(mesh_device):
    mesh_device.enable_program_cache()

    latent = get_input()

    pt_vae = VaeDecoderPT()
    pt_result = pt_vae.forward(latent)
    del pt_vae

    tt_vae = VaeDecoderTTNN(mesh_device)
    tt_result = tt_vae(latent)

    correlation = pcc(pt_result, tt_result)
    print(f"\nVAE PCC={correlation:.6f}")
    print(f"  PT shape={pt_result.shape}  "
          f"range=[{pt_result.min():.4f}, {pt_result.max():.4f}]")
    print(f"  TT shape={tt_result.shape}  "
          f"range=[{tt_result.min():.4f}, {tt_result.max():.4f}]")
    assert correlation > 0.998, f"VAE PCC too low: {correlation:.6f}"
