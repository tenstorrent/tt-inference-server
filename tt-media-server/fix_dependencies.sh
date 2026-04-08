# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# Remove packages that might contain CUDA
uv pip uninstall xformers diffusers torch torchvision torchaudio

# Install CPU-only versions
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

uv pip install diffusers==0.35.1

# Re-sync torch ecosystem: diffusers may have upgraded torch without upgrading
# torchvision to match, causing "operator torchvision::nms does not exist"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu