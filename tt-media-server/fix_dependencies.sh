# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# Capture torch version from tt-metal build before uninstalling so CPU swap matches
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "Pinning CPU torch to tt-metal's version: ${TORCH_VERSION}"

# Remove packages that might contain CUDA
uv pip uninstall xformers diffusers torch torchvision torchaudio

# Install CPU-only versions
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

uv pip install diffusers==0.35.1

# Re-sync torch ecosystem: diffusers may have upgraded torch without upgrading
# torchvision to match, causing "operator torchvision::nms does not exist"
uv pip install "torch==${TORCH_VERSION}+cpu" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu