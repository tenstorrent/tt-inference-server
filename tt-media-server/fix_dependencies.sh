# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Remove packages that might contain CUDA
uv pip uninstall xformers diffusers torch torchvision torchaudio

# Install CPU-only versions
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install xformers without its CUDA sub-deps (--no-deps avoids re-pulling
# the CUDA torch wheels we just replaced with CPU-only ones above).
uv pip install xformers --no-deps

uv pip install diffusers
