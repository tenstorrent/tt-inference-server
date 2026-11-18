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

# The plugin pins transformers==4.55.0, which requires huggingface-hub<1.0.
# Newer diffusers releases need Hub APIs unavailable after that downgrade.
# Keep this pin until the shared environment is upgraded and image models are tested.
uv pip install "diffusers==0.39.0"
