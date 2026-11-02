# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Remove packages that might contain CUDA
uv pip uninstall xformers diffusers torch torchvision torchaudio

# Install CPU-only versions
# Keep the torch version tt-metal pins (tt_metal/python_env/requirements-dev.txt);
# unpinned, this resolves to whatever is newest on the CPU index.
TORCH_PIN=$(sed -nE "s/^torch==([0-9.]+) ; platform_machine == '$(uname -m)'.*/\1/p" \
    tt_metal/python_env/requirements-dev.txt 2>/dev/null | head -1)
uv pip install "torch${TORCH_PIN:+==$TORCH_PIN}" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install xformers without its CUDA sub-deps (--no-deps avoids re-pulling
# the CUDA torch wheels we just replaced with CPU-only ones above).
uv pip install xformers --no-deps

# The plugin pins transformers==4.55.0, which requires huggingface-hub<1.0.
# Newer diffusers releases need Hub APIs unavailable after that downgrade.
# Keep this pin until the shared environment is upgraded and image models are tested.
uv pip install "diffusers==0.39.0"
