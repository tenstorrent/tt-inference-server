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

# Pinned: tt-metal's tt_dit pipelines are developed against diffusers 0.38.0
# (models/tt_dit/pipelines/ltx/requirements.txt). Unpinned, this resolves to
# 0.40.0, whose pipeline_utils needs huggingface_hub>=1 (get_cached_repo_tree)
# while this venv keeps huggingface_hub 0.36.2, so importing any diffusers
# pipeline (Qwen-Image-Edit included) fails.
uv pip install diffusers==0.38.0
