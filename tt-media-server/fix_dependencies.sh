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

# Pin to match tt-metal (tt_metal/python_env/requirements-dev.txt and
# models/tt_dit/pipelines/ltx/requirements.txt both pin diffusers==0.38.0).
#
# Unpinned, uv resolves diffusers 0.40.0, which declares
# `huggingface-hub>=1.23.0`. tt-metal's python_env already has
# huggingface_hub 0.36.2 installed (its own pin is `huggingface-hub >=
# 0.30.0`), and that pre-installed copy is not upgraded here, so the pair
# ends up incompatible: diffusers 0.40.0 imports `get_cached_repo_tree`,
# which does not exist before hub 1.23. Every diffusers-based runner then
# dies at import with
#   cannot import name 'get_cached_repo_tree' from 'huggingface_hub'
# and the worker never reaches device init (observed on SDXL: the server
# starts and /health answers, but `tt-sdxl-trace` fails to construct).
#
# diffusers 0.38.0 declares `huggingface-hub>=0.34.0`, which the existing
# 0.36.2 satisfies. SD3.5 and Wan2.2 are unaffected either way -- they
# already run on 0.38.0 in tt-metal's python_env -- so pinning here aligns
# the server with the rest of the stack rather than diverging from it.
uv pip install "diffusers==0.38.0"
