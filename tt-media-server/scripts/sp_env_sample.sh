# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

### These are all optional reference envs for the inference server.

#Central location for bundled python on the quad.
export UV_PYTHON_INSTALL_DIR=/data/sadesoye/uv_pythons

#MPI Vars for mpi4py and ffmpeg (skip if ffmpeg is already in PATH). Its important to use a python package that links the same library has the c++ code.
#This also works by itself: MPICC=/opt/openmpi-v5.0.7-ulfm/bin/mpicc uv pip install --no-binary mpi4py mpi4py --force-reinstall
#Overriding ffmpeg
export MPICC=/opt/openmpi-v5.0.7-ulfm/bin/mpicc
export PATH="/opt/openmpi-v5.0.7-ulfm/bin:$HOME/.local/bin:/data/sadesoye/ffmpeg-git-20240629-amd64-static:$PATH"
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

#Shared drive vars for tt-metal.
export PYTHONPATH="/data/sadesoye/tt-metal"
export TT_METAL_HOME="/data/sadesoye/tt-metal"
export TT_METAL_RUNTIME_ROOT="/data/sadesoye/tt-metal"
export HF_HOME=/data/cglagovich/hf_data

#DiT Vars. Caching makes initialization faster.
export TT_DIT_CACHE_DIR=/data/sadesoye/tt_dit_cache

#Fabric Vars copied over from original performance testing. May not be needed.
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120

#inference server Vars. These must be the same as used for SPPRunner.
export TT_VIDEO_SHM_INPUT=tt_video_in
export TT_VIDEO_SHM_OUTPUT=tt_video_out
export MODEL_RUNNER=tt-wan2.2
export TT_VIDEO_EXPORT_CRF=23
export USE_GREEDY_BASED_ALLOCATION=false
