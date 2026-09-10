# base_env_om.sh -- the OM replacement for C12's metal_env_H3.sh: the base MiniMax-H3 environment on an OM Blackhole quad
# (weights, DiT cache, MPI, fabric timeouts, SHM handshake). h3_deploy.sh's generated env file `source`s it FIRST and then
# sets TT_METAL_HOME / TT_METAL_RUNTIME_ROOT / PYTHONPATH / LD_LIBRARY_PATH from the two pinned trees and the C12 knobs
# (MINIMAX_H3_TRACE_DENOISE / BUCKET_DENOISE / CONSTRUCTION_WARMUP / VAE_OUTPUT, MODEL_RUNNER) -- so NONE of those belong
# here. Adapted from quad-agent h3-deploy/env_h3.sh (itself adapted from Samuel Adesoye's metal_env_H3.sh).
#
# Two things that matter on this quad:
#  * weights live on the shared read-only NFS /data_bh/h3, identical on all four ranks, so nothing is replicated;
#  * TT_DIT_CACHE_DIR is a LOCAL writable overlay (cache-merged = overlayfs of cache-upper over the prebuilt NFS cache,
#    which is readable by all but writable only by ubuntu; om_quad3_prep.sh mounts it on every host). Reads hit the
#    prebuilt entries; a cache miss writes locally instead of failing on a read-only mount, which is how the quad2
#    deployment died. Without the overlay the first start converts ~80 GB of weights per host, or fails on the read-only dir.

# H3_VM is the deployment root (env_om_quad3.sh). The MPI ranks and the ssh'd frontend start from a FRESH environment in
# which only the generated env file's `source /home/zni/c12/base_env_om.sh` runs, so H3_VM is not set there: fall back to
# this file's own directory, which is $H3_VM once om_quad3_prep.sh has copied it there. (Sourcing the checked-in copy
# without H3_VM set would point the cache at the git checkout -- always source the copy under $H3_VM.)
_om_self=${BASH_SOURCE[0]:-$0}
export H3_VM=${H3_VM:-$(cd "$(dirname "$(readlink -f "$_om_self")")" && pwd)}
unset _om_self
export TT_DIT_CACHE_DIR=${H3_VM:?}/cache-merged

# model: shared read-only NFS, identical on all four ranks
export H3_WEIGHTS=${H3_WEIGHTS:-/data_bh/h3}
export MINIMAX_H3_DIFFUSERS_DIR=$H3_WEIGHTS
export MINIMAX_H3_MODEL_PATH=$H3_WEIGHTS
export MINIMAX_H3_REPO=$H3_WEIGHTS
# prebuilt DiT cache on NFS = the lower layer of the overlay om_quad3_prep.sh mounts at $TT_DIT_CACHE_DIR
export H3_DIT_CACHE_LOWER=${H3_DIT_CACHE_LOWER:-$H3_WEIGHTS/dit_cache/tt_dit_cache}
# MINIMAX_H3_PIPELINE_PREFER_MAC is intentionally not exported: metal #54675 (exact fp32 depthwise conv1d) retired the
# prefer_mac lever; on one branch tip setting it to 0 raised "unexpected keyword argument 'prefer_mac'" after a full denoise.

# MPI / ffmpeg
export MPICC=/opt/openmpi-v5.0.7-ulfm/bin/mpicc
# ulfm bin for shells that source only this file; NOT /usr/bin (env_h3.sh has it): h3_deploy.sh sources this AFTER the media
# venv's activate, and /usr/bin ahead of $MEDIA_ENV/bin would shadow its python / uvicorn / tt-run with a system one.
export PATH="/opt/openmpi-v5.0.7-ulfm/bin:$PATH"

# tt-metal runtime knobs (TT_METAL_HOME, TT_METAL_RUNTIME_ROOT, PYTHONPATH, LD_LIBRARY_PATH come from the C12 env file)
export TT_METAL_INSPECTOR=0
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
# 300, not 120: a device waits in a fetch-queue collective for its peers, and a peer host that is
# still JIT-compiling kernels the others already have (per-host kernel caches diverge after a
# rebuild) can lag by minutes. At 120 s that lag was reported as "device timeout in fetch queue
# wait, potential hang detected" on a healthy mesh (2026-09-04, ref2va, second request). A real
# hang now takes 5 min to surface; the live tests budget 45 min per job.
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300
export TT_METAL_CLEAR_L1=1
# First-request ("warmup") speed, Jonathan Su 2026-09-05, measured on quad1 the same day:
# t2va first request in a fresh process 173 s -> 42 s with both set (warm kernel cache both
# times; startup and warm requests unchanged).
#  * TT_METAL_SHM_TRACKING_DISABLED=1 turns off the per-PID device-memory statistics tt-metal
#    publishes for tt-smi in /dev/shm/tt_device_*_memory. With them on, every buffer
#    allocate/free takes a mutex and records into all 32 devices' regions, and every newly
#    created program walks all active programs' circular buffers per device
#    (tt_metal/impl/memory_tracking/, program.cpp update_from_allocator). That walk is what
#    makes program-creation-heavy phases (audio decoder build, first denoise step) 5-10x
#    slower. Nothing on the quad reads the stats.
#  * TT_METAL_LOGS_PATH moves tt-metal's generated/ tree off the process CWD. Even with the
#    watcher off, every kernel registration appends to generated/watcher/kernel_*.txt
#    (~7k lines, ~2 MB per rank per process, fprintf+fflush each) and the fabric control
#    plane dumps its mappings there. Writers create the directory themselves.
export TT_METAL_SHM_TRACKING_DISABLED=1
export TT_METAL_LOGS_PATH=${TT_METAL_LOGS_PATH:-/tmp/tt-logs}
unset TT_METAL_CACHE                       # per-host ~/.cache/tt-metal-cache; never a shared dir (rank races). The C12 env file unsets it too.

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# media server <-> runner SHM handshake (must match on both sides; host-local, one deployment per host)
export TT_VIDEO_SHM_INPUT=tt_video_in
export TT_VIDEO_SHM_OUTPUT=tt_video_out
export TT_VIDEO_OUTPUT_DIR=$H3_VM/videos
export TT_VIDEO_EXPORT_CRF=23
export USE_GREEDY_BASED_ALLOCATION=false
export USE_ASYNC_VIDEO=true
