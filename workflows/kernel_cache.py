# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Container location of the tt-metal kernel cache.

tt-metal reuses compiled kernels only at the same absolute cache path, so the
capture mount (run.py --kernel-cache-dir), tt-media-server/Dockerfile.kernel-cache
and every deployment of the resulting image use this one path.
"""

KERNEL_CACHE_ENV = "TT_METAL_CACHE"
KERNEL_CACHE_CONTAINER_DIR = "/home/container_app_user/tt_metal_kernel_cache"
KERNEL_CACHE_MANIFEST_CONTAINER_PATH = (
    "/home/container_app_user/kernel_cache_manifest.json"
)
