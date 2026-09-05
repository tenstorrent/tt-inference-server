#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Stage a Quetzal artifact bundle read-only for the mount-based admission gate.
#
# This is the FALLBACK for hosts where content-address admission
# (verify_bundle, the default in run_vllm_api_server.py) is unavailable. It
# replaces the ad-hoc `fuse-overlayfs -o ro` one-liner with a supported,
# reviewable step. Prefer content-address admission on a writable FS
# (see docs/quetzal_bare_node_serve.md); use this only if you must.
#
# Usage:
#   scripts/quetzal_stage_bundle.sh <src-package-dir> <ro-mount-target>
#
# Then point QUETZAL_PACKAGE_ROOT / QZ_MODELS_ROOT at <ro-mount-target>.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <src-package-dir> <ro-mount-target>" >&2
    exit 2
fi

SRC="$1"
DST="$2"

if [[ ! -d "$SRC" ]]; then
    echo "error: source package dir does not exist: $SRC" >&2
    exit 1
fi

mkdir -p "$DST"

if command -v fuse-overlayfs >/dev/null 2>&1; then
    # Rootless, read-only overlay: lowerdir is the writable package, the mount
    # exposes it with write bits cleared so the mount-based gate admits it.
    fuse-overlayfs -o "lowerdir=${SRC}",ro "$DST"
    echo "staged (fuse-overlayfs ro): $SRC -> $DST"
elif mountpoint -q "$DST" 2>/dev/null; then
    echo "already mounted: $DST"
else
    # Fallback: read-only bind mount (requires privileges).
    mount --bind "$SRC" "$DST"
    mount -o remount,ro,bind "$DST"
    echo "staged (ro bind mount): $SRC -> $DST"
fi
