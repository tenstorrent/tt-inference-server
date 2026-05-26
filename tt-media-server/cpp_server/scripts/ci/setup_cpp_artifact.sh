#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

ARTIFACT="cpp_server/cpp-server-build.tar.gz"
INSTALL_DEPS=1

usage() {
    cat <<'EOF'
Usage: setup_cpp_artifact.sh [options]

Options:
  --artifact PATH         Build artifact tarball (default: cpp_server/cpp-server-build.tar.gz)
  --skip-deps            Do not install runtime dependencies first
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact) ARTIFACT="$2"; shift 2 ;;
        --skip-deps) INSTALL_DEPS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$INSTALL_DEPS" = 1 ]]; then
    cpp_server/install_dependencies.sh --runtime
fi

if [[ ! -f "$ARTIFACT" ]]; then
    echo "::error::C++ build artifact not found: $ARTIFACT"
    exit 1
fi

artifact_dir="$(dirname "$ARTIFACT")"
tar -xzf "$ARTIFACT" -C "$artifact_dir"
rm -f "$ARTIFACT"
chmod +x cpp_server/build/tt_media_server_cpp
