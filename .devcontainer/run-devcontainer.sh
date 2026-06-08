#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
image_name="${TT_INFERENCE_DEV_IMAGE:-tt-inference-server-dev:latest}"
data_root="$(cd "${repo_root}/../.." && pwd -P)"
cpp_server="${repo_root}/tt-media-server/cpp_server"
tt_metal_home="${cpp_server}/tt-llm-engine/tt-metal"
install_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            install_args+=(--clean)
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--clean]"
            echo "  --clean    Clean tt-metal build artifacts before installing dependencies"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

"${script_dir}/check-workspace-path.sh" "${repo_root}"

docker build -t "${image_name}" "${script_dir}"

docker run --rm -it \
    --device=/dev/tenstorrent \
    -v /dev/shm:/dev/shm \
    --ipc=host \
    --mount source=/dev/hugepages-1G,target=/dev/hugepages-1G,type=bind \
    --mount source="${data_root}",target="${data_root}",type=bind \
    -p 8000:8000 \
    -p 9000:9000 \
    -e "TT_METAL_HOME=${tt_metal_home}" \
    -e "LD_LIBRARY_PATH=${tt_metal_home}/build/lib:${cpp_server}/build-full:${LD_LIBRARY_PATH:-}" \
    -w "${repo_root}" \
    "${image_name}" \
    bash -lc '.devcontainer/install-dependencies.sh "$PWD" "$@" && exec bash' \
    bash "${install_args[@]}"
