#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# Format all C/C++ sources under include, src, tests, benchmarks with clang-format.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CLANG_FORMAT=""
for cmd in clang-format-20 clang-format-19 clang-format-18 clang-format; do
    if command -v "$cmd" >/dev/null 2>&1; then
        CLANG_FORMAT="$cmd"
        break
    fi
done
if [ -z "${CLANG_FORMAT}" ]; then
    echo "No clang-format found. Install clang-format-20 or run install_dependencies.sh" >&2
    exit 1
fi

find include src tests benchmarks -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0 | xargs -0 "${CLANG_FORMAT}" -i
echo "Formatted with ${CLANG_FORMAT}"
