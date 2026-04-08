#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Capture a Tracy profile from the running server.
# Usage: ./tracy-capture.sh [SECONDS] [PORT]
#   SECONDS  capture duration (default: 60)
#   PORT     Tracy port to connect to (default: 8086 = main process)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION="${1:-60}"
PORT="${2:-8086}"
OUTPUT="${SCRIPT_DIR}/capture.tracy"

# Find tracy-src from whichever build directory has it
TRACY_SRC=""
for dir in build-tracy build; do
    candidate="${SCRIPT_DIR}/${dir}/_deps/tracy-src"
    if [ -d "${candidate}/capture" ]; then
        TRACY_SRC="${candidate}"
        break
    fi
done

if [ -z "${TRACY_SRC}" ]; then
    echo "ERROR: Tracy source not found. Build with Tracy first:"
    echo "  ./build.sh --tracy"
    exit 1
fi

CAPTURE_DIR="${TRACY_SRC}/capture/build/unix"
CAPTURE_BIN="${CAPTURE_DIR}/capture-release"

if [ ! -f "${CAPTURE_BIN}" ]; then
    echo "Building Tracy capture tool..."
    make -C "${CAPTURE_DIR}" -j"$(nproc 2>/dev/null || echo 1)" release
    echo ""
fi

echo "Capturing Tracy profile for ${DURATION}s on port ${PORT}..."
echo "  Output: ${OUTPUT}"
echo "  Press Ctrl+C to stop early."
echo ""

"${CAPTURE_BIN}" -o "${OUTPUT}" -p "${PORT}" -s "${DURATION}" -f
