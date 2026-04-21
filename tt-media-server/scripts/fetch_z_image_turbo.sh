#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Sparse checkout of z_image_turbo model from tt-metal.
set -euo pipefail

REPO_URL="https://github.com/tenstorrent/tt-metal.git"
BRANCH="svuckovic/z-image-turbo-inference-server"
CHECKOUT_DIR="${1:-$(dirname "$0")/../models/z_image_turbo_repo}"

if [ -d "$CHECKOUT_DIR/z_image_turbo" ]; then
    echo "Updating existing checkout at $CHECKOUT_DIR ..."
    git -C "$CHECKOUT_DIR" fetch origin "$BRANCH"
    git -C "$CHECKOUT_DIR" checkout "origin/$BRANCH" -- z_image_turbo models/tt_dit
else
    echo "Sparse checkout into $CHECKOUT_DIR ..."
    mkdir -p "$CHECKOUT_DIR"
    git -C "$CHECKOUT_DIR" init
    git -C "$CHECKOUT_DIR" remote add origin "$REPO_URL" 2>/dev/null || true
    git -C "$CHECKOUT_DIR" config core.sparseCheckout true
    printf "z_image_turbo\nmodels/tt_dit\n" > "$CHECKOUT_DIR/.git/info/sparse-checkout"
    git -C "$CHECKOUT_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$CHECKOUT_DIR" checkout FETCH_HEAD
fi

echo "Done: $(realpath "$CHECKOUT_DIR/z_image_turbo")"
