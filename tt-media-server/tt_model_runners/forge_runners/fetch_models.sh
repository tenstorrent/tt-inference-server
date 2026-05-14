#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# Script to fetch specific folders from tt-forge-models using git sparse checkout

set -e  # Exit on any error

FORGE_MODELS_SHA="93b03296089bceab40e7d371b6399fc8a88b1f2e"

REPO_URL="https://github.com/tenstorrent/tt-forge-models.git"
TARGET_DIR="model_loaders"
CHECKOUT_PATHS="
    tools
    resnet/pytorch
    vovnet/pytorch
    efficientnet/pytorch
    mobilenetv2/pytorch
    segformer/pytorch
    unet/pytorch
    vit/pytorch
    centerpoint/pytorch
    centerpoint/src"
GIT_SHA="${1:-$FORGE_MODELS_SHA}"

# Clean up any existing directory
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing $TARGET_DIR directory..."
    rm -rf "$TARGET_DIR"
fi

echo "Cloning tt-forge-models repository with sparse checkout..."
git clone --no-checkout "$REPO_URL" "$TARGET_DIR"
cd "$TARGET_DIR"
git sparse-checkout init --cone
git sparse-checkout set $CHECKOUT_PATHS
git fetch
git checkout $GIT_SHA
echo "Successfully fetched specified paths from tt-forge-models: $CHECKOUT_PATHS"
