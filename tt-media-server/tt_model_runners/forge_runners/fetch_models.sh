#!/bin/bash

# Script to fetch specific folders from tt-forge-models using git sparse checkout

set -e  # Exit on any error

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
    vit/pytorch"
GIT_SHA="${1:-77ef8ec930cb5631edff3780e24de6856d5c2dd3}"

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
