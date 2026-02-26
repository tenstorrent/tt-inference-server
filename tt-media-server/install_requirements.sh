#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -eo pipefail

echo "Installing requirements with CPU-only PyTorch..."

# Install main requirements
uv pip install -r requirements.txt

# Install WhisperX without dependencies to avoid PyTorch conflicts
echo "Installing WhisperX without dependencies..."
uv pip install whisperx==3.4.3 --no-deps

echo "Installation complete!"
