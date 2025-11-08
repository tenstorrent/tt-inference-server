#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Run latency benchmarks for TT-Comfy Bridge

set -e

echo "Running TT-Comfy Bridge Latency Benchmarks..."
echo ""

cd "$(dirname "$0")"

# Run benchmarks
python tests/test_latency_benchmark.py

echo ""
echo "Benchmark run complete!"

