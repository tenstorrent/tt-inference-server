# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

#!/bin/bash
set -eo pipefail

# Suppress xFormers warnings when using CPU-only PyTorch
export XFORMERS_FORCE_DISABLE_TRITON=1

uvicorn --host 0.0.0.0 main:app --lifespan on --port 8000