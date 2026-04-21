#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

set -eo pipefail

if [ "$1" != "--skip-venv" ]; then
    source "${TT_METAL_HOME}/python_env/bin/activate"
fi

exec uvicorn --host 0.0.0.0 main:app --lifespan on --port "${SERVICE_PORT:-8000}"
