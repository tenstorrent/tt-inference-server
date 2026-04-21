#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

set -eo pipefail

if [ "$1" != "--skip-venv" ]; then
    if [ -z "${TT_METAL_HOME}" ]; then
        echo "Error: TT_METAL_HOME is not set" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${TT_METAL_HOME}/python_env/bin/activate"
fi

exec uvicorn --host 0.0.0.0 main:app --lifespan on --port "${SERVICE_PORT:-8000}"
