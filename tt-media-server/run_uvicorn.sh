# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

#!/bin/bash
set -eo pipefail

if [ "$1" != "--skip-venv" ]; then
    source "${TT_METAL_HOME}/python_env/bin/activate"
fi

uvicorn --host 0.0.0.0 main:app --lifespan on --port "${SERVICE_PORT:-8000}"
