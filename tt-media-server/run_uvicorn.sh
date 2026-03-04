# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

#!/bin/bash
set -eo pipefail
SERVICE_PORT="${SERVICE_PORT:-8000}"
echo "Starting uvicorn on port ${SERVICE_PORT}"
uvicorn --host 0.0.0.0 main:app --lifespan on --port "${SERVICE_PORT}"