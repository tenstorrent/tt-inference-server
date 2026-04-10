# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

#!/bin/bash
set -eo pipefail
uvicorn --host 0.0.0.0 main:app --lifespan on --port "${SERVICE_PORT:-8000}"