# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

#!/bin/bash
set -eo pipefail

# Increase HuggingFace download timeout for slower networks
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10 minutes

uvicorn --host 0.0.0.0 main:app --lifespan on --port 8000