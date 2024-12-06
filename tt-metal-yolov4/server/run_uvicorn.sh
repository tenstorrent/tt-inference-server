# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

#!/bin/bash
uvicorn --host 0.0.0.0 --port 7000 server.fast_api_yolov4:app
