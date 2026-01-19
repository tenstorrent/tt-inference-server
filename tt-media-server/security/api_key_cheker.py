# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
from utils.logger import TTLogger

logger = TTLogger()

API_KEY = os.getenv("API_KEY", "your-secret-key")  # Or use os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="Authorization")


def get_api_key(api_key: str = Security(api_key_header)):
    expected = f"Bearer {API_KEY}"
    logger.info(f"Auth check - received: '{api_key}', expected: '{expected}'")
    if api_key != expected:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
