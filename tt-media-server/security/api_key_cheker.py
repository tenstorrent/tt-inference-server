# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

API_KEY = os.getenv("API_KEY", "your-secret-key")  # Or use os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="Authorization")


def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
