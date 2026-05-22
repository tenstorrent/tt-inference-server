# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

API_KEY = os.getenv("API_KEY", "your-secret-key")
NO_AUTH = os.getenv("NO_AUTH", "").lower() in ("1", "true", "yes")
# auto_error=False when NO_AUTH so missing header is None, not 403.
api_key_header = APIKeyHeader(name="Authorization", auto_error=not NO_AUTH)


def get_api_key(api_key: str | None = Security(api_key_header)):
    if NO_AUTH:
        return None
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
