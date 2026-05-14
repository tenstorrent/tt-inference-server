# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

# No API_KEY → no auth required, matching vllm-tt-metal's handle_secrets() convention.
API_KEY = os.getenv("API_KEY")
# auto_error=False so missing header is None (handled below), not 403.
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def get_api_key(api_key: str | None = Security(api_key_header)):
    if not API_KEY:
        return None
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
