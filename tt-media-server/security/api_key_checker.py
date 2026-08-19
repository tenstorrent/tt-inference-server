# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import hashlib
import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

# API_KEYS (comma-separated) supersedes API_KEY. Distinct keys are what make
# per-tenant isolation possible: each resolves to its own org_id, and JobManager
# already refuses cross-org reads. A single-key deployment behaves exactly as
# before -- every caller shares one org, so nothing is isolated until a second
# key is issued.
_RAW_KEYS = os.getenv("API_KEYS") or os.getenv("API_KEY", "your-secret-key")
API_KEYS: tuple[str, ...] = tuple(k.strip() for k in _RAW_KEYS.split(",") if k.strip())
# Back-compat for anything importing the single-key name.
API_KEY = API_KEYS[0] if API_KEYS else ""
NO_AUTH = os.getenv("NO_AUTH", "").lower() in ("1", "true", "yes")
# auto_error=False when NO_AUTH so missing header is None, not 403.
api_key_header = APIKeyHeader(name="Authorization", auto_error=not NO_AUTH)


def _match_key(presented: str | None) -> str | None:
    """Return the configured key the caller presented, or None if none match.

    Uses compare_digest rather than ``==`` so the comparison does not short-circuit
    on the first differing byte.
    """
    if not presented:
        return None
    for key in API_KEYS:
        if hmac.compare_digest(presented, f"Bearer {key}"):
            return key
    return None


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


def get_api_key(api_key: str | None = Security(api_key_header)):
    if NO_AUTH:
        return None
    if _match_key(api_key) is None:
        raise _unauthorized()
    return api_key


def get_org_id(api_key: str | None = Security(api_key_header)) -> str | None:
    """Stable tenant id derived from the presenting credential.

    Jobs are scoped by this, so callers holding different keys cannot read,
    download, or cancel each other's work. Derived by hash rather than stored
    raw so that a job dump can never leak a live credential.

    Returns None only when auth is disabled, which JobManager treats as
    unscoped -- the pre-existing single-tenant behaviour. Raises rather than
    returning None for a bad key, so a failure here cannot fail open into
    unscoped access.
    """
    if NO_AUTH:
        return None
    key = _match_key(api_key)
    if key is None:
        raise _unauthorized()
    return hashlib.sha256(key.encode()).hexdigest()[:16]
