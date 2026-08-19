# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import hmac
import os
import re

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

# API_KEYS (comma-separated) supersedes API_KEY. Distinct keys are what make
# per-tenant isolation possible: each resolves to its own org_id, and JobManager
# already refuses cross-org reads. A single-key deployment behaves exactly as
# before -- every caller shares one org, so nothing is isolated until a second
# key is issued.
#
# Entries are "label:key" or a bare "key". The label is the tenant id, so job
# records carry a readable name and nothing derived from the credential. An
# earlier revision derived the id by hashing the key; that put a fast, unsalted
# hash of a possibly low-entropy secret into every job response via
# Job.to_public_dict, which is brute-forceable and was flagged by CodeQL. A label
# is both safer and simpler -- there is nothing to reverse.
#
# NOTE: video jobs deliberately do not use security/org_id_checker.get_org_id,
# which reads a caller-supplied X-TT-Organization header. That header is only
# trustworthy behind a gateway that sets it; presented directly, any caller could
# claim another tenant's id and read their jobs. Do not unify the two.
_DEFAULT_ORG = "default"
# Conservative label charset so a bare key is not mistaken for "label:key". A key
# that itself contains ':' must therefore be given an explicit label.
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _parse_key_specs(raw: str) -> tuple[tuple[str, ...], dict[str, str]]:
    """Parse API_KEYS into (keys, {key: org_id}).

    A bare key is only allowed when it is the only key. With several keys and no
    labels there is no non-arbitrary way to name the tenants, and naming them by
    position would silently reassign every tenant's jobs the first time the list
    is reordered or a key is rotated -- so that is rejected at startup rather than
    guessed at.
    """
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    keys: list[str] = []
    org_by_key: dict[str, str] = {}
    unlabelled: list[str] = []

    for entry in entries:
        label, sep, key = entry.partition(":")
        if sep and key and _LABEL_RE.match(label):
            org_id = label
        else:
            key, org_id = entry, _DEFAULT_ORG
            unlabelled.append(entry)
        if key in org_by_key and org_by_key[key] != org_id:
            msg = (
                f"API_KEYS: key appears twice with different tenant labels "
                f"({org_by_key[key]!r} and {org_id!r}). One key cannot belong to "
                "two tenants."
            )
            raise RuntimeError(msg)
        if key not in org_by_key:
            keys.append(key)
        org_by_key[key] = org_id

    if len(keys) > 1 and unlabelled:
        msg = (
            "API_KEYS: multiple keys are configured but "
            f"{len(unlabelled)} of them have no tenant label. Use "
            '"label:key" for each (e.g. API_KEYS="acme:k1,globex:k2") so jobs are '
            "scoped to a named tenant. Naming tenants by position would reassign "
            "their jobs whenever the list changes."
        )
        raise RuntimeError(msg)

    return tuple(keys), org_by_key


_RAW_KEYS = os.getenv("API_KEYS") or os.getenv("API_KEY", "your-secret-key")
API_KEYS, _ORG_BY_KEY = _parse_key_specs(_RAW_KEYS)
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
    """Tenant id for the presenting credential.

    Jobs are scoped by this, so callers holding keys labelled for different
    tenants cannot read, download, or cancel each other's work.

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
    return _ORG_BY_KEY[key]
