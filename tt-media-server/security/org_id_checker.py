# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
from fastapi import Header, HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

ORG_ID_HEADER = os.getenv("ORG_ID_HEADER", "X-TT-Organization")


def get_org_id(
    org_id_header_field: str = Header(None, alias=ORG_ID_HEADER),
) -> str:
    if not org_id_header_field or not org_id_header_field.strip():
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Organization header must not be empty",
        )
    return org_id_header_field
