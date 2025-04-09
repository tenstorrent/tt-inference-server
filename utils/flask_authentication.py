# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from flask import abort, request
from functools import wraps
from http import HTTPStatus
import jwt
import os
from typing import Optional


def normalize_token(token) -> [str, str]:
    """
    Note that scheme is case insensitive for the authorization header.
    See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization#directives
    """  # noqa: E501
    one_space = " "
    words = token.split(one_space)
    scheme = words[0].lower()
    return [scheme, " ".join(words[1:])]


def read_authorization(
    headers,
) -> Optional[dict]:
    authorization = headers.get("authorization")
    if not authorization:
        abort(HTTPStatus.UNAUTHORIZED, description="Must provide Authorization header.")
    [scheme, parameters] = normalize_token(authorization)
    if scheme != "bearer":
        user_error_msg = f"Authorization scheme was '{scheme}' instead of bearer"
        abort(HTTPStatus.UNAUTHORIZED, description=user_error_msg)
    try:
        payload = jwt.decode(parameters, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        if not payload:
            abort(HTTPStatus.UNAUTHORIZED)
        return payload
    except jwt.InvalidTokenError as exc:
        user_error_msg = f"JWT payload decode error: {exc}"
        abort(HTTPStatus.BAD_REQUEST, description=user_error_msg)


def api_key_required(f):
    """Decorates an endpoint to require API key validation"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        _ = read_authorization(request.headers)

        return f(*args, **kwargs)

    return wrapper
