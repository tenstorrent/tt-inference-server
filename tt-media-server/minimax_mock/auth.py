"""Authentication and request-header checks for the MiniMax mock."""

from __future__ import annotations

import hmac
import os

from fastapi import Request

from minimax_mock.errors import MiniMaxAPIError

MOCK_API_KEY_ENV = "MINIMAX_MOCK_API_KEY"
_AUTH_ERROR_MESSAGE = (
    "login fail: Please carry the API secret key in the "
    "'Authorization' field of the request header (1004)"
)


def configured_api_key(explicit_key: str | None = None) -> str | None:
    key = explicit_key if explicit_key is not None else os.getenv(MOCK_API_KEY_ENV)
    if key is None or not key.strip():
        return None
    return key


def require_mock_api_key(request: Request) -> None:
    expected_key = configured_api_key(
        getattr(request.app.state, "minimax_mock_api_key", None)
    )
    if expected_key is None:
        raise RuntimeError(f"{MOCK_API_KEY_ENV} must be configured")

    authorization = request.headers.get("Authorization")
    if not authorization:
        _raise_authentication_error()

    scheme, separator, provided_key = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not provided_key
        or not hmac.compare_digest(provided_key, expected_key)
    ):
        _raise_authentication_error()


def require_json_content_type(request: Request) -> None:
    content_type = request.headers.get("Content-Type", "")
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type != "application/json":
        raise MiniMaxAPIError(
            status_code=400,
            error_type="bad_request_error",
            message="invalid params, Content-Type must be application/json (2013)",
        )


def _raise_authentication_error() -> None:
    raise MiniMaxAPIError(
        status_code=401,
        error_type="authorized_error",
        message=_AUTH_ERROR_MESSAGE,
    )
