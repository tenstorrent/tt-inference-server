"""MiniMax-compatible error responses."""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class MiniMaxAPIError(Exception):
    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(MiniMaxAPIError)
    async def handle_minimax_error(
        request: Request, exc: MiniMaxAPIError
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=exc.status_code,
            error_type=exc.error_type,
            message=exc.message,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=400,
            error_type="bad_request_error",
            message=f"invalid params, {_validation_message(exc)} (2013)",
        )


def _error_response(
    *,
    request: Request,
    status_code: int,
    error_type: str,
    message: str,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or secrets.token_hex(16)
    body = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
            "http_code": str(status_code),
        },
        "request_id": request_id,
    }
    return JSONResponse(status_code=status_code, content=body)


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "request body is invalid"

    error: dict[str, Any] = errors[0]
    location = ".".join(
        str(part) for part in error.get("loc", ()) if part not in {"body", "__root__"}
    )
    message = str(error.get("msg", "request body is invalid"))
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
