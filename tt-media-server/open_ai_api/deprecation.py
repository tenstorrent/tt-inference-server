# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# Paths excluded from deprecation checks (internal / infrastructure)
EXCLUDED_PREFIXES = (
    "/v1/",
    "/tt-",
    "/static",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/health",
    "/ready",
)


class DeprecatedPathMiddleware(BaseHTTPMiddleware):
    """Middleware that adds deprecation headers to responses for non-/v1 API paths.

    During the deprecation period, both /v1/... and legacy /... paths are served.
    Requests to legacy paths receive Deprecation and Sunset headers to signal
    clients should migrate to the /v1-prefixed endpoints.

    Headers added (RFC 8594, RFC 8288):
        - Deprecation: true
        - Sunset: <date>
        - Link: </v1/path>; rel="successor-version"
    """

    def __init__(self, app, sunset_date: str = "2025-12-31"):
        super().__init__(app)
        self.sunset_date = sunset_date
        self._warned_paths: set[str] = set()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        is_deprecated = self._is_deprecated_path(path)

        response = await call_next(request)

        if is_deprecated:
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = self.sunset_date
            response.headers["Link"] = f'</v1{path}>; rel="successor-version"'
            self._log_deprecation_warning(path)

        return response

    def _is_deprecated_path(self, path: str) -> bool:
        """Check if a path is a deprecated non-/v1 API path."""
        if any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            return False
        return path != "/" and not path.startswith("/v1")

    def _log_deprecation_warning(self, path: str) -> None:
        """Log a warning for deprecated path usage (once per unique path)."""
        if path not in self._warned_paths:
            self._warned_paths.add(path)
            logger.warning(
                "Deprecated API path accessed: %s. Migrate to /v1%s before %s.",
                path,
                path,
                self.sunset_date,
            )
