# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Route authentication and shared vLLM media-policy configuration."""

import argparse
import hmac
import json
import os


_POLICY_ENV = "TT_INFERENCE_HTTP_AUTH_KEYS"
_MIDDLEWARE = "utils.inference_security.InferenceSecurityMiddleware"
# Only console-facing inference APIs are exposed by default. Utility routes
# such as /invocations and /tokenize require an explicit operator opt-in.
_DEFAULT_ROUTES = [
    "/v1/models",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/audio/speech",
]


def configure_inference_security(argv, *, no_auth=False):
    """Install the guard using the same effective keys as vLLM.

    Called after model defaults and passthrough CLI arguments are merged. The
    serialized keys are inherited by spawned API workers; missing configuration
    is an error, not an implicit anonymous-serving mode. Never log this value.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--api-key", nargs="+")
    parser.add_argument("--allowed-media-domains", nargs="+")
    # Match vLLM's FlexibleArgumentParser, including underscore-style flags
    # emitted by model-spec defaults. Do not normalize argument values.
    normalized = []
    for token in argv[1:]:
        if token.startswith("--"):
            name, separator, value = token.partition("=")
            token = name.replace("_", "-") + separator + value
        normalized.append(token)
    args, _ = parser.parse_known_args(normalized)
    keys = args.api_key or [os.environ.get("VLLM_API_KEY", "")]
    keys = [key for key in keys if key]
    if not keys and not no_auth:
        raise RuntimeError(
            "Inference authentication requires VLLM_API_KEY, JWT_SECRET, or "
            "--api-key. Use --no-auth only for isolated development."
        )
    from utils.secure_media import configure_media_policy

    configure_media_policy(argv, args.allowed_media_domains)
    os.environ[_POLICY_ENV] = json.dumps(keys)
    argv.extend(["--middleware", _MIDDLEWARE])


async def _error(send, status, code, message):
    body = json.dumps(
        {"error": {"type": "invalid_request_error", "code": code, "message": message}}
    ).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class InferenceSecurityMiddleware:
    """Pure ASGI middleware; response streams pass through without buffering."""

    def __init__(self, app):
        self.app = app
        raw = os.environ.get(_POLICY_ENV)
        if raw is None:
            raise RuntimeError("Inference HTTP security was not configured")
        keys = json.loads(raw)
        if not isinstance(keys, list) or any(
            not isinstance(key, str) or not key for key in keys
        ):
            raise RuntimeError("Invalid inference HTTP authentication configuration")
        self.keys = [key.encode() for key in keys]
        routes = json.loads(
            os.environ.get("TT_INFERENCE_ALLOWED_ROUTES", json.dumps(_DEFAULT_ROUTES))
        )
        if not isinstance(routes, list) or any(
            not isinstance(route, str)
            or not route.startswith("/")
            or "?" in route
            or "#" in route
            or "*" in route
            for route in routes
        ):
            raise RuntimeError(
                "TT_INFERENCE_ALLOWED_ROUTES must contain exact route paths"
            )
        self.routes = {route.rstrip("/") for route in routes}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            # This gateway exposes HTTP APIs only. Do not create an unauthenticated
            # alternate transport when a newer vLLM version adds a WebSocket route.
            return await send({"type": "websocket.close", "code": 1008})
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        root_path = scope.get("root_path", "").rstrip("/")
        if root_path and (path == root_path or path.startswith(root_path + "/")):
            path = path[len(root_path) :]
        path = path.rstrip("/")
        method = scope["method"]
        # Keep health checks, metrics scrapes and CORS preflights usable. All
        # other HTTP routes require authentication, including /invocations.
        public = method in ("GET", "HEAD") and path in ("/health", "/ping", "/metrics")
        if not public and path not in self.routes:
            return await _error(send, 404, "not_found", "Not found")
        public = public or method == "OPTIONS"
        if self.keys and not public:
            auth = [
                value
                for name, value in scope["headers"]
                if name.lower() == b"authorization"
            ]
            scheme, _, token = (
                auth[0].partition(b" ") if len(auth) == 1 else (b"", b"", b"")
            )
            if scheme.lower() != b"bearer" or not any(
                hmac.compare_digest(token, key) for key in self.keys
            ):
                return await _error(
                    send, 401, "invalid_api_key", "Invalid or missing API key"
                )

        # Media policy belongs in vLLM's shared connector, not in route-specific
        # request parsing. Forward bodies and streamed responses untouched.
        await self.app(scope, receive, send)
