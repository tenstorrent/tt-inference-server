# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Validate resources/openapi.json in two ways:

1. Schema validity  -- the spec is valid OpenAPI 3.x (via openapi-spec-validator).
2. Route coverage   -- every ADD_METHOD_TO route in C++ is documented in the spec
                       and every spec path has a matching C++ route.

Usage:
    pip install openapi-spec-validator
    python3 tools/validate_openapi.py

Run from the cpp_server directory. Exits 0 only if both checks pass.
"""

import json
import re
import sys
from pathlib import Path

try:
    from openapi_spec_validator import validate as validateOpenApiSpec
    from openapi_spec_validator.readers import read_from_filename

    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False

# Routes registered in code but intentionally absent from the spec
# (infrastructure / Swagger UI endpoints).
SPEC_ALLOWLIST = {
    ("GET", "/openapi.json"),
    ("GET", "/docs"),
    ("GET", "/swagger"),
}

DROGON_METHOD_MAP = {
    "drogon::Get": "GET",
    "drogon::Post": "POST",
    "drogon::Put": "PUT",
    "drogon::Delete": "DELETE",
    "drogon::Patch": "PATCH",
    "drogon::Head": "HEAD",
    "drogon::Options": "OPTIONS",
}

CPP_SERVER_ROOT = Path(__file__).parent.parent
SPEC_PATH = CPP_SERVER_ROOT / "resources" / "openapi.json"


def checkSpecValidity() -> bool:
    """Validate openapi.json against the OpenAPI 3.x JSON Schema."""
    if not HAS_VALIDATOR:
        print(
            "SKIP: openapi-spec-validator not installed "
            "(run: pip install openapi-spec-validator)"
        )
        return True

    try:
        spec_dict, _ = read_from_filename(str(SPEC_PATH))
        validateOpenApiSpec(spec_dict)
        print("OK: openapi.json is valid OpenAPI 3.x")
        return True
    except Exception as exc:
        print(f"ERROR: openapi.json schema validation failed:\n  {exc}")
        return False


def extractCodeRoutes() -> set[tuple[str, str]]:
    """Return {(METHOD, path)} from all ADD_METHOD_TO macros in include/ and src/."""
    source_text = ""
    for directory in ("include", "src"):
        for filepath in (CPP_SERVER_ROOT / directory).rglob("*"):
            if filepath.suffix in (".hpp", ".cpp", ".h"):
                source_text += filepath.read_text(errors="replace") + "\n"

    # Collapse backslash-continuations so a multi-line macro becomes one line.
    source_text = re.sub(r"\\\n\s*", " ", source_text)

    routes: set[tuple[str, str]] = set()
    for match in re.finditer(r"ADD_METHOD_TO\s*\(([^)]+)\)", source_text):
        args = [a.strip() for a in match.group(1).split(",")]
        if len(args) < 3:
            continue
        path = args[1].strip('"').strip("'")
        for raw_method, http_verb in DROGON_METHOD_MAP.items():
            if any(raw_method in arg for arg in args[2:]):
                routes.add((http_verb, path))

    return routes


def extractSpecRoutes() -> set[tuple[str, str]]:
    """Return {(METHOD, path)} from openapi.json paths section."""
    with SPEC_PATH.open() as f:
        spec = json.load(f)

    routes: set[tuple[str, str]] = set()
    for path, methods in spec.get("paths", {}).items():
        for method in methods:
            routes.add((method.upper(), path))

    return routes


def checkRouteCoverage() -> bool:
    code_routes = extractCodeRoutes()
    spec_routes = extractSpecRoutes()

    checkable_code = code_routes - SPEC_ALLOWLIST
    missing_from_spec = checkable_code - spec_routes
    stale_in_spec = spec_routes - code_routes

    ok = True

    if missing_from_spec:
        print("ERROR: routes registered in C++ but missing from openapi.json:")
        for method, path in sorted(missing_from_spec):
            print(f"  {method:8s} {path}")
        ok = False

    if stale_in_spec:
        print("ERROR: paths in openapi.json with no matching C++ route:")
        for method, path in sorted(stale_in_spec):
            print(f"  {method:8s} {path}")
        ok = False

    if ok:
        print(f"OK: {len(checkable_code)} route(s) match between code and openapi.json")
        for method, path in sorted(checkable_code):
            print(f"  {method:8s} {path}")

    return ok


def main() -> int:
    spec_ok = checkSpecValidity()
    routes_ok = checkRouteCoverage()
    return 0 if (spec_ok and routes_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
