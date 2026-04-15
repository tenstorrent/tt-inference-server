# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Validate that resources/openapi.json matches routes registered with ADD_METHOD_TO macros.

Usage:
    python3 tools/validate_openapi.py

Run from the cpp_server directory. Exits 0 if the spec and code agree, 1 if they diverge.
"""

import json
import re
import sys
from pathlib import Path

# Routes that are registered in code but intentionally omitted from the spec
# (infrastructure / UI endpoints that callers don't need to know about).
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


def extractCodeRoutes() -> set[tuple[str, str]]:
    """Return {(METHOD, path)} from all ADD_METHOD_TO macros in include/ and src/."""
    source_text = ""
    for directory in ("include", "src"):
        for filepath in (CPP_SERVER_ROOT / directory).rglob("*"):
            if filepath.suffix in (".hpp", ".cpp", ".h"):
                source_text += filepath.read_text(errors="replace") + "\n"

    # Collapse continuation lines so a multi-line macro becomes one line.
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
    spec_path = CPP_SERVER_ROOT / "resources" / "openapi.json"
    with spec_path.open() as f:
        spec = json.load(f)

    routes: set[tuple[str, str]] = set()
    for path, methods in spec.get("paths", {}).items():
        for method in methods:
            routes.add((method.upper(), path))

    return routes


def main() -> int:
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
        routes_list = sorted(checkable_code)
        print(f"OK: {len(routes_list)} route(s) match between code and openapi.json")
        for method, path in routes_list:
            print(f"  {method:8s} {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
