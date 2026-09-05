#!/usr/bin/env python3
"""Resolve the required tt-metal commit for a model, from a spec/identity source.

The wrapper needs one thing: the immutable 40-hex tt-metal commit a given model's serve
must run against. This resolves it from, in priority order:

  1. an explicit 40-hex sha argument;
  2. a tt-metal tree path -> its .ttq-runtime-identity.json "base_revision" (else git HEAD);
  3. a JSON model-spec file carrying one of the known keys
     (tt_metal_commit / ttmetal_commit / base_revision / QUETZAL_REQUIRED_SOURCE_REVISION
      under top-level, "impl", or "device_model_spec");
  4. the env var QUETZAL_REQUIRED_SOURCE_REVISION / TT_QUETZAL_COMMIT_SHA.

Prints the resolved 40-hex commit to stdout, or exits nonzero with a message on stderr.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
KEYS = (
    "tt_metal_commit",
    "ttmetal_commit",
    "base_revision",
    "QUETZAL_REQUIRED_SOURCE_REVISION",
    "tt_metal_source_revision",
)


def _from_mapping(obj: dict) -> str | None:
    scopes = [obj]
    for k in ("impl", "device_model_spec", "runtime", "tt_metal"):
        v = obj.get(k)
        if isinstance(v, dict):
            scopes.append(v)
    for scope in scopes:
        for key in KEYS:
            val = scope.get(key)
            if isinstance(val, str) and SHA_RE.match(val):
                return val
    return None


def resolve(arg: str) -> str | None:
    if SHA_RE.match(arg):
        return arg
    p = Path(arg)
    # a tt-metal tree
    if p.is_dir():
        ident = p / ".ttq-runtime-identity.json"
        if ident.is_file():
            try:
                rev = json.loads(ident.read_text()).get("base_revision", "")
                if isinstance(rev, str) and SHA_RE.match(rev):
                    return rev
            except (OSError, ValueError):
                pass
        try:
            head = subprocess.check_output(
                ["git", "-C", str(p), "rev-parse", "HEAD"], text=True
            ).strip()
            if SHA_RE.match(head):
                return head
        except (OSError, subprocess.CalledProcessError):
            pass
    # a json spec file
    if p.is_file():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                got = _from_mapping(data)
                if got:
                    return got
        except (OSError, ValueError):
            pass
    return None


def main(argv: list[str]) -> int:
    arg = argv[1] if len(argv) > 1 else ""
    got = resolve(arg) if arg else None
    if not got:
        for env in ("QUETZAL_REQUIRED_SOURCE_REVISION", "TT_QUETZAL_COMMIT_SHA"):
            v = os.getenv(env, "")
            if SHA_RE.match(v):
                got = v
                break
    if not got:
        print(f"resolve_commit: could not resolve a 40-hex tt-metal commit from {arg!r}",
              file=sys.stderr)
        return 1
    print(got)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
