#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
build_release_manifest.py
=========================

Emit a release manifest -- the machine-readable, per-version record of what a
release published (see ``release_manifest.py`` for the schema and rationale).

The ``live`` subcommand builds the manifest for the release being cut, from the
same evidence the post-release PR uses (release scope x promoted prod x prod
base x the tt-shield Release run). This is what the release tooling calls:

    python3 scripts/release/build_release_manifest.py live \
        --version 0.22.0 --tt-shield-run-id 26592936143

By default the manifest is written to ``workflows/release_manifests/v<version>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.release import release_manifest as rm  # noqa: E402

DEFAULT_MANIFEST_DIR = REPO_ROOT / "workflows" / "release_manifests"
DEFAULT_VERSION_FILE = REPO_ROOT / "VERSION"


def _default_output(version: str) -> Path:
    return DEFAULT_MANIFEST_DIR / f"{rm.with_v(version)}.json"


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------
def cmd_live(args) -> None:
    from scripts.release.create_post_release_pr import (
        build_rows,
        fetch_run_jobs,
        resolve_release_scope,
        resolve_token,
    )
    from scripts.release.release_scope import (
        load_prod_leaves,
        load_prod_leaves_from_ref,
    )

    version = rm.strip_v(args.version or DEFAULT_VERSION_FILE.read_text().strip())

    jobs = None
    if args.tt_shield_run_id:
        token = resolve_token(args.token)
        if token:
            jobs = fetch_run_jobs(args.tt_shield_repo, args.tt_shield_run_id, token)
            if jobs is None:
                print(
                    "WARNING: could not read tt-shield jobs; ci_job_url will be null.",
                    file=sys.stderr,
                )
        else:
            print("WARNING: no token; ci_job_url will be null.", file=sys.stderr)
    else:
        print(
            "WARNING: --tt-shield-run-id not given; ci_job_url will be null.",
            file=sys.stderr,
        )

    try:
        ci_config = json.loads(Path(args.ci_config).read_text())
        scope = resolve_release_scope(ci_config, Path(args.dev_dir))
        current_prod = load_prod_leaves(Path(args.prod_dir))
        base_prod = load_prod_leaves_from_ref(args.base_ref)
        rows = build_rows(
            scope,
            current_prod,
            base_prod,
            jobs,
            args.tt_shield_repo,
            args.tt_shield_run_id,
            version,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"ERROR: {exc}")

    manifest = rm.build_live_manifest(
        version=version,
        run_id=args.tt_shield_run_id,
        rows=rows,
        current_prod=current_prod,
        generated=args.generated,
    )
    out = args.output or _default_output(version)
    rm.write_manifest(manifest, out)
    print(
        f"{rm.with_v(version)}: {len(manifest['changed'])} released leaves "
        f"-> {out}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emit a release manifest (live or reconstructed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--generated",
        default=None,
        help="Value for the manifest 'generated' field (e.g. an ISO date). "
        "Passed in rather than read from the clock so output is reproducible.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    live = sub.add_parser("live", help="build the current release's manifest")
    live.add_argument("--version", default=None, help="release version (default: VERSION)")
    live.add_argument("--tt-shield-run-id", default=None)
    live.add_argument("--tt-shield-repo", default="tenstorrent/tt-shield")
    live.add_argument("--token", default=None)
    live.add_argument(
        "--ci-config",
        type=Path,
        default=REPO_ROOT / ".github" / "workflows" / "models-ci-config.json",
    )
    live.add_argument("--dev-dir", type=Path, default=REPO_ROOT / "workflows" / "model_specs" / "dev")
    live.add_argument("--prod-dir", type=Path, default=REPO_ROOT / "workflows" / "model_specs" / "prod")
    live.add_argument("--base-ref", default="origin/main")
    live.add_argument("--output", type=Path, default=None)
    live.set_defaults(func=cmd_live)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
