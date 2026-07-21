#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""
resolve_release_source_images.py
================================

From a tenstorrent/tt-shield "Release" Actions run, resolve the three SOURCE dev
images produced by the run's build jobs, grouped by engine family:

    vllm   <- build job "build-tt-inference-server"
    media  <- build job "build-media-inference-server"
    forge  <- build job "build-forge-media-inference-server"

Usage
-----
    # human-readable
    python3 scripts/release/resolve_release_source_images.py --run-id 29037835062

    # machine-readable (for a later publish step to consume)
    python3 scripts/release/resolve_release_source_images.py --run-id 29037835062 --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

DEFAULT_REPO = "tenstorrent/tt-shield"

# Build-job caller segment  ->  engine family. Order does not matter; matched by
# exact segment equality so build-forge-media / build-blaze-media never collide
# with build-media.
BUILD_JOB_FAMILY = {
    "build-tt-inference-server": "vllm",
    "build-media-inference-server": "media",
    "build-forge-media-inference-server": "forge",
}
FAMILIES = ("vllm", "media", "forge")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# The build jobs write `dev-image-tag=<url>` (summary/output) and
# `dev_image_tag=<url>` (GITHUB_OUTPUT); accept either separator. Capture the
# ghcr.io reference up to the first quote / space / comma.
_TAG_RE = re.compile(r"dev[-_]image[-_]tag=\s*(ghcr\.io/[^\s\"',]+)")


def run_gh(args: list[str], binary: bool = False):
    """Run a `gh` command, exiting with its stderr on failure."""
    proc = subprocess.run(["gh", *args], capture_output=True)
    if proc.returncode != 0:
        sys.exit(
            f"ERROR: `gh {' '.join(args)}` failed (exit {proc.returncode}):\n"
            + proc.stderr.decode(errors="replace")
        )
    return proc.stdout if binary else proc.stdout.decode()


def list_jobs(repo: str, run_id: str) -> list[dict]:
    """All jobs of a run as [{id, name}, ...]."""
    out = run_gh(
        [
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--jq",
            ".jobs[] | {id, name}",
        ]
    )
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def job_family(job_name: str) -> str | None:
    """Return the engine family for a build job, or None if it isn't one.

    Matches any '/'-separated segment of the job name against BUILD_JOB_FAMILY
    by exact equality (so build-forge-media / build-blaze-media don't collide
    with build-media-inference-server)."""
    for seg in (s.strip() for s in job_name.split("/")):
        if seg in BUILD_JOB_FAMILY:
            return BUILD_JOB_FAMILY[seg]
    return None


def job_log(repo: str, job_id: int) -> str | None:
    """Download a job's log via gh (which follows the log redirect), or None."""
    try:
        return run_gh(["api", f"repos/{repo}/actions/jobs/{job_id}/logs"])
    except SystemExit:
        return None


def extract_dev_image_tag(log_text: str) -> str | None:
    """First `dev[-_]image[-_]tag=<ghcr ref>` value in the log, or None."""
    m = _TAG_RE.search(_ANSI_RE.sub("", log_text))
    return m.group(1) if m else None


def resolve_source_images(repo: str, run_id: str) -> dict[str, dict]:
    """Return {family: {"image": tag|None, "build_job": name|None, "job_id": id|None}}."""
    result = {fam: {"image": None, "build_job": None, "job_id": None} for fam in FAMILIES}
    for job in list_jobs(repo, run_id):
        fam = job_family(job.get("name", ""))
        if fam is None or result[fam]["image"] is not None:
            continue
        log = job_log(repo, job["id"])
        tag = extract_dev_image_tag(log) if log else None
        result[fam] = {"image": tag, "build_job": job.get("name"), "job_id": job["id"]}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve the vllm/media/forge source dev images built by a tt-shield Release run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-id", required=True, help="tt-shield Release Actions run ID")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text")
    args = ap.parse_args()

    result = resolve_source_images(args.repo, args.run_id)

    if args.json:
        # Compact {family: image|None} plus a detailed block for traceability.
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "repo": args.repo,
                    "images": {fam: result[fam]["image"] for fam in FAMILIES},
                    "detail": result,
                },
                indent=2,
            )
        )
        return

    print(f"Repo:   {args.repo}")
    print(f"Run:    {args.run_id}")
    print("Source dev images by engine family:")
    for fam in FAMILIES:
        image = result[fam]["image"]
        print(f"  {fam:6s}: {image if image else 'None'}")


if __name__ == "__main__":
    main()
