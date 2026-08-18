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
import time

DEFAULT_REPO = "tenstorrent/tt-shield"

# Job-log downloads (gh api .../jobs/<id>/logs) 302-redirect to blob storage and
# occasionally fail transiently. Retry a few times with linear backoff before
# giving up, so a one-off hiccup does not silently drop an image.
LOG_FETCH_ATTEMPTS = 4
LOG_FETCH_BACKOFF_SECONDS = 3

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
    """All jobs of a run as [{id, name, conclusion}, ...]."""
    out = run_gh(
        [
            "api",
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--jq",
            ".jobs[] | {id, name, conclusion}",
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


def job_log(repo: str, job_id: int) -> tuple[str | None, str | None]:
    """Download a job's log via gh, retrying transient failures.

    Returns (log_text, None) on success, or (None, error_message) if every
    attempt failed. Never silently returns an empty result — a persistent
    failure is reported so the caller can fail loudly.
    """
    last_err = ""
    for attempt in range(1, LOG_FETCH_ATTEMPTS + 1):
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/jobs/{job_id}/logs"],
            capture_output=True,
        )
        if proc.returncode == 0:
            return proc.stdout.decode(errors="replace"), None
        last_err = (
            proc.stderr.decode(errors="replace").strip() or f"exit {proc.returncode}"
        )
        print(
            f"WARNING: log fetch for job {job_id} failed "
            f"(attempt {attempt}/{LOG_FETCH_ATTEMPTS}): {last_err}",
            file=sys.stderr,
        )
        if attempt < LOG_FETCH_ATTEMPTS:
            time.sleep(LOG_FETCH_BACKOFF_SECONDS * attempt)
    return None, last_err


def extract_dev_image_tag(log_text: str) -> str | None:
    """First `dev[-_]image[-_]tag=<ghcr ref>` value in the log, or None."""
    m = _TAG_RE.search(_ANSI_RE.sub("", log_text))
    return m.group(1) if m else None


def resolve_source_images(repo: str, run_id: str) -> dict[str, dict]:
    """Resolve each family's source dev image from the run's build jobs.

    Returns {family: {image, build_job, job_id, conclusion, log_fetch_error}}.
    `image` is None when: no build job matched, the build job did not succeed,
    or (an error condition) the log could not be fetched / held no tag.
    `log_fetch_error` is set only when the log download failed after retries.
    """
    result = {
        fam: {
            "image": None,
            "build_job": None,
            "job_id": None,
            "conclusion": None,
            "log_fetch_error": None,
        }
        for fam in FAMILIES
    }
    for job in list_jobs(repo, run_id):
        fam = job_family(job.get("name", ""))
        # First matching build job per family wins; skip non-build jobs and
        # families already handled.
        if fam is None or result[fam]["job_id"] is not None:
            continue
        result[fam]["build_job"] = job.get("name")
        result[fam]["job_id"] = job["id"]
        result[fam]["conclusion"] = job.get("conclusion")
        # Only a successful build produces a publishable image; don't spend
        # retries downloading logs for skipped/failed/cancelled builds.
        if job.get("conclusion") != "success":
            continue
        log, err = job_log(repo, job["id"])
        if log is None:
            result[fam]["log_fetch_error"] = err
            continue
        result[fam]["image"] = extract_dev_image_tag(log)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve the vllm/media/forge source dev images built by a tt-shield Release run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-id", required=True, help="tt-shield Release Actions run ID")
    ap.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})"
    )
    ap.add_argument(
        "--json", action="store_true", help="Emit JSON instead of human-readable text"
    )
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
    else:
        print(f"Repo:   {args.repo}")
        print(f"Run:    {args.run_id}")
        print("Source dev images by engine family:")
        for fam in FAMILIES:
            image = result[fam]["image"]
            print(f"  {fam:6s}: {image if image else 'None'}")

    # Fail loudly: a build job that SUCCEEDED but produced no resolvable image is
    # an error (log fetch failed after retries, or the tag was missing from the
    # log) — never let that pass as a silent "None". Families with no build job,
    # or a skipped/failed/cancelled build, legitimately have no image.
    errors = []
    for fam in FAMILIES:
        r = result[fam]
        if r["job_id"] is None or r["image"] or r["conclusion"] != "success":
            continue
        reason = (
            f"log fetch failed after {LOG_FETCH_ATTEMPTS} attempts ({r['log_fetch_error']})"
            if r["log_fetch_error"]
            else "no dev image tag found in the job log"
        )
        errors.append(
            f"{fam}: build job '{r['build_job']}' (id {r['job_id']}) succeeded "
            f"but no image could be resolved — {reason}."
        )

    if errors:
        print(
            "\nERROR: could not resolve source image(s) for successful build job(s):",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
