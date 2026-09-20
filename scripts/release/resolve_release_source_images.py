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
import os
import re
import subprocess
import sys
import time

DEFAULT_REPO = "tenstorrent/tt-shield"

# Job-log downloads (.../jobs/<id>/logs) 302-redirect to blob storage and can
# fail transiently. Retry a few times with linear backoff before giving up, so
# a one-off hiccup does not drop an image. Fetched with curl (not `gh api`):
# newer gh refuses to emit logs containing ANSI escape sequences unless
# --allow-escape-sequences is passed, a flag older gh lacks — curl has no such
# guard, so this works the same locally and on the runner.
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


def _gh_token() -> str | None:
    """The token gh is configured to use (works locally and in CI where the
    step sets GH_TOKEN), with env fallbacks. Used for the curl log download."""
    proc = subprocess.run(["gh", "auth", "token"], capture_output=True)
    if proc.returncode == 0:
        tok = proc.stdout.decode(errors="replace").strip()
        if tok:
            return tok
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_PAT"):
        val = os.environ.get(name)
        if val:
            return val
    return None


def job_log(repo: str, job_id: int) -> tuple[str | None, str | None]:
    """Download a job's log via curl, retrying transient failures.

    Returns (log_text, None) on success, or (None, error_message) if every
    attempt failed. Never silently returns an empty result — a persistent
    failure is reported so the caller can fail loudly. curl (not `gh api`) is
    used to avoid gh's escape-sequence guard on log output; `curl -L` follows
    the redirect to blob storage and drops the auth header cross-host.
    """
    token = _gh_token()
    if not token:
        return None, "no GitHub token available (gh auth token / GH_TOKEN)"
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    last_err = ""
    for attempt in range(1, LOG_FETCH_ATTEMPTS + 1):
        proc = subprocess.run(
            [
                "curl",
                "-fsSL",
                "-H",
                f"Authorization: Bearer {token}",
                "-H",
                "Accept: application/vnd.github+json",
                "-H",
                "X-GitHub-Api-Version: 2022-11-28",
                url,
            ],
            capture_output=True,
        )
        if proc.returncode == 0:
            return proc.stdout.decode(errors="replace"), None
        last_err = (
            proc.stderr.decode(errors="replace").strip()
            or f"curl exit {proc.returncode}"
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


_HEX40_RE = re.compile(r"[0-9a-f]{40}")


def _parse_tag_commits(image_ref: str | None) -> dict:
    """Best-effort extract {tt_metal, vllm} from a tt-shield dev image ref. Tag shapes:
      vllm/media: <ver>-<ttmetal40>-<vllm|other>-<jobid>
      forge:      <ttmetal40>_<other>_<jobid>   (underscores, no version)
    Only the vLLM family's `vllm` slot is meaningful (media/forge's 3rd token is a
    different commit and is never read by the validator). The tag's VERSION is
    intentionally NOT parsed: source images carry the PREVIOUS release version,
    which never equals the release being cut — version is validated only against
    the VERSION file (in the workflow's Resolve-inputs step), never the image."""
    if not image_ref or image_ref == "None":
        return {}
    tag = image_ref.rsplit(":", 1)[-1]
    out: dict = {}
    m = _HEX40_RE.search(tag)
    if m:
        out["tt_metal"] = m.group(0)
        nxt = re.split(r"[-_]", tag[m.end() :].lstrip("-_"), maxsplit=1)[0]
        if nxt:
            out["vllm"] = nxt
    return out


def validate_expected(result: dict[str, dict], expect: dict[str, str]) -> list[str]:
    """Mismatches between the expected inputs and the commits the tt-shield run
    actually baked. `expect` keys: ttm, vllm. tt-metal is a prefix match (input is
    short, embedded SHA is 40-char); vllm is a prefix either way. VERSION is NOT
    checked against the image — source images carry the previous release version;
    the version input is validated only against the VERSION file, in the workflow."""
    errors: list[str] = []
    parsed = {fam: _parse_tag_commits(result[fam]["image"]) for fam in FAMILIES}

    want = expect.get("ttm")
    if want:
        ttms = {fam: p["tt_metal"] for fam, p in parsed.items() if p.get("tt_metal")}
        if not ttms:
            errors.append(
                f"tt-metal-commit '{want}': no built image found to verify against"
            )
        for fam, full in ttms.items():
            if not full.startswith(want.lower()):
                errors.append(
                    f"tt-metal-commit '{want}' does not match the {fam} image (built {full[:12]}…)"
                )

    want = expect.get("vllm")
    if want:
        vref = result["vllm"]["image"]
        got = (parsed["vllm"].get("vllm") or "").lower()
        w = want.lower()
        if not vref or vref == "None":
            errors.append(
                f"vllm-commit '{want}': no vLLM image was built to verify against"
            )
        elif not (got.startswith(w) or w.startswith(got)):
            errors.append(
                f"vllm-commit '{want}' does not match the vLLM image (built '{got}')"
            )

    return errors


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
    ap.add_argument(
        "--expect",
        default=None,
        help="Validate the run's built images against expected commits, e.g. "
        "'ttm=<sha>,vllm=<sha>'. Empty values are ignored. Exits "
        "non-zero on any mismatch (skips the JSON/text report).",
    )
    args = ap.parse_args()

    result = resolve_source_images(args.repo, args.run_id)

    if args.expect:
        expect = {
            k: v
            for kv in args.expect.split(",")
            if "=" in kv
            for k, v in [kv.split("=", 1)]
            if v.strip()
        }
        errs = validate_expected(result, expect)
        for e in errs:
            print(f"::error::{e}", file=sys.stderr)
        if errs:
            sys.exit(f"{len(errs)} commit mismatch(es) vs tt-shield run {args.run_id}")
        print(f"OK - inputs match the tt-shield run's built images: {expect}")
        return

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
