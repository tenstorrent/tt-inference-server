#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""
create_post_release_pr.py
=========================

Open a DRAFT "Post release v<version>" PR from the post-release branch into ``main``

The body has four sections:

  # Summary of Changes           -> header + placeholder (filled in manually)
  # SW versions recommended ...   -> static (tt-smi / Firmware / tt-kmd)
  # Model Spec Release Updates    -> AUTO-GENERATED table (the tricky part)
  # Release Artifacts Summary     -> auto-listed promoted images + Total (from --promoted-images)

The "Model Spec Release Updates" table has one row per released runtime leaf --
the (hf_model_repo, device, engine, impl_id) tuple that a release actually
publishes. Each ``ci.release`` entry in ``.github/workflows/models-ci-config.json``
is resolved against the dev catalogue to exactly one such leaf, and that leaf is
then looked up in ``workflows/model_specs/prod/*.yaml`` by identity. Columns:

  Impl                   <- impl_id of the released leaf
  Model Arch             <- weight basename (the models-ci-config.json model key)
  Weights                <- the released weight
  Devices                <- the released device
  TT-Metal Commit Change <- old->new (old = base branch's prod; new = this branch)
                            "`old` -> `new`" if changed, else "`new`"
  Status Change          <- "No change [STATUS]" if unchanged, else the new STATUS
                            (a newly-added model/device shows the new status alone)
  CI Job Link            <- [CI Link](.../runs/<run-id>/job/<job-id>) resolved from
                            the tt-shield run's ``run-release-<model>-...-<device>`` job

Any value that cannot be computed is rendered as ``UNKNOWN``.

Resolution is by exact identity throughout: a selector that matches no leaf, or
more than one, aborts instead of silently reporting whichever prod block happened
to come first.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.release.model_spec_resolver import (  # noqa: E402
    collect_release_combos,
    load_dev_model_spec_sources,
    resolve_release_combos,
)
from scripts.release.release_scope import (  # noqa: E402
    UNKNOWN,
    load_prod_leaves,
    load_prod_leaves_from_ref,
)
from utils.model_naming import ci_job_matches_device  # noqa: E402
from workflows.workflow_types import DeviceTypes  # noqa: E402

DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_DEV_DIR = REPO_ROOT / "workflows" / "model_specs" / "dev"
DEFAULT_PROD_DIR = REPO_ROOT / "workflows" / "model_specs" / "prod"
DEFAULT_VERSION_FILE = REPO_ROOT / "VERSION"
DEFAULT_REPO = "tenstorrent/tt-inference-server"
DEFAULT_TT_SHIELD_REPO = "tenstorrent/tt-shield"
TOKEN_ENV_VARS = ("TMP_VCANKOVIC_SHIELD_CRANE_PAT", "GH_PAT", "GITHUB_TOKEN")

# Strips ANSI colour codes from fetched job logs before parsing.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _safe_repo_path(p) -> Path:
    """Resolve a supplied path and confine it to the repo root (guards path traversal)."""
    resolved = Path(p).expanduser().resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        sys.exit(f"ERROR: path '{resolved}' is outside the repository root {REPO_ROOT}")
    return resolved


def model_name_from_weight(weight: str) -> str:
    return Path(weight).name


# ---------------------------------------------------------------------------
# release scope (exact leaves, shared with promote_dev_spec_to_prod.py)
# ---------------------------------------------------------------------------
def resolve_release_scope(ci_config: dict, dev_dir: Path):
    # resolve_release_combos() rejects two selectors resolving to one identity.
    resolved = resolve_release_combos(
        collect_release_combos(ci_config),
        load_dev_model_spec_sources(dev_dir),
    )
    # A CI job name carries only repository and device, so two identities that
    # share that pair cannot be told apart when a row is linked to its job.
    # Check the whole scope here rather than while matching jobs: that path is
    # skipped whenever the GitHub API returns nothing, which would make an
    # ambiguous release scope pass or fail depending on the network.
    owner_by_repo_device = {}
    for item in resolved:
        repo_device = item.identity[:2]
        owner = owner_by_repo_device.get(repo_device)
        if owner is not None:
            raise ValueError(
                f"CI job names cannot distinguish release identities "
                f"{[owner, item.identity]!r}"
            )
        owner_by_repo_device[repo_device] = item.identity
    return tuple(resolved)


# ---------------------------------------------------------------------------
# CI job links (tenstorrent/tt-shield)
# ---------------------------------------------------------------------------
def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return next(
        (os.environ[name] for name in TOKEN_ENV_VARS if os.environ.get(name)),
        None,
    )


def fetch_run_jobs(repo: str, run_id: str, token: str) -> list[dict] | None:
    """All jobs of a workflow run, or None if unreachable (no access/expired)."""
    jobs: list[dict] = []
    for page in range(1, 16):
        url = (
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
            f"/jobs?per_page=100&page={page}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "create-post-release-pr",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                batch = json.loads(response.read()).get("jobs", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None
        jobs.extend(batch)
        if len(batch) < 100:
            break
    return jobs


def fetch_job_log(repo: str, job_id, token: str) -> str | None:
    """A single job's raw log text, or None if unreachable. urllib follows the
    logs endpoint's redirect to the signed blob URL automatically."""
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "create-post-release-pr",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _matching_ci_jobs(jobs, *, identity, scope_identities) -> list[dict]:
    if not jobs:
        return []
    other_repos = [candidate[0] for candidate in scope_identities]
    return [
        job
        for job in jobs
        if ci_job_matches_device(
            job.get("name", ""),
            "release",
            identity[0],
            identity[1],
            other_repos,
        )
    ]


# ---------------------------------------------------------------------------
# row model + rendering
# ---------------------------------------------------------------------------
def build_rows(scope, current_prod, base_prod, jobs, tt_shield_repo, run_id, version):
    identities = tuple(item.identity for item in scope)
    job_urls: dict = {}
    job_owners: dict = {}
    if jobs and run_id:
        for identity in identities:
            matches = _matching_ci_jobs(
                jobs, identity=identity, scope_identities=identities
            )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple CI jobs match release identity {identity!r}"
                )
            if not matches:
                job_urls[identity] = None
                continue
            job_id = matches[0]["id"]
            owner = job_owners.get(job_id)
            if owner is not None and owner != identity:
                raise ValueError(
                    f"CI job {job_id!r} ambiguously matches {owner!r} and {identity!r}"
                )
            job_owners[job_id] = identity
            job_urls[identity] = (
                f"https://github.com/{tt_shield_repo}/actions/runs/"
                f"{run_id}/job/{job_id}"
            )

    rows = []
    for item in scope:
        identity = item.identity
        # The release publishes this exact leaf, so prod must carry it at the
        # release version. A miss here means the promotion step did not run, ran
        # against a different scope, or wrote a different pin.
        current = current_prod.get(identity)
        if current is None:
            raise ValueError(f"Current prod is missing release identity {identity!r}")
        if current.pin.version != version:
            raise ValueError(
                f"Prod identity {identity!r} has version {current.pin.version!r}, "
                f"expected {version!r}"
            )
        before = base_prod.get(identity)
        rows.append(
            {
                "identity": identity,
                "impl": identity[3],
                "model_arch": model_name_from_weight(identity[0]),
                "weights": [identity[0]],
                "device": identity[1],
                "tt_before": before.pin.tt_metal_commit if before else None,
                "tt_after": current.pin.tt_metal_commit,
                "status_before": before.status if before else None,
                "status_after": current.status,
                "ci_url": job_urls.get(identity),
            }
        )
    return rows


def _commit_cell(before, after) -> str:
    if after and before:
        return f"`{before}` → `{after}`" if before != after else f"`{after}`"
    if after:
        return f"`{after}`"  # new model/device: new value only
    return UNKNOWN


def _status_cell(before, after) -> str:
    if not after:
        return UNKNOWN
    if before and before == after:
        return f"No change [{after}]"
    return f"{after}"  # changed, or newly added -> the new status alone


def _weights_cell(weights) -> str:
    return "<br>".join(f"`{w}`" for w in weights) if weights else UNKNOWN


def render_table(rows) -> str:
    lines = [
        "# Model Spec Release Updates\n",
        "\nThis document shows model specification updates.\n",
    ]
    if not rows:
        lines.append("\nNo model specification updates were detected.\n")
        return "\n".join(lines)
    lines.append(
        "| Impl | Model Arch | Weights | Devices | TT-Metal Commit Change | Status Change | CI Job Link |"
    )
    lines.append(
        "|------|------------|---------|---------|------------------------|---------------|-------------|"
    )
    for r in rows:
        impl = f"`{r['impl']}`" if r["impl"] else UNKNOWN
        arch = f"`{r['model_arch']}`" if r["model_arch"] else UNKNOWN
        weights = _weights_cell(r["weights"])
        device = r["device"] or UNKNOWN
        commit = _commit_cell(r["tt_before"], r["tt_after"])
        status = _status_cell(r["status_before"], r["status_after"])
        ci = f"[CI Link]({r['ci_url']})" if r["ci_url"] else UNKNOWN
        lines.append(
            f"| {impl} | {arch} | {weights} | {device} | {commit} | {status} | {ci} |"
        )
    return "\n".join(lines)


def _promoted_section(promoted_images) -> str:
    """Render the 'Images Promoted from Models CI' section: one bullet per
    destination image (https://-prefixed) followed by '**Total:** N'."""
    lines = ["## Images Promoted from Models CI", ""]
    for img in promoted_images:
        url = img if img.startswith(("http://", "https://")) else f"https://{img}"
        lines.append(f"- {url}")
    if promoted_images:
        lines.append("")
    lines.append(f"**Total:** {len(promoted_images)}")
    return "\n".join(lines) + "\n"


def _is_galaxy(device) -> bool:
    """True iff `device` is Wormhole GALAXY (BLACKHOLE_GALAXY is a distinct device)."""
    try:
        return DeviceTypes.from_string(device) == DeviceTypes.GALAXY
    except (ValueError, KeyError, TypeError):
        return False


def _release_has_galaxy(rows) -> bool:
    """True if any released spec targets Wormhole GALAXY (BLACKHOLE_GALAXY excluded)."""
    return any(_is_galaxy(r["device"]) for r in rows)


def _parse_galaxy_sw_versions(log_text: str) -> dict:
    """Extract {tt_smi, firmware, kmd} from a galaxy job's setup tt-smi JSON dump.

    The setup ("reset") step prints a pretty-printed JSON object on
    timestamp-prefixed log lines; strip the prefixes and parse the first
    top-level object that carries ``host_sw_vers``. Any field that cannot be
    read comes back as None (rendered as UNKNOWN).
      tt_smi   <- host_sw_vers.tt_smi
      firmware <- device_info[].firmwares.fw_bundle_version (uniform across chips)
      kmd      <- host_info.Driver "TT-KMD X.Y.Z" -> "X.Y.Z"
    """
    blank = {"tt_smi": None, "firmware": None, "kmd": None}
    lines = [
        re.sub(r"^[0-9T:.\-]+Z\s?", "", _ANSI_RE.sub("", raw.rstrip("\n")))
        for raw in log_text.splitlines()
    ]
    data = None
    for start in (i for i, line in enumerate(lines) if line == "{"):
        end = next((j for j in range(start + 1, len(lines)) if lines[j] == "}"), None)
        if end is None:
            continue
        try:
            candidate = json.loads("\n".join(lines[start : end + 1]))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "host_sw_vers" in candidate:
            data = candidate
            break
    if data is None:
        return blank

    tt_smi = (data.get("host_sw_vers") or {}).get("tt_smi") or None
    fw_versions = {
        (dev.get("firmwares") or {}).get("fw_bundle_version")
        for dev in (data.get("device_info") or [])
        if isinstance(dev, dict)
    }
    fw_versions.discard(None)
    firmware = sorted(fw_versions)[0] if fw_versions else None
    m = re.search(r"TT-KMD\s+(\S+)", (data.get("host_info") or {}).get("Driver") or "")
    kmd = m.group(1) if m else None
    return {"tt_smi": tt_smi, "firmware": firmware, "kmd": kmd}


def resolve_galaxy_sw_versions(rows, jobs, tt_shield_repo, run_id, token) -> dict:
    """SW versions from the release's GALAXY job setup log, or all-None if it
    cannot be read. Host-level values are identical across galaxy jobs on the
    same runner, so the first GALAXY release job is used. Best-effort: any
    failure yields None fields, which render as UNKNOWN."""
    blank = {"tt_smi": None, "firmware": None, "kmd": None}
    if not jobs or not run_id or not token:
        return blank
    identities = [r["identity"] for r in rows]
    galaxy = next((r["identity"] for r in rows if _is_galaxy(r["device"])), None)
    if galaxy is None:
        return blank
    matches = _matching_ci_jobs(jobs, identity=galaxy, scope_identities=identities)
    if not matches:
        return blank
    log = fetch_job_log(tt_shield_repo, matches[0]["id"], token)
    if not log:
        return blank
    return _parse_galaxy_sw_versions(log)


def _sw_versions_block(sw_versions) -> str:
    sw = sw_versions or {}
    return (
        "# SW versions recommended for Wormhole Galaxy:\n\n"
        f"- tt-smi: {sw.get('tt_smi') or UNKNOWN}\n"
        f"- Firmware: {sw.get('firmware') or UNKNOWN}\n"
        f"- tt-kmd: {sw.get('kmd') or UNKNOWN}\n"
    )


def render_body(version: str, run_id, rows, promoted_images, sw_versions=None) -> str:
    # The recommended Wormhole-Galaxy SW versions only apply when the release
    # ships a GALAXY model; omit the whole section otherwise (BLACKHOLE_GALAXY
    # does not trigger it). Values are read from the galaxy job's setup log;
    # any value that could not be read renders as UNKNOWN.
    sw_block = (
        (_sw_versions_block(sw_versions) + "\n") if _release_has_galaxy(rows) else ""
    )
    return (
        # Machine-readable metadata block (parsed by downstream tooling); keep
        # it first, before the Summary. run_id = tt-shield Release run id,
        # version = the release version with a leading 'v'.
        "<!--\n"
        f"metadata:run_id={run_id or ''}\n"
        f"metadata:version=v{version}\n"
        "-->\n\n"
        "# Summary of Changes\n\n"
        "<!-- Fill in the summary of changes manually. -->\n"
        "- placeholder\n\n\n"
        + sw_block
        + render_table(rows)
        + "\n\n\n# Release Artifacts Summary\n\n"
        + _promoted_section(promoted_images)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def read_version(version_file: Path) -> str:
    return _safe_repo_path(version_file).read_text().strip()


def current_branch() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def create_pr(repo, base, head, title, body) -> None:
    # Validate the values that flow into the gh command. The call is list-form
    # (not shell-interpreted), but validate anyway as defence-in-depth.
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo):
        sys.exit(f"ERROR: invalid repo '{repo}'")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", base) or not re.fullmatch(
        r"[A-Za-z0-9._/-]+", head
    ):
        sys.exit("ERROR: invalid branch ref for --base/--head")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(body)
        body_path = fh.name
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                repo,
                "--base",
                base,
                "--head",
                head,
                "--draft",
                "--title",
                title,
                "--body-file",
                body_path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.exit(f"ERROR: `gh pr create` failed:\n{proc.stderr}")
        print(proc.stdout.strip())
    finally:
        os.unlink(body_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Open a draft 'Post release v<version>' PR (post-release branch -> main).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--version", default=None, help="Release version (default: read VERSION file)"
    )
    ap.add_argument("--version-file", type=Path, default=DEFAULT_VERSION_FILE)
    ap.add_argument(
        "--tt-shield-run-id",
        default=None,
        help="tt-shield Release run id (for CI Job Links)",
    )
    ap.add_argument("--tt-shield-repo", default=DEFAULT_TT_SHIELD_REPO)
    ap.add_argument("--ci-config", type=Path, default=DEFAULT_CI_CONFIG)
    ap.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
        help="dev catalogue the release selectors are resolved against",
    )
    ap.add_argument(
        "--prod-dir",
        type=Path,
        default=DEFAULT_PROD_DIR,
        help="'new' prod catalogue (this branch's working tree)",
    )
    ap.add_argument(
        "--base-ref",
        default="origin/main",
        help="git ref for 'old' prod + PR base history (default: origin/main)",
    )
    ap.add_argument("--base", default="main", help="PR base branch")
    ap.add_argument(
        "--head-branch", default=None, help="PR head branch (default: current branch)"
    )
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Repo to open the PR on")
    ap.add_argument(
        "--token",
        default=None,
        help="Token for tt-shield reads (default: env TMP_VCANKOVIC_SHIELD_CRANE_PAT/GH_PAT/GITHUB_TOKEN)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the body; do not open a PR"
    )
    ap.add_argument(
        "--output", type=Path, default=None, help="Also write the body to this file"
    )
    ap.add_argument(
        "--promoted-images",
        default="",
        help="Whitespace/newline-separated destination image URLs to list under "
        "'Images Promoted from Models CI' (the release-scoped publish plan).",
    )
    args = ap.parse_args()

    # Destination images actually published (already release-scoped upstream);
    # split on any whitespace/newlines and drop blanks / None sentinels.
    promoted_images = [
        img for img in args.promoted_images.split() if img and img != "None"
    ]

    version = args.version or read_version(args.version_file)
    head_branch = args.head_branch or current_branch()
    title = f"Post release v{version}"

    # CI jobs (best-effort; UNKNOWN on any failure).
    jobs = None
    token = None
    if args.tt_shield_run_id:
        token = resolve_token(args.token)
        if token:
            jobs = fetch_run_jobs(args.tt_shield_repo, args.tt_shield_run_id, token)
            if jobs is None:
                print(
                    "WARNING: could not read tt-shield jobs (no access / expired); "
                    "CI Job Link cells will be UNKNOWN.",
                    file=sys.stderr,
                )
        else:
            print(
                "WARNING: no token (TMP_VCANKOVIC_SHIELD_CRANE_PAT/GH_PAT/GITHUB_TOKEN); "
                "CI Job Link cells will be UNKNOWN.",
                file=sys.stderr,
            )
    else:
        print(
            "WARNING: --tt-shield-run-id not given; CI Job Link cells will be UNKNOWN.",
            file=sys.stderr,
        )

    try:
        ci_config = json.loads(args.ci_config.read_text())
        scope = resolve_release_scope(ci_config, _safe_repo_path(args.dev_dir))
        current_prod = load_prod_leaves(_safe_repo_path(args.prod_dir))
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

    # Galaxy SW versions (best-effort; UNKNOWN per field on any failure). Only
    # rendered when the release ships a GALAXY model.
    sw_versions = resolve_galaxy_sw_versions(
        rows, jobs, args.tt_shield_repo, args.tt_shield_run_id, token
    )

    body = render_body(
        version, args.tt_shield_run_id, rows, promoted_images, sw_versions
    )

    print(f"Version:      {version}", file=sys.stderr)
    print(f"Title:        {title}", file=sys.stderr)
    print(f"Head -> Base: {head_branch} -> {args.base}", file=sys.stderr)
    print(f"Release rows: {len(rows)}", file=sys.stderr)

    if args.output:
        args.output.write_text(body)
        print(f"Wrote body to {args.output}", file=sys.stderr)

    if args.dry_run:
        print(body)
        return

    create_pr(args.repo, args.base, head_branch, title, body)


if __name__ == "__main__":
    main()
