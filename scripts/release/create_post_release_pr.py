#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""
create_post_release_pr.py
=========================

Open a DRAFT "Post release v<version>" pull request from the post-release branch
into ``main``, with a body in the exact format of
https://github.com/tenstorrent/tt-inference-server/pull/4398.

The body has four sections:

  # Summary of Changes           -> header + placeholder (filled in manually)
  # SW versions recommended ...   -> static (tt-smi / Firmware / tt-kmd)
  # Model Spec Release Updates    -> AUTO-GENERATED table (the tricky part)
  # Release Artifacts Summary     -> header + placeholder (filled in manually)

The "Model Spec Release Updates" table has one row per (model, device) release
combo taken from ``.github/workflows/models-ci-config.json`` (the ``ci.release``
entries), matched to its ``workflows/model_specs/prod/*.yaml`` block. Columns:

  Impl                   <- prod block ``impl:``
  Model Arch             <- the models-ci-config.json model key
  Weights                <- prod block ``weights:`` (all, <br>-joined)
  Devices                <- the release device from models-ci-config.json
  TT-Metal Commit Change <- old->new (old = base branch's prod; new = this branch)
                            "`old` -> `new`" if changed, else "`new`"
  Status Change          <- "No change [STATUS]" if unchanged, else the new STATUS
                            (a newly-added model/device shows the new status alone)
  CI Job Link            <- [CI Link](.../runs/<run-id>/job/<job-id>) resolved from
                            the tt-shield run's ``run-release-<model>-...-<device>`` job

Any value that cannot be computed is rendered as ``UNKNOWN``.

"old" vs "new" is a pure catalogue diff: ``new`` = this (post-release) branch's
working-tree prod, ``old`` = the base branch's prod (``--base-ref``, default
``origin/main``) — i.e. exactly what the PR itself changes.

CI links need read access to tenstorrent/tt-shield Actions. The token is taken
from --token or, failing that, the env vars TMP_VCANKOVIC_SHIELD_CRANE_PAT / GH_PAT /
GITHUB_TOKEN. If no token works (or --tt-shield-run-id is omitted), CI Job Link
cells fall back to UNKNOWN.

The PR is created with ``gh pr create --draft`` (gh uses its own auth /
GH_TOKEN / GITHUB_TOKEN; this must be able to open PRs on --repo). Use --dry-run
to print the body without opening a PR.

Usage (typically the final step of the release-automation pipeline, run while
checked out on the post-release branch):

    python3 scripts/release/create_post_release_pr.py \
        --tt-shield-run-id 29037835062

    # preview only, no PR:
    python3 scripts/release/create_post_release_pr.py --tt-shield-run-id ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from workflows.workflow_types import DeviceTypes, InferenceEngine  # noqa: E402

DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_PROD_DIR = REPO_ROOT / "workflows" / "model_specs" / "prod"
DEFAULT_VERSION_FILE = REPO_ROOT / "VERSION"
DEFAULT_REPO = "tenstorrent/tt-inference-server"
DEFAULT_TT_SHIELD_REPO = "tenstorrent/tt-shield"
TOKEN_ENV_VARS = ("TMP_VCANKOVIC_SHIELD_CRANE_PAT", "GH_PAT", "GITHUB_TOKEN")
UNKNOWN = "UNKNOWN"

STATIC_SW_VERSIONS = (
    "# SW versions recommended for Wormhole Galaxy:\n\n"
    "- tt-smi: 4.0.0\n"
    "- Firmware: 19.8.1\n"
    "- tt-kmd: 2.7.0\n"
)


# ---------------------------------------------------------------------------
# release scope (mirrors promote_dev_spec_to_prod.py)
# ---------------------------------------------------------------------------
def iter_implementations(model_entry: dict):
    """Yield each implementation dict (flat or implementations:[...] shape)."""
    if "implementations" in model_entry:
        yield from model_entry["implementations"]
    else:
        yield model_entry


def model_name_from_weight(weight: str) -> str:
    return Path(weight).name


def collect_release_combos(
    ci_config: dict,
) -> list[tuple[str, InferenceEngine, DeviceTypes]]:
    """Ordered, de-duplicated list of (model_name, engine, device) marked release."""
    combos: list[tuple[str, InferenceEngine, DeviceTypes]] = []
    seen: set = set()
    for model_name, entry in ci_config.get("models", {}).items():
        for impl in iter_implementations(entry):
            release = impl.get("ci", {}).get("release")
            if not release:
                continue
            try:
                engine = InferenceEngine.from_string(impl["inference_engine"])
            except (ValueError, KeyError):
                engine = None
            for device in release.get("devices", []):
                try:
                    dev = DeviceTypes.from_string(device)
                except ValueError:
                    continue
                key = (model_name, engine, dev)
                if key not in seen:
                    seen.add(key)
                    combos.append((model_name, engine, dev))
    return combos


# ---------------------------------------------------------------------------
# prod catalogue parsing
# ---------------------------------------------------------------------------
def _parse_catalogue(text: str) -> list[dict]:
    data = yaml.safe_load(text)
    if isinstance(data, dict):
        data = data.get("templates") or next(
            (v for v in data.values() if isinstance(v, list)), []
        )
    return [b for b in (data or []) if isinstance(b, dict)]


def load_prod_blocks_from_dir(prod_dir: Path) -> list[dict]:
    blocks: list[dict] = []
    for f in sorted(Path(prod_dir).glob("*.yaml")):
        blocks += _parse_catalogue(f.read_text())
    return blocks


def load_prod_blocks_from_ref(ref: str, prod_filenames: list[str]) -> list[dict]:
    blocks: list[dict] = []
    for name in prod_filenames:
        try:
            text = subprocess.run(
                ["git", "show", f"{ref}:workflows/model_specs/prod/{name}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except subprocess.CalledProcessError:
            continue  # file absent on that ref
        blocks += _parse_catalogue(text)
    return blocks


def _block_engine(block: dict):
    try:
        return InferenceEngine.from_string(block.get("inference_engine", ""))
    except (ValueError, KeyError):
        return None


def _block_devices(block: dict) -> set:
    out = set()
    for d in block.get("device_model_specs", []) or []:
        try:
            out.add(DeviceTypes.from_string(d.get("device", "")))
        except ValueError:
            pass
    return out


def _block_models(block: dict) -> set:
    return {model_name_from_weight(w) for w in block.get("weights", []) or []}


def find_block(blocks, model_name, engine, device) -> dict | None:
    """First block that provides (model_name, engine) on `device`.

    Matches by device MEMBERSHIP (not the exact device-set), so a block that
    bundles extra devices still matches the specific release device.
    """
    for b in blocks:
        if (
            model_name in _block_models(b)
            and _block_engine(b) == engine
            and device in _block_devices(b)
        ):
            return b
    return None


# ---------------------------------------------------------------------------
# CI job links (tenstorrent/tt-shield)
# ---------------------------------------------------------------------------
def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for name in TOKEN_ENV_VARS:
        val = os.environ.get(name)
        if val:
            return val
    return None


def fetch_run_jobs(repo: str, run_id: str, token: str) -> list[dict] | None:
    """All jobs of a workflow run, or None if unreachable (no access/expired)."""
    owner_repo = repo
    jobs: list[dict] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{owner_repo}/actions/runs/{run_id}"
            f"/jobs?per_page=100&page={page}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "create-post-release-pr",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None
        batch = data.get("jobs", [])
        jobs += batch
        if len(batch) < 100 or page >= 15:
            break
        page += 1
    return jobs


def ci_job_url(
    jobs, repo: str, run_id: str, model_name: str, device: DeviceTypes
) -> str | None:
    """URL of the run-release-<model>-<runner>-<device> job for this combo."""
    if not jobs:
        return None
    token = device.name.lower()  # device token used in tt-shield job names
    prefix = f"run-release-{model_name}-"
    for job in jobs:
        leaf = job.get("name", "").split("/")[-1].strip()
        if leaf.startswith(prefix) and leaf.endswith(f"-{token}"):
            return f"https://github.com/{repo}/actions/runs/{run_id}/job/{job['id']}"
    return None


# ---------------------------------------------------------------------------
# row model + rendering
# ---------------------------------------------------------------------------
def build_rows(new_blocks, old_blocks, combos, jobs, tt_shield_repo, run_id):
    rows = []
    seen: set = set()
    for model_name, engine, device in combos:
        new_b = find_block(new_blocks, model_name, engine, device)
        old_b = find_block(old_blocks, model_name, engine, device)

        # De-duplicate: one row per (prod block weights+engine, device). A block
        # bundling several weights (e.g. whisper) yields a single row.
        ident_block = new_b or old_b
        weights = tuple((ident_block or {}).get("weights", []) or [])
        dedup_key = (weights, engine, device)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        rows.append(
            {
                "impl": (new_b or old_b or {}).get("impl"),
                "model_arch": model_name,
                "weights": list(weights),
                "device": device,
                "tt_before": (old_b or {}).get("tt_metal_commit"),
                "tt_after": (new_b or {}).get("tt_metal_commit"),
                "status_before": (old_b or {}).get("status"),
                "status_after": (new_b or {}).get("status"),
                "ci_url": ci_job_url(jobs, tt_shield_repo, run_id, model_name, device)
                if (jobs and run_id)
                else None,
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
        device = r["device"].name if r["device"] else UNKNOWN
        commit = _commit_cell(r["tt_before"], r["tt_after"])
        status = _status_cell(r["status_before"], r["status_after"])
        ci = f"[CI Link]({r['ci_url']})" if r["ci_url"] else UNKNOWN
        lines.append(
            f"| {impl} | {arch} | {weights} | {device} | {commit} | {status} | {ci} |"
        )
    return "\n".join(lines)


def render_body(version: str, rows) -> str:
    return (
        "# Summary of Changes\n\n"
        "<!-- Fill in the summary of changes manually. -->\n"
        "- placeholder\n\n\n"
        + STATIC_SW_VERSIONS
        + "\n"
        + render_table(rows)
        + "\n\n\n# Release Artifacts Summary\n\n"
        "## Images Promoted from Models CI\n\n"
        "<!-- Add promoted image paths manually. -->\n\n"
        "**Total:** \n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def read_version(version_file: Path) -> str:
    return version_file.read_text().strip()


def current_branch() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def create_pr(repo, base, head, title, body) -> None:
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
    args = ap.parse_args()

    version = args.version or read_version(args.version_file)
    head_branch = args.head_branch or current_branch()
    title = f"Post release v{version}"

    ci_config = json.loads(args.ci_config.read_text())
    combos = collect_release_combos(ci_config)

    new_blocks = load_prod_blocks_from_dir(args.prod_dir)
    prod_filenames = [f.name for f in sorted(Path(args.prod_dir).glob("*.yaml"))]
    old_blocks = load_prod_blocks_from_ref(args.base_ref, prod_filenames)

    # CI jobs (best-effort; UNKNOWN on any failure).
    jobs = None
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

    rows = build_rows(
        new_blocks, old_blocks, combos, jobs, args.tt_shield_repo, args.tt_shield_run_id
    )
    body = render_body(version, rows)

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
