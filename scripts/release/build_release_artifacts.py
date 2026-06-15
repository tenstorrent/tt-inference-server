#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""
build_release_artifacts.py
==========================

Build a ``<version>-release_artifacts.zip`` from a tenstorrent/tt-shield
"Release" GitHub Actions run, for a chosen set of models and the devices you
want for each model.

What it produces (matches the v0.14.0 / v0.15.0 release package layout):

    <version>-release_artifacts.zip
    └── <version>-release_artifacts/                     (a single top-level folder)
        ├── workflow_logs_release_<model>_<device>.zip   (one per model+device)
        ├── workflow_logs_release_<model>_<device>.zip
        └── ...

Each inner ``workflow_logs_release_<model>_<device>.zip`` is the full GitHub
artifact bundle for that job (ai_summaries/, docker_server/, benchmarks_output/,
evals_output/, reports_output/{benchmarks,benchmarks_aiperf,evals,release}/,
run_logs/, runtime_model_specs/, sometimes spec_tests_output/). The outer zip
stores its entries uncompressed (ZIP_STORED), exactly like the previous release.

The tricky bit — runner label vs. device name
----------------------------------------------
GitHub names the artifacts after the *runner label*, with a ``_default`` suffix:

    workflow_logs_release_speecht5_tts_tt-ubuntu-2204-p150b-stable_default   (p150)
    workflow_logs_release_speecht5_tts_bh-qb-ge_default                      (p300x2)

The release package, however, names the inner zips after the *device*
(``p150``, ``p300x2``, ``t3k``, ...). This script bridges the two:

  1. PRIMARY: it reads the run's job names (``run-release-<model>-<runner>-<device>``)
     to learn which runner ran which device. No hardcoded runner table, so it
     keeps working if runner labels change.
  2. VERIFY / FALLBACK: after download it confirms the requested device token
     actually appears inside the bundle's own file names (the ground truth). If
     the job-name mapping can't resolve a device, it falls back to classifying
     bundles purely by that internal token.

Requirements: Python 3.8+, the GitHub CLI ``gh`` installed and authenticated
(``gh auth status``) with ``repo`` scope.

Usage
-----
Reproduce the v0.15.0 release (uses the embedded defaults below):

    python3 build_release_artifacts.py

Run for a new release / different models (CLI overrides the defaults):

    python3 build_release_artifacts.py \
        --run-id 26592936143 \
        --version v0.15.0 \
        --model speecht5_tts=p150,p300x2 \
        --model whisper-large-v3=p150,p300x2 \
        --model distil-large-v3=p150,p300x2 \
        --output-dir .

Other flags: --repo, --output-dir, --keep-temp, --strict (turn the internal
device-token check from a warning into a hard error), --reference-zip <prev.zip>
(sanity-check the output layout against a previous release package).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ARTIFACT_PREFIX = "workflow_logs_release_"
DEFAULT_REPO = "tenstorrent/tt-shield"

# ---------------------------------------------------------------------------
# Defaults — running the script with no CLI args reproduces this release.
# Edit these, or override any of them on the command line.
# ---------------------------------------------------------------------------
DEFAULT_VERSION = "v0.15.0"
DEFAULT_RUN_ID = "26592936143"
DEFAULT_MODELS: dict[str, list[str]] = {
    "speecht5_tts":     ["p150", "p300x2"],
    "whisper-large-v3": ["p150", "p300x2"],
    "distil-large-v3":  ["p150", "p300x2"],
}


# ---------------------------------------------------------------------------
# gh helpers
# ---------------------------------------------------------------------------
def run_gh(args: list[str], binary: bool = False):
    """Run a `gh` command, exiting with its stderr on failure."""
    proc = subprocess.run(["gh", *args], capture_output=True)
    if proc.returncode != 0:
        sys.exit(
            f"ERROR: `gh {' '.join(args)}` failed (exit {proc.returncode}):\n"
            + proc.stderr.decode(errors="replace")
        )
    return proc.stdout if binary else proc.stdout.decode()


def gh_json_lines(endpoint: str, jq: str) -> list[dict]:
    """Page through a REST endpoint and parse one compact JSON object per line."""
    out = run_gh(["api", "--paginate", endpoint, "--jq", jq])
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def list_artifacts(repo: str, run_id: str) -> list[dict]:
    return gh_json_lines(
        f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
        ".artifacts[] | {id, name, size_in_bytes, expired}",
    )


def list_jobs(repo: str, run_id: str) -> list[dict]:
    return gh_json_lines(
        f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
        ".jobs[] | {id, name}",
    )


# ---------------------------------------------------------------------------
# runner / device resolution
# ---------------------------------------------------------------------------
def runner_of(artifact_name: str, model: str) -> str:
    """workflow_logs_release_<model>_<runner>_<suffix>  ->  <runner>.

    Assumes a single-token suffix (e.g. ``default``). `model` is matched
    exactly, so models that share a prefix (foo vs foo-turbo) are unambiguous
    because of the underscore boundary after the model name.
    """
    rest = artifact_name[len(ARTIFACT_PREFIX) + len(model) + 1:]  # "<runner>_<suffix>"
    return rest.rsplit("_", 1)[0]


def device_from_jobs(jobs: list[dict], model: str, runner: str) -> str | None:
    """Find the device a (model, runner) pair ran on, from the job name
    pattern ``run-release-<model>-<runner>-<device>``."""
    marker = f"run-release-{model}-{runner}-"
    for job in jobs:
        leaf = job.get("name", "").split("/")[-1].strip()
        idx = leaf.find(marker)
        if idx != -1:
            tail = leaf[idx + len(marker):].strip()
            if tail:
                return tail.split()[0]  # device token has no spaces
    return None


def token_in_names(names: list[str], device: str) -> bool:
    """True if `device` appears as a standalone token in any path (so that
    ``p150`` does NOT match inside the runner label ``p150b``)."""
    pat = re.compile(r"(?<![0-9A-Za-z])" + re.escape(device) + r"(?![0-9A-Za-z])")
    return any(pat.search(n) for n in names)


# ---------------------------------------------------------------------------
# download + verification
# ---------------------------------------------------------------------------
def download_artifact(repo: str, artifact: dict, dest_dir: Path, cache: dict[int, Path]) -> Path:
    """Download an artifact as its raw .zip (not extracted). Verifies the
    on-disk size against the API-reported size and the zip's integrity."""
    aid = artifact["id"]
    if aid in cache:
        return cache[aid]
    if artifact.get("expired"):
        sys.exit(f"ERROR: artifact '{artifact['name']}' (id {aid}) has expired and cannot be downloaded.")
    data = run_gh(["api", f"repos/{repo}/actions/artifacts/{aid}/zip"], binary=True)
    path = dest_dir / f"{aid}.zip"
    path.write_bytes(data)

    expected = artifact.get("size_in_bytes")
    actual = path.stat().st_size
    if expected and actual != expected:
        sys.exit(f"ERROR: size mismatch for '{artifact['name']}': downloaded {actual} bytes, expected {expected}.")
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        if bad is not None:
            sys.exit(f"ERROR: corrupt entry '{bad}' in '{artifact['name']}'.")
    except zipfile.BadZipFile:
        sys.exit(f"ERROR: '{artifact['name']}' is not a valid zip file.")

    cache[aid] = path
    return path


def resolve_model(
    model: str,
    devices: list[str],
    artifacts: list[dict],
    jobs: list[dict],
    repo: str,
    tmp: Path,
    cache: dict[int, Path],
) -> dict[str, dict]:
    """Return {device: artifact} for the requested devices of one model."""
    candidates = [a for a in artifacts if a["name"].startswith(f"{ARTIFACT_PREFIX}{model}_")]
    if not candidates:
        sys.exit(
            f"ERROR: no '{ARTIFACT_PREFIX}{model}_*' artifacts found for model '{model}'.\n"
            f"       Check the model name and that the run produced its bundle."
        )

    # Primary: map each candidate's runner -> device via the run's job names.
    by_device: dict[str, dict] = {}
    for a in candidates:
        runner = runner_of(a["name"], model)
        dev = device_from_jobs(jobs, model, runner)
        if dev and dev not in by_device:
            by_device[dev] = a

    # Fallback: for any still-missing requested device, classify candidates by
    # the device token embedded in their own file names (ground truth).
    if any(d not in by_device for d in devices):
        for a in candidates:
            path = download_artifact(repo, a, tmp, cache)
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            for d in devices:
                if d not in by_device and token_in_names(names, d):
                    by_device[d] = a

    chosen: dict[str, dict] = {}
    for d in devices:
        if d not in by_device:
            found = ", ".join(sorted(by_device)) or "none"
            runners = ", ".join(runner_of(a["name"], model) for a in candidates)
            sys.exit(
                f"ERROR: could not find an artifact for model '{model}' device '{d}'.\n"
                f"       Devices resolved for this model: {found}.\n"
                f"       Candidate runner labels seen: {runners}."
            )
        chosen[d] = by_device[d]
    return chosen


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------
def package(version: str, staged: dict[str, Path], out_dir: Path) -> Path:
    """Write <version>-release_artifacts.zip: a single top-level folder holding
    the inner zips, all stored uncompressed (matches the previous release)."""
    root = f"{version}-release_artifacts"
    out_path = out_dir / f"{root}.zip"
    now = time.localtime()[:6]

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        # explicit directory entry, like the previous release package
        dir_info = zipfile.ZipInfo(root + "/", date_time=now)
        dir_info.external_attr = (0o40755 << 16) | 0x10  # drwxr-xr-x + MS-DOS dir bit
        zf.writestr(dir_info, b"")

        for arcname in sorted(staged):
            src = staged[arcname]
            mtime = time.localtime(src.stat().st_mtime)[:6]
            info = zipfile.ZipInfo(f"{root}/{arcname}", date_time=mtime)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, src.read_bytes())

    return out_path


def validate_structure(zip_path: Path, reference: Path | None) -> None:
    """Assert the produced package has the expected shape; optionally diff the
    inner-zip naming pattern against a previous release package."""
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    top_dirs = {i.filename.split("/")[0] for i in infos}
    if len(top_dirs) != 1:
        sys.exit(f"ERROR: expected exactly one top-level folder, found: {sorted(top_dirs)}")
    inner = [i for i in infos if not i.filename.endswith("/")]
    for i in inner:
        if not i.filename.endswith(".zip"):
            sys.exit(f"ERROR: unexpected non-zip entry in package: {i.filename}")
        if i.compress_type != zipfile.ZIP_STORED:
            sys.exit(f"ERROR: entry not stored uncompressed: {i.filename}")

    if reference and reference.exists():
        pat = re.compile(r"^[^/]+/workflow_logs_release_.+\.zip$")
        with zipfile.ZipFile(reference) as zf:
            ref_inner = [n for n in zf.namelist() if not n.endswith("/")]
        ours_ok = all(pat.match(i.filename) for i in inner)
        ref_ok = all(pat.match(n) for n in ref_inner)
        status = "matches" if (ours_ok and ref_ok) else "DIFFERS FROM"
        print(f"  Layout vs reference '{reference.name}': inner-zip naming {status} "
              f"the 'workflow_logs_release_<model>_<device>.zip' pattern.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_model_specs(specs: list[str]) -> dict[str, list[str]]:
    """Parse repeated --model MODEL=dev1,dev2 into {model: [devices]}."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            sys.exit(f"ERROR: bad --model spec '{spec}'. Use MODEL=device1,device2 (e.g. speecht5_tts=p150,p300x2).")
        model, devs = spec.split("=", 1)
        model = model.strip()
        devices = [d.strip() for d in devs.split(",") if d.strip()]
        if not model or not devices:
            sys.exit(f"ERROR: bad --model spec '{spec}'. Need a model and at least one device.")
        out.setdefault(model, [])
        for d in devices:
            if d not in out[model]:
                out[model].append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build <version>-release_artifacts.zip from a tt-shield Release Actions run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})")
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Actions run ID")
    ap.add_argument("--version", default=DEFAULT_VERSION, help="Release version, e.g. v0.15.0")
    ap.add_argument(
        "--model", action="append", default=[], metavar="MODEL=dev1,dev2",
        help="Model and its devices. Repeatable. If omitted, the embedded DEFAULT_MODELS is used.",
    )
    ap.add_argument("--output-dir", "--destination", default=".", dest="output_dir",
                    help="Where to write the final zip. Accepts absolute path, relative path, or '.' for cwd (default: cwd)")
    ap.add_argument("--reference-zip", default=None, help="Optional previous release zip to sanity-check layout against")
    ap.add_argument("--keep-temp", action="store_true", help="Keep the temp download dir for inspection")
    ap.add_argument("--strict", action="store_true",
                    help="Treat a missing internal device token as a hard error (default: warn)")
    args = ap.parse_args()

    models = parse_model_specs(args.model) if args.model else dict(DEFAULT_MODELS)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = Path(args.reference_zip).expanduser().resolve() if args.reference_zip else None

    print(f"Repo:    {args.repo}")
    print(f"Run:     {args.run_id}")
    print(f"Version: {args.version}")
    print("Scope:")
    for m, devs in models.items():
        print(f"  - {m}: {', '.join(devs)}")
    print()

    print("Fetching artifact and job listings ...")
    artifacts = list_artifacts(args.repo, args.run_id)
    jobs = list_jobs(args.repo, args.run_id)
    print(f"  {len(artifacts)} artifacts, {len(jobs)} jobs.\n")

    tmp_dir = Path(tempfile.mkdtemp(prefix="release_artifacts_"))
    cache: dict[int, Path] = {}
    staged: dict[str, Path] = {}  # inner-zip filename -> staged path

    try:
        for model, devices in models.items():
            chosen = resolve_model(model, devices, artifacts, jobs, args.repo, tmp_dir, cache)
            for device in devices:
                artifact = chosen[device]
                src = download_artifact(args.repo, artifact, tmp_dir, cache)

                # ground-truth check: the device must appear as a token inside the bundle
                with zipfile.ZipFile(src) as zf:
                    names = zf.namelist()
                if not token_in_names(names, device):
                    msg = (f"device token '{device}' not found inside bundle "
                           f"'{artifact['name']}' (resolved from runner "
                           f"'{runner_of(artifact['name'], model)}').")
                    if args.strict:
                        sys.exit(f"ERROR: {msg}")
                    print(f"  WARNING: {msg}")

                inner_name = f"{ARTIFACT_PREFIX}{model}_{device}.zip"
                staged_path = tmp_dir / inner_name
                staged_path.write_bytes(src.read_bytes())
                staged[inner_name] = staged_path
                print(f"  + {inner_name:55s}  <- {artifact['name']}  "
                      f"({artifact.get('size_in_bytes', '?')} bytes)")

        print(f"\nPackaging {len(staged)} bundles ...")
        out_path = package(args.version, staged, out_dir)
        validate_structure(out_path, reference)
        print(f"\nDone: {out_path}  ({out_path.stat().st_size} bytes)")
    finally:
        if args.keep_temp:
            print(f"\n(temp dir kept: {tmp_dir})")
        else:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
