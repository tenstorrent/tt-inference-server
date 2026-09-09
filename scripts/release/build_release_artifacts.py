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

Requirements: Python 3.10+, the GitHub CLI ``gh`` installed and authenticated
(``gh auth status``) with ``repo`` scope.

Usage
-----
By default the model+device scope is read from the ``release`` entries in
``.github/workflows/models-ci-config.json`` (the same release list
promote_dev_spec_to_prod.py consumes), so a normal release run needs only the
run id and version:

    python3 build_release_artifacts.py \
        --run-id 26592936143 \
        --version v0.15.0 \
        --output-dir .

Override the scope with one or more --model flags (e.g. to rebuild a single
model, or to package models that are not in the release list):

    python3 build_release_artifacts.py \
        --run-id 26592936143 \
        --version v0.15.0 \
        --model speecht5_tts=p150,p300x2 \
        --model whisper-large-v3=p150,p300x2 \
        --output-dir .

Other flags: --repo, --ci-config, --output-dir, --keep-temp, and --strict
(turn the internal device-token check from a warning into a hard error).
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.release.model_spec_resolver import (  # noqa: E402
    collect_release_combos,
    load_dev_model_spec_sources,
    resolve_release_combos,
)
from scripts.release.release_scope import extract_bundle_identity  # noqa: E402
from utils.model_naming import model_name_variants, slugify_model_id  # noqa: E402

ARTIFACT_PREFIX = "workflow_logs_release_"
DEFAULT_REPO = "tenstorrent/tt-shield"
DEFAULT_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"
DEFAULT_DEV_DIR = REPO_ROOT / "workflows" / "model_specs" / "dev"

# ---------------------------------------------------------------------------
# Defaults — override any of them on the command line.
# ---------------------------------------------------------------------------
DEFAULT_VERSION = "v0.15.0"
DEFAULT_RUN_ID = "26592936143"


def resolve_configured_scope(ci_config: dict, dev_dir: Path):
    # resolve_release_combos() rejects two selectors resolving to one identity.
    resolved = resolve_release_combos(
        collect_release_combos(ci_config),
        load_dev_model_spec_sources(dev_dir),
    )
    expected = {}
    models = {}
    archive_owners = {}
    for item in resolved:
        model = item.identity[0]
        device = item.combo.device.name.lower()
        key = (model, device)
        archive_key = (slugify_model_id(model), device)
        archive_owner = archive_owners.get(archive_key)
        if archive_owner is not None and archive_owner != item.identity:
            raise ValueError(
                f"Release identities {archive_owner!r} and {item.identity!r} "
                f"collide on artifact filename token {archive_key!r}"
            )
        archive_owners[archive_key] = item.identity
        if key in expected and expected[key] != item.identity:
            raise ValueError(
                f"Artifact names cannot distinguish exact identities "
                f"{expected[key]!r} and {item.identity!r}"
            )
        expected[key] = item.identity
        models.setdefault(model, [])
        if device not in models[model]:
            models[model].append(device)
    return models, expected


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
    token = artifact_model_token(artifact_name, model)
    if token is None:
        raise ValueError(f"Artifact {artifact_name!r} does not match model {model!r}")
    rest = artifact_name[len(ARTIFACT_PREFIX) + len(token) + 1 :]
    return rest.rsplit("_", 1)[0]


def artifact_model_token(artifact_name: str, model: str) -> str | None:
    matches = [
        token
        for token in model_name_variants(model)
        if artifact_name.startswith(f"{ARTIFACT_PREFIX}{token}_")
    ]
    return max(matches, key=len) if matches else None


def device_from_jobs(jobs: list[dict], model: str, runner: str) -> str | None:
    """Find the device a (model, runner) pair ran on, from the job name
    pattern ``run-release-<model>-<runner>-<device>``."""
    for job in jobs:
        name = job.get("name", "").strip()
        for token in model_name_variants(model):
            marker = f"run-release-{token}-{runner}-"
            idx = name.find(marker)
            if idx != -1:
                tail = name[idx + len(marker) :].strip()
                if tail:
                    return tail.split()[0]
    return None


def token_in_names(names: list[str], device: str) -> bool:
    """True if `device` appears as a standalone token in any path (so that
    ``p150`` does NOT match inside the runner label ``p150b``)."""
    pat = re.compile(r"(?<![0-9A-Za-z])" + re.escape(device) + r"(?![0-9A-Za-z])")
    return any(pat.search(n) for n in names)


def validate_bundle_identity(bundle: Path, expected_identity):
    actual_identity = extract_bundle_identity(bundle)
    if actual_identity != expected_identity:
        raise ValueError(
            f"Runtime identity {actual_identity!r} does not match configured "
            f"{expected_identity!r}"
        )
    return actual_identity


def validate_staged_identity_set(expected, validated) -> None:
    if validated != expected:
        raise ValueError(
            "Staged runtime identities do not match configured scope: "
            f"missing={sorted(expected - validated)!r}, "
            f"extra={sorted(validated - expected)!r}"
        )


# ---------------------------------------------------------------------------
# download + verification
# ---------------------------------------------------------------------------
def download_artifact(
    repo: str, artifact: dict, dest_dir: Path, cache: dict[int, Path]
) -> Path:
    """Download an artifact as its raw .zip (not extracted). Verifies the
    on-disk size against the API-reported size and the zip's integrity."""
    aid = artifact["id"]
    if aid in cache:
        return cache[aid]
    if artifact.get("expired"):
        sys.exit(
            f"ERROR: artifact '{artifact['name']}' (id {aid}) has expired and cannot be downloaded."
        )
    data = run_gh(["api", f"repos/{repo}/actions/artifacts/{aid}/zip"], binary=True)
    path = dest_dir / f"{aid}.zip"
    path.write_bytes(data)

    expected = artifact.get("size_in_bytes")
    actual = path.stat().st_size
    if expected and actual != expected:
        sys.exit(
            f"ERROR: size mismatch for '{artifact['name']}': downloaded {actual} bytes, expected {expected}."
        )
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
    expected_identities: dict[tuple[str, str], tuple] | None = None,
) -> dict[str, dict]:
    """Return {device: artifact} for the requested devices of one model."""
    expected_identities = expected_identities or {}
    candidates = [
        artifact
        for artifact in artifacts
        if artifact_model_token(artifact["name"], model) is not None
    ]
    if not candidates:
        sys.exit(
            f"ERROR: no '{ARTIFACT_PREFIX}{model}_*' artifacts found for model '{model}'.\n"
            f"       Check the model name and that the run produced its bundle."
        )

    # Primary: map each candidate's runner -> device via the run's job names.
    by_device: dict[str, list[dict]] = {}
    for a in candidates:
        runner = runner_of(a["name"], model)
        dev = device_from_jobs(jobs, model, runner)
        if dev:
            by_device.setdefault(dev, []).append(a)

    # Fallback: for any still-missing requested device, classify candidates by
    # the device token embedded in their own file names (ground truth).
    if any(d not in by_device for d in devices):
        for a in candidates:
            path = download_artifact(repo, a, tmp, cache)
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            for d in devices:
                if d not in by_device and token_in_names(names, d):
                    by_device.setdefault(d, []).append(a)

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
        device_candidates = by_device[d]
        expected_identity = expected_identities.get((model, d))
        if expected_identity is None:
            chosen[d] = device_candidates[0]
            continue

        matching = []
        rejected = []
        for artifact in device_candidates:
            path = download_artifact(repo, artifact, tmp, cache)
            try:
                validate_bundle_identity(path, expected_identity)
            except ValueError as exc:
                rejected.append(f"{artifact['name']!r}: {exc}")
            else:
                matching.append(artifact)
        if len(matching) != 1:
            details = "; ".join(rejected) or "none"
            sys.exit(
                f"ERROR: expected one artifact for exact identity "
                f"{expected_identity!r}, found {len(matching)}. "
                f"Rejected candidates: {details}"
            )
        chosen[d] = matching[0]
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_model_specs(specs: list[str]) -> dict[str, list[str]]:
    """Parse repeated --model MODEL=dev1,dev2 into {model: [devices]}."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            sys.exit(
                f"ERROR: bad --model spec '{spec}'. Use MODEL=device1,device2 (e.g. speecht5_tts=p150,p300x2)."
            )
        model, devs = spec.split("=", 1)
        model = model.strip()
        devices = [d.strip() for d in devs.split(",") if d.strip()]
        if not model or not devices:
            sys.exit(
                f"ERROR: bad --model spec '{spec}'. Need a model and at least one device."
            )
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
    ap.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})"
    )
    ap.add_argument(
        "--ci-config",
        default=str(DEFAULT_CI_CONFIG),
        help="CI config JSON providing the default release model list "
        f"(default: {DEFAULT_CI_CONFIG}). Ignored when --model is given.",
    )
    ap.add_argument(
        "--dev-dir",
        default=str(DEFAULT_DEV_DIR),
        help="Dev catalog used to resolve exact config identities",
    )
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Actions run ID")
    ap.add_argument(
        "--version", default=DEFAULT_VERSION, help="Release version, e.g. v0.15.0"
    )
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="MODEL=dev1,dev2",
        help="Model and its devices, e.g. Qwen3-32B=galaxy. Repeatable. If omitted, "
        "the scope is read from the 'release' entries in --ci-config.",
    )
    ap.add_argument(
        "--output-dir",
        "--destination",
        default=".",
        dest="output_dir",
        help="Where to write the final zip. Accepts absolute path, relative path, or '.' for cwd (default: cwd)",
    )
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temp download dir for inspection",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Treat a missing internal device token as a hard error (default: warn)",
    )
    args = ap.parse_args()

    if args.model:
        models = parse_model_specs(args.model)
        expected_identities = {}
        print(
            "WARNING: manual --model scope does not perform final exact-identity "
            "verification.",
            file=sys.stderr,
        )
    else:
        ci_config_path = Path(args.ci_config).expanduser()
        try:
            ci_config = json.loads(ci_config_path.read_text())
        except FileNotFoundError:
            sys.exit(f"ERROR: CI config not found: {ci_config_path}")
        try:
            models, expected_identities = resolve_configured_scope(
                ci_config,
                Path(args.dev_dir).expanduser(),
            )
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        if not models:
            sys.exit(
                f"ERROR: no models are marked 'release' in {ci_config_path}.\n"
                f"       Add a ci.release entry, or pass --model MODEL=dev1,dev2 explicitly."
            )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    validated_identities = set()

    try:
        for model, devices in models.items():
            chosen = resolve_model(
                model,
                devices,
                artifacts,
                jobs,
                args.repo,
                tmp_dir,
                cache,
                expected_identities,
            )
            for device in devices:
                artifact = chosen[device]
                src = download_artifact(args.repo, artifact, tmp_dir, cache)

                expected_identity = expected_identities.get((model, device))
                if expected_identity is not None:
                    try:
                        validated_identities.add(
                            validate_bundle_identity(src, expected_identity)
                        )
                    except ValueError as exc:
                        sys.exit(
                            f"ERROR: bundle '{artifact['name']}' has invalid "
                            f"runtime-spec evidence: {exc}"
                        )

                # ground-truth check: the device must appear as a token inside the bundle
                with zipfile.ZipFile(src) as zf:
                    names = zf.namelist()
                if not token_in_names(names, device):
                    msg = (
                        f"device token '{device}' not found inside bundle "
                        f"'{artifact['name']}' (resolved from runner "
                        f"'{runner_of(artifact['name'], model)}')."
                    )
                    if args.strict:
                        sys.exit(f"ERROR: {msg}")
                    print(f"  WARNING: {msg}")

                inner_name = f"{ARTIFACT_PREFIX}{slugify_model_id(model)}_{device}.zip"
                staged_path = tmp_dir / inner_name
                staged_path.write_bytes(src.read_bytes())
                staged[inner_name] = staged_path
                print(
                    f"  + {inner_name:55s}  <- {artifact['name']}  "
                    f"({artifact.get('size_in_bytes', '?')} bytes)"
                )

        expected_identity_set = set(expected_identities.values())
        if expected_identity_set:
            try:
                validate_staged_identity_set(
                    expected_identity_set,
                    validated_identities,
                )
            except ValueError as exc:
                sys.exit(f"ERROR: {exc}")

        print(f"\nPackaging {len(staged)} bundles ...")
        out_path = package(args.version, staged, out_dir)
        print(f"\nDone: {out_path}  ({out_path.stat().st_size} bytes)")
    finally:
        if args.keep_temp:
            print(f"\n(temp dir kept: {tmp_dir})")
        else:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
