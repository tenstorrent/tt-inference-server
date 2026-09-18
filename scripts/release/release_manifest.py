# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Release manifest model and builder.

A *release manifest* is the machine-readable, per-version record of what a
release actually published. It is the structured twin of the "Model Spec
Release Updates" table in the post-release PR body: one entry per released
runtime leaf -- the ``(hf_model_repo, device, engine, impl_id)`` tuple -- that
the version added or changed. It is a *delta*: only the leaves a release
publishes appear, not the whole standing catalogue.

``build_live_manifest`` builds it at release time from the exact evidence the
post-release PR already computes (release scope x prod-now x prod-base x the
tt-shield Release run), so the manifest and the note table can never drift.
Identity is canonicalised through the repo enums.

(The one-off reconstruction of manifests for *historical* releases -- from a
Snowflake-derived evidence snapshot plus git-tag commit pins -- is not part of
the shipped tooling; it lives in the catalog-repair analysis workspace.)
"""

from __future__ import annotations

import json
from pathlib import Path

from workflows.workflow_types import DeviceTypes, InferenceEngine

MANIFEST_SCHEMA = "tt-release-manifest/1"


# ---------------------------------------------------------------------------
# identity canonicalisation
# ---------------------------------------------------------------------------
def canonical_device(device: str) -> str:
    return DeviceTypes.from_string(str(device).strip()).to_string()


def canonical_engine(engine: str) -> str:
    return InferenceEngine.from_string(str(engine).strip()).value


def version_key(version: str) -> tuple:
    """Numeric version key. ``0.9.0`` sorts before ``0.21.0`` (string sort does
    not); every ordering/threshold in this module goes through it."""
    core = str(version).lstrip("v").split("-", 1)[0]
    return tuple(int(part) for part in core.split("."))


def strip_v(version: str) -> str:
    return str(version).lstrip("v")


def with_v(version: str) -> str:
    text = str(version)
    return text if text.startswith("v") else f"v{text}"


# ---------------------------------------------------------------------------
# manifest entry / envelope
# ---------------------------------------------------------------------------
def _entry(
    *,
    impl_id,
    hf_model_repo,
    device,
    engine,
    version,
    tt_metal_commit,
    vllm_commit=None,
    status=None,
    previous_status=None,
    previous_tt_metal_commit=None,
    ci_job_url=None,
    checks=None,
) -> dict:
    """A single released-leaf entry. Optional fields a given mode cannot supply
    (e.g. ci_job_url when reconstructed, checks when live) are omitted rather
    than emitted as null -- an entry carries only fields that have a value."""
    entry = {
        "impl_id": impl_id,
        "hf_model_repo": hf_model_repo,
        # Weight basename without the HF org prefix (e.g. "Qwen3.6-27B"), case
        # preserved. hf_model_repo carries at most one "/", and this basename was
        # verified to equal the pipeline/DB-validated model name for every leaf.
        "model_name": str(hf_model_repo or "").rsplit("/", 1)[-1].strip() or None,
        "device": canonical_device(device),
        "engine": canonical_engine(engine),
        "version": strip_v(version),
        "tt_metal_commit": tt_metal_commit or None,
        "vllm_commit": vllm_commit or None,
        "status": status or None,
        "previous_status": previous_status or None,
        "previous_tt_metal_commit": previous_tt_metal_commit or None,
        "ci_job_url": ci_job_url or None,
        "checks": _clean_checks(checks),
    }
    return {key: value for key, value in entry.items() if value not in (None, {})}


def _clean_checks(checks) -> dict | None:
    """Informational check health (accuracy/perf verdicts) with the empty
    ('none'/absent) values dropped; None if nothing is left."""
    if not checks:
        return None
    cleaned = {}
    for family in ("accuracy", "perf"):
        value = checks.get(family)
        if value and value != "none":
            cleaned[family] = value
    return cleaned or None


def _entry_sort_key(entry: dict) -> tuple:
    return (entry["hf_model_repo"], entry["device"], entry["engine"], entry["impl_id"])


def _envelope(*, release, generated, run_id, entries):
    envelope = {
        "manifest_schema": MANIFEST_SCHEMA,
        "release": with_v(release),
        "generated": generated or None,
        # The tt-shield Release run this version was validated in (the
        # github_pipeline_id pinned in the release note).
        "tt_shield_run_id": run_id or None,
        "changed": sorted(entries, key=_entry_sort_key),
    }
    return {key: value for key, value in envelope.items() if value is not None}


def write_manifest(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# live builder (release time) -- from create_post_release_pr rows + prod pins
# ---------------------------------------------------------------------------
def build_live_manifest(
    *,
    version,
    run_id,
    rows,
    current_prod,
    generated=None,
):
    """Build a manifest from the post-release rows and the promoted prod pins.

    ``rows`` are the dicts ``create_post_release_pr.build_rows`` returns (they
    carry identity, the tt-metal before/after commits, status before/after and
    the CI job url). ``current_prod`` is the ``load_prod_leaves`` index of the
    promoted catalogue -- it supplies the full pin (version + vllm_commit) that a
    row does not carry.
    """
    entries = []
    for row in rows:
        identity = row["identity"]
        leaf = current_prod.get(identity)
        if leaf is None:
            raise ValueError(f"Promoted prod is missing release identity {identity!r}")
        pin = leaf.pin
        entries.append(
            _entry(
                impl_id=identity[3],
                hf_model_repo=identity[0],
                device=identity[1],
                engine=identity[2],
                version=pin.version,
                tt_metal_commit=pin.tt_metal_commit,
                vllm_commit=pin.vllm_commit,
                status=row.get("status_after"),
                previous_status=row.get("status_before"),
                previous_tt_metal_commit=row.get("tt_before"),
                ci_job_url=row.get("ci_url"),
            )
        )
    return _envelope(
        release=version,
        generated=generated,
        run_id=run_id,
        entries=entries,
    )
