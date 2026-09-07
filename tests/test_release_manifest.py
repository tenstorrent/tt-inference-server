# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from scripts.release import release_manifest as rm
from scripts.release.release_scope import ProdLeaf, ProdPin

# Identity + version are always present; every other field (commit pins
# included) is emitted only when it has a value (no null fields in the output).
REQUIRED_KEYS = {
    "impl_id",
    "hf_model_repo",
    "model_name",
    "device",
    "engine",
    "version",
}
OPTIONAL_KEYS = {
    "tt_metal_commit",
    "vllm_commit",
    "status",
    "previous_status",
    "previous_tt_metal_commit",
    "ci_job_url",
    "checks",
}


def _assert_lean(entry):
    """Every entry carries the required keys, only known optional keys, and no
    null/empty values."""
    keys = set(entry)
    assert REQUIRED_KEYS <= keys
    assert keys <= REQUIRED_KEYS | OPTIONAL_KEYS
    assert all(value not in (None, {}) for value in entry.values())


# ---------------------------------------------------------------------------
# canonicalisation / version helpers
# ---------------------------------------------------------------------------
def test_version_key_is_numeric_not_lexicographic():
    assert rm.version_key("0.9.0") < rm.version_key("0.21.0")
    assert rm.version_key("v0.16.0") == (0, 16, 0)
    assert rm.version_key("0.18.0-temp") == (0, 18, 0)


def test_strip_and_with_v_roundtrip():
    assert rm.strip_v("v0.20.0") == "0.20.0"
    assert rm.with_v("0.20.0") == "v0.20.0"
    assert rm.with_v("v0.20.0") == "v0.20.0"


def test_canonical_identity_matches_repo_enums():
    assert rm.canonical_device("p300x2") == "P300X2"
    assert rm.canonical_device("galaxy") == "GALAXY"
    assert rm.canonical_engine("vllm") == "vLLM"
    assert rm.canonical_engine("FORGE") == "forge"


# ---------------------------------------------------------------------------
# live builder
# ---------------------------------------------------------------------------
def _prod_leaf(identity, version, tt, vllm, status):
    return ProdLeaf(
        identity=identity,
        pin=ProdPin(
            version=version, tt_metal_commit=tt, vllm_commit=vllm, docker_image=None
        ),
        status=status,
    )


def test_build_live_manifest_from_rows_and_prod():
    identity = ("meta-llama/Llama-3.1-8B-Instruct", "T3K", "vLLM", "tt_transformers")
    current_prod = {identity: _prod_leaf(identity, "0.22.0", "abc1234", "def5678", "READY")}
    rows = [
        {
            "identity": identity,
            "status_after": "READY",
            "status_before": "EXPERIMENTAL",
            "tt_before": "0000000",
            "ci_url": "https://github.com/tenstorrent/tt-shield/actions/runs/1/job/2",
        }
    ]
    manifest = rm.build_live_manifest(
        version="0.22.0",
        run_id="123",
        rows=rows,
        current_prod=current_prod,
        generated="2026-09-07",
    )
    assert manifest["manifest_schema"] == rm.MANIFEST_SCHEMA
    assert manifest["release"] == "v0.22.0"
    assert "source" not in manifest
    assert "sentinel_pipeline_ids" not in manifest
    assert "notes" not in manifest
    assert manifest["tt_shield_run_id"] == "123"  # present for a live run
    (entry,) = manifest["changed"]
    _assert_lean(entry)
    assert entry["model_name"] == "Llama-3.1-8B-Instruct"
    assert entry["tt_metal_commit"] == "abc1234"
    assert entry["vllm_commit"] == "def5678"
    assert entry["previous_tt_metal_commit"] == "0000000"
    assert entry["status"] == "READY"
    assert entry["previous_status"] == "EXPERIMENTAL"
    assert entry["ci_job_url"].endswith("/job/2")
    assert entry["version"] == "0.22.0"
    # live mode has no DB-derived check history -> that field is simply absent
    assert "checks" not in entry


def test_build_live_manifest_rejects_row_missing_from_prod():
    identity = ("org/m", "T3K", "vLLM", "impl")
    with pytest.raises(ValueError, match="missing release identity"):
        rm.build_live_manifest(
            version="0.22.0",
            run_id=None,
            rows=[{"identity": identity}],
            current_prod={},
        )
