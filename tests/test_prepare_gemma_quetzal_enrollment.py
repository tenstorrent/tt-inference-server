# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_gemma_quetzal_enrollment import (
    EnrollmentError,
    HF_REVISION,
    INIT_SHA256,
    MODEL,
    PATCHSET,
    QUETZAL_SOURCE,
    RUNNER,
    SCHEMA,
    TT_METAL,
    render_fragments,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "sha256-" + "1" * 64 + "-" + "2" * 64
SHIELD_REVISION = "da43fee60603da9a3b7a6c1bf5643fe0928eab0f"


def current_ttis_revision():
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def shield_checkout(tmp_path):
    root = tmp_path / "tt-shield"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True
    )
    (root / "identity").write_text("exact shield checkout\n")
    subprocess.run(["git", "-C", str(root), "add", "identity"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "identity"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return root, revision


def evidence(*, ttis_revision=None, shield_revision=SHIELD_REVISION):
    return {
        "schema_version": SCHEMA,
        "decision": "approved",
        "administrator_owned": True,
        "read_only": True,
        "no_writable_aliases": True,
        "revocation_status": "active",
        "identity": {
            "model_id": MODEL,
            "hf_revision": HF_REVISION,
            "quetzal_source_revision": QUETZAL_SOURCE,
            "ttis_revision": ttis_revision or current_ttis_revision(),
            "shield_revision": shield_revision,
            "tt_metal_revision": TT_METAL,
            "tt_metal_patchset_sha256": PATCHSET,
            "patchset_applied_manifest_matches": True,
            "initialization_milestones_sha256": INIT_SHA256,
        },
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": "3" * 64,
        "host_package_root": f"/mnt/models/quetzal/immutable/v1/{PACKAGE_ID}",
        "container_package_root": f"/home/container_app_user/quetzal/packages/{PACKAGE_ID}",
        "profile": {
            "batch_size": 1,
            "concurrency": 1,
            "prefill_capacity": 1024,
            "decode_capacity": 2048,
            "precision": "BFP8",
        },
        "topology": {
            "chip_count": 4,
            "mesh_shape": [2, 2],
            "collective": "Ring",
            "links": 2,
            "runner_label": RUNNER,
        },
        "roles": {
            "compiled_weights": "compiled_weights/gemma/weights.pt",
            "generated_prefill": "compiled/gemma/prefill/generated.py",
            "generated_decode": "compiled/gemma/decode/generated.py",
            "qualification_manifest": "qualification_manifest.yaml",
        },
        "qualification": {
            "pcc": 0.991,
            "fresh": True,
            "exact_package_identity": PACKAGE_ID,
            "endpoint_isl": 1024,
            "endpoint_osl": 512,
            "http_200": True,
            "clean_unload": True,
            "zero_device_holders_after": True,
            "initialization_terminal": {"event": "engine_ready", "state": "complete"},
        },
    }


def test_exact_evidence_renders_schema_valid_non_dispatching_fragments(shield_checkout):
    shield_root, shield_revision = shield_checkout
    rendered = render_fragments(
        evidence(shield_revision=shield_revision), ROOT, shield_repo_root=shield_root
    )
    row = rendered["implementation"]
    assert set(row["ci"]) == {"nightly", "release"}
    device_spec = rendered["catalogue"]["templates"][0]["device_model_specs"][0]
    assert device_spec["max_context"] == 1024
    assert device_spec["vllm_args"]["max_model_len"] == 2048
    env = device_spec["env_vars"]
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert env["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert env["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert env["QUETZAL_REQUIRED_SOURCE_REVISION"] == QUETZAL_SOURCE
    assert rendered["handoff"]["quetzal_source_revision"] == QUETZAL_SOURCE
    assert rendered["handoff"]["ttis_revision"] == current_ttis_revision()
    assert rendered["handoff"]["shield_revision"] == shield_revision
    assert rendered["handoff"]["fallback_allowed"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda x: x.update(decision="pending"), "decision"),
        (
            lambda x: x["identity"].update(quetzal_source_revision="0" * 40),
            "quetzal_source_revision",
        ),
        (lambda x: x["profile"].update(decode_capacity=1024), "profile"),
        (lambda x: x["topology"].update(runner_label="p300x2"), "runner_label"),
        (lambda x: x["qualification"].update(pcc=0.989), "pcc"),
        (lambda x: x["qualification"].update(endpoint_osl=511), "endpoint_osl"),
        (
            lambda x: x["qualification"].update(initialization_terminal=None),
            "initialization_terminal",
        ),
        (
            lambda x: x["roles"].update(generated_decode="../generated.py"),
            "contained relative",
        ),
    ],
)
def test_dispatch_critical_mismatch_fails_closed(mutation, match, shield_checkout):
    shield_root, shield_revision = shield_checkout
    bad = copy.deepcopy(evidence(shield_revision=shield_revision))
    mutation(bad)
    with pytest.raises(EnrollmentError, match=match):
        render_fragments(bad, ROOT, shield_repo_root=shield_root)


def test_preconvergence_ttis_and_shield_evidence_is_rejected(shield_checkout):
    shield_root, shield_revision = shield_checkout
    with pytest.raises(EnrollmentError, match="identity.ttis_revision"):
        render_fragments(
            evidence(
                ttis_revision="fa81a5ea8d5a33a527192f0de1452b51366f0eee",
                shield_revision=shield_revision,
            ),
            ROOT,
            shield_repo_root=shield_root,
        )
    with pytest.raises(EnrollmentError, match="identity.shield_revision"):
        render_fragments(
            evidence(shield_revision="e52823404f76495769b03b02697b3328587a135f"),
            ROOT,
            shield_repo_root=shield_root,
        )


def test_shield_checkout_head_change_invalidates_prior_evidence(shield_checkout):
    shield_root, old_revision = shield_checkout
    (shield_root / "identity").write_text("new shield head\n")
    subprocess.run(
        ["git", "-C", str(shield_root), "commit", "-qam", "new head"], check=True
    )
    with pytest.raises(EnrollmentError, match="identity.shield_revision"):
        render_fragments(
            evidence(shield_revision=old_revision), ROOT, shield_repo_root=shield_root
        )


def test_active_config_intentionally_has_no_gemma_quetzal_lane():
    config = json.loads((ROOT / ".github/workflows/models-ci-config.json").read_text())
    rows = config["models"]["gemma-4-31B-it"]["implementations"]
    assert all(row.get("impl") != "quetzal" for row in rows)
    blocker = json.loads(
        (
            ROOT / "productization/gemma4_31b_models_ci_enrollment.blocked.json"
        ).read_text()
    )
    assert blocker["status"] == "blocked_not_dispatchable"
