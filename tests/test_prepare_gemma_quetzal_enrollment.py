# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import copy
import json
import subprocess
from pathlib import Path

import pytest

import scripts.prepare_gemma_quetzal_enrollment as gemma_enrollment
from scripts.prepare_gemma_quetzal_enrollment import (
    ARTIFACT_SOURCE,
    DESCRIPTOR_CONTAINER_PATH,
    DESCRIPTOR_SHA256,
    EXPECTED_ARTIFACT_SHA256,
    EnrollmentError,
    HF_REVISION,
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
    workflow = root / ".github/workflows/dynamic-workflow.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        json.dumps(
            {
                "on": {
                    "workflow_call": {
                        "inputs": {"tt-quetzal-commit": {"default": QUETZAL_SOURCE}}
                    }
                }
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "add",
            "identity",
            ".github/workflows/dynamic-workflow.yml",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "identity"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return root, revision


@pytest.fixture(autouse=True)
def exact_shield_image_ancestor(shield_checkout, monkeypatch):
    _, revision = shield_checkout
    monkeypatch.setattr(gemma_enrollment, "SHIELD_IMAGE_ANCESTOR", revision)


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
            "artifact_source_revision": ARTIFACT_SOURCE,
            "quetzal_runtime_revision": QUETZAL_SOURCE,
            "ttis_revision": ttis_revision or current_ttis_revision(),
            "shield_revision": shield_revision,
            "tt_metal_revision": TT_METAL,
            "tt_metal_patchset_sha256": PATCHSET,
            "patchset_applied_manifest_matches": True,
            "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
        },
        "runtime": {
            "image": "ghcr.io/tenstorrent/ttis-quetzal-gemma@sha256:" + "a" * 64,
            "quetzal_source_revision": QUETZAL_SOURCE,
            "tt_metal_revision": TT_METAL,
            "tt_metal_patchset_sha256": PATCHSET,
            "server_boundary": "official_ttis",
            "platform_provider": "vllm-tt-plugin",
            "plugin_entrypoint": "quetzal_model_registry",
            "vllm_plugins": "quetzal_model_registry,tt",
            "tt_vllm_builtin_models": "0",
            "native_fallback_allowed": False,
            "descriptor_container_path": DESCRIPTOR_CONTAINER_PATH,
            "descriptor_sha256": DESCRIPTOR_SHA256,
        },
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": "3" * 64,
        "host_package_root": f"/mnt/models/quetzal/immutable/v1/{PACKAGE_ID}",
        "container_package_root": f"/home/container_app_user/quetzal/packages/{PACKAGE_ID}",
        "profile": {
            "batch_size": 1,
            "concurrency": 1,
            "prefill_capacity": 4096,
            "decode_capacity": 4096,
            "precision": "BF16",
        },
        "topology": {
            "chip_count": 4,
            "mesh_shape": [2, 2],
            "collective": "Ring",
            "links": 2,
            "runner_label": RUNNER,
            "descriptor_sha256": DESCRIPTOR_SHA256,
        },
        "roles": {
            "compiled_weights": "compiled_weights/gemma/weights.pt",
            "generated_prefill": "compiled/gemma/prefill/generated.py",
            "prefill_metadata": "compiled/gemma/prefill/metadata.json",
            "generated_decode": "compiled/gemma/decode/generated.py",
            "decode_metadata": "compiled/gemma/decode/metadata.json",
            "qualification_manifest": "qualification_manifest.yaml",
        },
        "qualification": {
            "pcc": 0.991,
            "fresh": True,
            "exact_package_identity": PACKAGE_ID,
            "endpoint_collective": "Ring",
            "endpoint_collective_links": 2,
            "capacity_endpoint": {
                "isl": 4095,
                "osl": 1,
                "total_tokens": 4096,
                "http_200": True,
            },
            "semantic_endpoint": {
                "http_200": True,
                "visible_nonempty": True,
                "completion_tokens": 12,
            },
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
    assert device_spec["max_context"] == 4096
    assert device_spec["vllm_args"]["max_model_len"] == 4096
    env = device_spec["env_vars"]
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert env["TT_MESH_GRAPH_DESC_PATH"] == DESCRIPTOR_CONTAINER_PATH
    assert env["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert env["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert env["QUETZAL_REQUIRED_SOURCE_REVISION"] == QUETZAL_SOURCE
    assert env["QUETZAL_PREFILL_METADATA_JSON"].endswith(
        "/compiled/gemma/prefill/metadata.json"
    )
    assert env["QUETZAL_DECODE_METADATA_JSON"].endswith(
        "/compiled/gemma/decode/metadata.json"
    )
    assert rendered["handoff"]["quetzal_source_revision"] == QUETZAL_SOURCE
    assert rendered["handoff"]["ttis_revision"] == current_ttis_revision()
    assert rendered["handoff"]["shield_revision"] == shield_revision
    assert rendered["handoff"]["fallback_allowed"] is False
    assert row["image"] == evidence()["runtime"]["image"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda x: x.update(decision="pending"), "decision"),
        (
            lambda x: x["identity"].update(artifact_source_revision="0" * 40),
            "artifact_source_revision",
        ),
        (
            lambda x: x["identity"].update(quetzal_runtime_revision="0" * 40),
            "quetzal_runtime_revision",
        ),
        (lambda x: x["profile"].update(decode_capacity=2048), "profile"),
        (lambda x: x["topology"].update(runner_label="p300x2"), "runner_label"),
        (
            lambda x: x["runtime"].update(descriptor_sha256="0" * 64),
            "descriptor_sha256",
        ),
        (
            lambda x: x.update(host_package_root=f"/data/user-packages/{PACKAGE_ID}"),
            "host_package_root",
        ),
        (
            lambda x: x.update(container_package_root=f"/tmp/quetzal/{PACKAGE_ID}"),
            "container_package_root",
        ),
        (lambda x: x["qualification"].update(pcc=0.989), "pcc"),
        (
            lambda x: x["qualification"].update(endpoint_collective="Linear"),
            "endpoint_collective",
        ),
        (
            lambda x: x["qualification"]["capacity_endpoint"].update(osl=2),
            "capacity_endpoint.osl",
        ),
        (
            lambda x: x["qualification"]["semantic_endpoint"].update(
                visible_nonempty=False
            ),
            "visible_nonempty",
        ),
        (
            lambda x: x["runtime"].update(
                image="ghcr.io/tenstorrent/ttis-quetzal-gemma:latest"
            ),
            "runtime.image",
        ),
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


def test_shield_checkout_must_contain_implementation_image_support(
    shield_checkout, monkeypatch
):
    shield_root, shield_revision = shield_checkout
    monkeypatch.setattr(gemma_enrollment, "SHIELD_IMAGE_ANCESTOR", "f" * 40)
    with pytest.raises(
        EnrollmentError, match="implementation-qualified immutable image"
    ):
        render_fragments(
            evidence(shield_revision=shield_revision),
            ROOT,
            shield_repo_root=shield_root,
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
