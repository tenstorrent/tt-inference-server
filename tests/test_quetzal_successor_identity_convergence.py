# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import json
from pathlib import Path

from workflows.model_spec import load_templates_from_yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "productization/quetzal_successor_identity_matrix.json"
CATALOGUE = ROOT / "workflows/model_specs/dev/llm.yaml"
MODELS_CI = ROOT / ".github/workflows/models-ci-config.json"


def _matrix() -> dict:
    return json.loads(MATRIX.read_text())


def _quetzal_templates() -> dict[str, object]:
    templates = load_templates_from_yaml(CATALOGUE)
    return {
        template.weights[0]: template
        for template in templates
        if template.impl.impl_id == "quetzal"
        and template.weights[0] in _matrix()["models"]
    }


def test_successor_dev_rows_bind_the_banked_executable_identities():
    matrix = _matrix()["models"]
    templates = _quetzal_templates()
    assert set(templates) == set(matrix)

    for model_id, identity in matrix.items():
        spec = templates[model_id].expand_to_specs()[0]
        env = spec.env_vars
        package_id = identity["package_id"]
        package_root = f"/home/container_app_user/quetzal/packages/{package_id}"

        assert spec.device_model_spec.max_context == identity["max_context"]
        assert spec.device_model_spec.max_concurrency == identity["max_concurrency"]
        assert env["QUETZAL_MODEL"] == model_id
        assert env["QUETZAL_HF_REVISION"] == identity["checkpoint_revision"]
        assert (
            env["QUETZAL_REQUIRED_SOURCE_REVISION"]
            == identity["runtime_source_revision"]
        )
        assert (
            env["QUETZAL_GENERATOR_SOURCE_REVISION"]
            == identity["generator_source_revision"]
        )
        assert env["QUETZAL_REQUIRED_TT_METAL_COMMIT"] == identity["tt_metal_revision"]
        assert (
            env["QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256"]
            == identity["tt_metal_patchset_sha256"]
        )
        assert env["QUETZAL_PACKAGE_ID"] == package_id
        assert (
            env["QUETZAL_BUNDLE_MANIFEST_SHA256"] == identity["bundle_manifest_sha256"]
        )
        assert env["QUETZAL_PACKAGE_ROOT"] == package_root
        assert env["QZ_MODELS_ROOT"] == package_root
        assert env["QUETZAL_REQUIRED_PREFILL_BUCKETS"] == identity["prefill_buckets"]
        for role in (
            "QZ_QUALIFICATION_MANIFEST",
            "QUETZAL_PREFILL_GENERATED_PY",
            "QUETZAL_PREFILL_METADATA_JSON",
            "QUETZAL_DECODE_GENERATED_PY",
            "QUETZAL_DECODE_METADATA_JSON",
            "QUETZAL_WEIGHTS",
        ):
            assert env[role].startswith(package_root + "/")

        slug = identity["artifact_slug"]
        assert (
            f"/compiled/{slug}/full/prefill/generated.py"
            in env["QUETZAL_PREFILL_GENERATED_PY"]
        )
        assert (
            f"/compiled/{slug}/full/decode/generated.py"
            in env["QUETZAL_DECODE_GENERATED_PY"]
        )
        assert f"/compiled_weights/{slug}/full/weights.pt" in env["QUETZAL_WEIGHTS"]


def test_staged_and_active_catalogues_cannot_drift():
    matrix = _matrix()["models"]
    config = json.loads(MODELS_CI.read_text())["models"]

    for identity in matrix.values():
        rows = config[identity["models_ci_key"]].get("implementations", [])
        active = [row for row in rows if row.get("impl") == "quetzal"]
        enrollment = identity["models_ci_enrollment"]
        assert bool(active) is enrollment["active"]

        if enrollment["active"]:
            assert identity["publication"]["administrator_owned"] is True
            assert len(active) == 1
            host_root = identity["publication"]["runner_visible_host_root"]
            assert host_root
            for schedule in ("nightly", "release"):
                args = active[0]["ci"][schedule]["device-args"]["P300X2"][
                    "additional-args"
                ]
                assert f"--quetzal-models-root {host_root}" in args
        else:
            assert enrollment["blocker"]


def test_attestation_is_informational_not_an_executable_identity_gate():
    matrix = _matrix()["models"]
    gpt = _quetzal_templates()["openai/gpt-oss-120b"].expand_to_specs()[0]

    assert matrix["openai/gpt-oss-120b"]["optional_runtime_attestation_sha256"]
    assert "QUETZAL_RUNTIME_ATTESTATION_SHA256" not in gpt.env_vars
