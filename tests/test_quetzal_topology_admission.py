# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.quetzal_topology_admission import (
    QuetzalTopologyAdmissionError,
    validate_gpt120_quetzal_preweight_admission,
)

NOW = datetime(2026, 8, 29, 12, 20, tzinfo=timezone.utc)


def _spec(impl="quetzal", model="openai/gpt-oss-120b"):
    return SimpleNamespace(
        impl=SimpleNamespace(impl_id=impl),
        model_name=model.rsplit("/", 1)[-1],
        hf_model_repo=model,
        hf_weights_repo=model,
    )


def _files(tmp_path: Path, *, physical=None, captured="2026-08-29T12:19:02Z"):
    physical = physical or {"2": 4}
    evidence_path = tmp_path / "topology-evidence.json"
    admission_path = tmp_path / "topology-admission.json"
    evidence = {
        "producer": {
            "schema": "quetzal.topology-evidence-producer.v1",
            "smoke_script_sha256": "bf3311c685554105cb420239467f4e5c32e294be57b2a34fc6cbf7b0b84573fa",
            "qualified_selection_sha256": "5ec9757ae74034c0cbc12569718c059b2b049416c736ad45a2048c5dda05b562",
            "descriptor_sha256": "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792",
            "selected_model_id": "openai/gpt-oss-120b",
            "selected_emit_sha256": "5cab85f26fe64fdea2a89c302f848a43152dcbd673133a1bfdfbf7054ba5862f",
        },
        "provenance": {
            "physical_degree_histogram": "tt_metal_topology_output",
            "collective_topology": "selected_qualified_artifact_configuration",
        },
    }
    evidence_path.write_text(json.dumps(evidence))
    admission = {
        "schema": "quetzal.topology-admission-result.v1",
        "status": "pass",
        "node": "ring2-runner",
        "slurm_job_id": 70001,
        "captured_at_utc": captured,
        "verified_at_utc": "2026-08-29T12:19:08Z",
        "chip_count": 4,
        "mesh_shape": [2, 2],
        "logical_degree_histogram": {"2": 4},
        "physical_degree_histogram": physical,
        "descriptor_sha256": "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792",
        "collective_topology": "Ring",
        "collective_num_links": 2,
        "device_holders_after": 0,
        "weights_loaded_at_capture": False,
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256(evidence_path.read_bytes()).hexdigest(),
    }
    admission_path.write_text(json.dumps(admission))
    return admission_path, evidence_path


def _environment(tmp_path, admission_path):
    return {
        "RUNNER_TEMP": str(tmp_path),
        "QUETZAL_TOPOLOGY_ADMISSION_JSON": str(admission_path),
        "SLURM_JOB_ID": "70001",
        "SLURMD_NODENAME": "ring2-runner",
    }


@pytest.fixture(autouse=True)
def _host_checks(monkeypatch):
    monkeypatch.setattr(
        "workflows.quetzal_topology_admission.socket.gethostname",
        lambda: "ring2-runner",
    )
    monkeypatch.setattr(
        "workflows.quetzal_topology_admission._require_current_slurm",
        lambda node, job_id: None,
    )
    monkeypatch.setattr(
        "workflows.quetzal_topology_admission._require_zero_holders", lambda: None
    )


def test_fresh_same_allocation_ring2_receipt_passes_before_weights(tmp_path):
    admission_path, evidence_path = _files(tmp_path)
    result = validate_gpt120_quetzal_preweight_admission(
        _spec(), environment=_environment(tmp_path, admission_path), now=NOW
    )
    assert result["status"] == "pass"
    assert result["weights_loaded"] is False
    assert result["evidence_sha256"] == sha256(evidence_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda admission, evidence: admission.update(status="fail"), "status"),
        (
            lambda admission, evidence: admission.update(
                physical_degree_histogram={"1": 2, "2": 2}
            ),
            "physical_degree_histogram",
        ),
        (
            lambda admission, evidence: admission.update(collective_topology="Linear"),
            "collective_topology",
        ),
        (
            lambda admission, evidence: evidence["producer"].update(
                selected_emit_sha256="0" * 64
            ),
            "selected_emit_sha256",
        ),
    ],
)
def test_wrong_admission_or_producer_identity_fails_closed(tmp_path, mutation, match):
    admission_path, evidence_path = _files(tmp_path)
    admission = json.loads(admission_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    mutation(admission, evidence)
    evidence_path.write_text(json.dumps(evidence))
    admission["evidence_sha256"] = sha256(evidence_path.read_bytes()).hexdigest()
    admission_path.write_text(json.dumps(admission))
    with pytest.raises(QuetzalTopologyAdmissionError, match=match):
        validate_gpt120_quetzal_preweight_admission(
            _spec(), environment=_environment(tmp_path, admission_path), now=NOW
        )


def test_stale_or_byte_changed_evidence_fails_closed(tmp_path):
    stale, _ = _files(tmp_path, captured="2026-08-29T11:00:00Z")
    with pytest.raises(QuetzalTopologyAdmissionError, match="not fresh"):
        validate_gpt120_quetzal_preweight_admission(
            _spec(), environment=_environment(tmp_path, stale), now=NOW
        )

    stale.unlink()
    evidence = tmp_path / "topology-evidence.json"
    evidence.unlink()
    admission_path, evidence_path = _files(tmp_path)
    evidence_path.write_text(evidence_path.read_text() + " ")
    with pytest.raises(QuetzalTopologyAdmissionError, match="evidence_sha256"):
        validate_gpt120_quetzal_preweight_admission(
            _spec(),
            environment=_environment(tmp_path, admission_path),
            now=NOW,
        )


def test_native_gpt_and_other_quetzal_models_are_unchanged():
    assert validate_gpt120_quetzal_preweight_admission(_spec(impl="gpt_oss")) is None
    assert (
        validate_gpt120_quetzal_preweight_admission(_spec(model="Qwen/Qwen3.6-27B"))
        is None
    )


def test_run_hook_precedes_host_weight_setup_and_docker_generation():
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text()
    hook = source.index("validate_gpt120_quetzal_preweight_admission(model_spec)")
    assert hook < source.index("setup_host(", hook)
    assert hook < source.index("generate_docker_run_command(", hook)


def test_gpt_quetzal_catalog_remains_disabled():
    from workflows.model_spec import load_templates_from_yaml
    from workflows.utils import get_repo_root_path

    for environment in ("dev", "prod"):
        templates = load_templates_from_yaml(
            get_repo_root_path() / f"workflows/model_specs/{environment}/llm.yaml"
        )
        assert not any(
            template.impl.impl_id == "quetzal"
            and "openai/gpt-oss-120b" in template.weights
            for template in templates
        )
