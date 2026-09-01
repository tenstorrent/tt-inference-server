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
        "schema": "quetzal.topology-evidence.v1",
        "captured_at_utc": captured,
        "node": "ring2-runner",
        "slurm_job_id": 70001,
        "slurm_state": "RUNNING",
        "chip_count": 4,
        "weights_loaded": False,
        "provenance": {
            "allocation_binding": "live_hostname_slurm_env_and_scontrol",
            "captured_at_utc": "producer_clock_after_close_and_holder_scan",
            "mesh_lifecycle": "bounded_mesh_smoke_log",
            "chip_count": "bounded_mesh_smoke_log",
            "weights_loaded": "exact_preweight_smoke_source",
            "device_holders_after": "post_close_fuser_device_scan",
            "mesh_shape": "bounded_mesh_smoke_log",
            "logical_degree_histogram": "tt_metal_topology_output",
            "physical_degree_histogram": "tt_metal_topology_output",
            "descriptor_sha256": "sha256_of_selected_descriptor_bytes",
            "collective_topology": "selected_qualified_artifact_configuration",
            "collective_num_links": "selected_qualified_artifact_configuration",
        },
        "mesh_lifecycle": {
            "opened": True,
            "synchronized": True,
            "closed": True,
            "exit_code": 0,
            "device_holders_after": 0,
        },
        "topology": {
            "mesh_shape": [2, 2],
            "logical_degree_histogram": {"2": 4},
            "physical_degree_histogram": physical,
            "descriptor_sha256": "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792",
            "collective_topology": "Ring",
            "collective_num_links": 2,
        },
        "producer": {
            "schema": "quetzal.topology-evidence-producer.v1",
            "smoke_script_path": "/runner/serving/mesh_open_smoke.py",
            "smoke_script_sha256": "bf3311c685554105cb420239467f4e5c32e294be57b2a34fc6cbf7b0b84573fa",
            "smoke_log_path": "/runner/output/mesh-open.log",
            "smoke_log_sha256": "1" * 64,
            "descriptor_path": "/runner/serving/mesh.textproto",
            "qualified_selection_sha256": "1852bfcc4a4acd234b83de0ce1b174b3334daa5f6f0361f835564a26f26291a7",
            "qualified_selection_path": "/runner/serving/gpt120_ring2_topology_selection.json",
            "descriptor_sha256": "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792",
            "selected_model_id": "openai/gpt-oss-120b",
            "selected_emit_sha256": "36ee31e273a66c478422fd2ff91bc4956f78d5ba6be1dd7b68f285b16c820489",
            "claim_boundary": "mesh lifecycle, count, shape, and degree histograms are observed; Ring and links=2 are selected qualified-artifact configuration",
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


def _rewrite_evidence(admission_path, evidence_path, mutation):
    evidence = json.loads(evidence_path.read_text())
    mutation(evidence)
    evidence_path.write_text(json.dumps(evidence))
    admission = json.loads(admission_path.read_text())
    admission["evidence_sha256"] = sha256(evidence_path.read_bytes()).hexdigest()
    admission_path.write_text(json.dumps(admission))


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


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e.update(schema="fabricated.v1"), "evidence.schema"),
        (
            lambda e: e.update(captured_at_utc="2026-08-29T12:19:03Z"),
            "evidence.captured_at_utc",
        ),
        (lambda e: e.update(node="other-runner"), "evidence.node"),
        (lambda e: e.update(slurm_job_id=70002), "evidence.slurm_job_id"),
        (lambda e: e.update(slurm_state="COMPLETED"), "evidence.slurm_state"),
        (lambda e: e.update(chip_count=8), "evidence.chip_count"),
        (lambda e: e.update(weights_loaded=True), "evidence.weights_loaded"),
        (
            lambda e: e["mesh_lifecycle"].update(closed=False),
            "evidence.mesh_lifecycle",
        ),
        (
            lambda e: e["mesh_lifecycle"].update(exit_code=1),
            "evidence.mesh_lifecycle",
        ),
        (
            lambda e: e["mesh_lifecycle"].update(device_holders_after=1),
            "evidence.mesh_lifecycle",
        ),
        (
            lambda e: e["topology"].update(mesh_shape=[1, 4]),
            "evidence.topology",
        ),
        (
            lambda e: e["topology"].update(logical_degree_histogram={"1": 4}),
            "evidence.topology",
        ),
        (
            lambda e: e["topology"].update(physical_degree_histogram={"1": 4}),
            "evidence.topology",
        ),
        (
            lambda e: e["topology"].update(descriptor_sha256="0" * 64),
            "evidence.topology",
        ),
        (
            lambda e: e["topology"].update(collective_topology="Linear"),
            "evidence.topology",
        ),
        (
            lambda e: e["topology"].update(collective_num_links=1),
            "evidence.topology",
        ),
    ],
)
def test_rehashed_evidence_cannot_disagree_with_pass_admission(
    tmp_path, mutation, match
):
    admission_path, evidence_path = _files(tmp_path)
    _rewrite_evidence(admission_path, evidence_path, mutation)
    with pytest.raises(QuetzalTopologyAdmissionError, match=match):
        validate_gpt120_quetzal_preweight_admission(
            _spec(), environment=_environment(tmp_path, admission_path), now=NOW
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e.pop("node"), "canonical fields"),
        (lambda e: e.pop("captured_at_utc"), "canonical fields"),
        (lambda e: e.pop("weights_loaded"), "canonical fields"),
        (
            lambda e: e["provenance"].pop("allocation_binding"),
            "evidence.provenance",
        ),
        (
            lambda e: e["provenance"].pop("collective_num_links"),
            "evidence.provenance",
        ),
        (
            lambda e: e["mesh_lifecycle"].pop("synchronized"),
            "evidence.mesh_lifecycle",
        ),
        (
            lambda e: e["topology"].pop("physical_degree_histogram"),
            "evidence.topology",
        ),
        (
            lambda e: e["producer"].pop("smoke_log_sha256"),
            "producer canonical fields",
        ),
    ],
)
def test_rehashed_evidence_omissions_fail_closed(tmp_path, mutation, match):
    admission_path, evidence_path = _files(tmp_path)
    _rewrite_evidence(admission_path, evidence_path, mutation)
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


def test_gpt_quetzal_dev_catalog_is_non_default_and_prod_remains_disabled():
    from workflows.model_spec import load_templates_from_yaml
    from workflows.utils import get_repo_root_path

    dev_templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    generated = [
        template
        for template in dev_templates
        if template.impl.impl_id == "quetzal"
        and "openai/gpt-oss-120b" in template.weights
    ]
    assert len(generated) == 1
    assert generated[0].status.name == "EXPERIMENTAL"
    assert generated[0].min_ram_gb == 32
    device = generated[0].device_model_specs[0]
    assert device.device.name == "P300X2"
    assert device.default_impl is False
    assert device.max_concurrency == 1
    assert device.max_context == 8192
    assert device.env_vars["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert device.env_vars["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert device.env_vars["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert device.env_vars["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"

    prod_templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/prod/llm.yaml"
    )
    assert not any(
        template.impl.impl_id == "quetzal" and "openai/gpt-oss-120b" in template.weights
        for template in prod_templates
    )
