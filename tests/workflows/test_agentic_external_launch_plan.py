# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import hashlib
import json
from copy import deepcopy

import pytest

from scripts.release.plan_agentic_external_run import (
    ContractError,
    build_contract,
    load_capability_receipt,
    verify_external_endpoint,
    write_plan,
)


_DIGESTS = {
    "codegen_fingerprint": "1" * 64,
    "weights_fingerprint": "2" * 64,
    "emit_hash": "3" * 64,
}


def _receipt(**capability_overrides):
    capabilities = {
        "max_input_tokens": 92 * 1024,
        "max_context_tokens": 128 * 1024,
        "max_concurrency": 1,
    }
    capabilities.update(capability_overrides)
    return {
        "schema": "ttis.external-generated-quetzal-capability/v1",
        "model_id": "openai/gpt-oss-120b",
        "served_model": "gpt-oss-120b@p150x4-b1",
        "implementation": "quetzal",
        "serving_backend": "generated_quetzal",
        "provider_policy": "generated_quetzal_only",
        "artifact_identity": {
            "model_id": "openai/gpt-oss-120b",
            "serving_backend": "generated_quetzal",
            **_DIGESTS,
        },
        "capabilities": capabilities,
    }


def _evidence(receipt=None):
    receipt = receipt or _receipt()
    return {
        "models": {
            "id": receipt["served_model"],
            "owned_by": "quetzal",
            "backend": "generated_quetzal",
            "model_id": receipt["model_id"],
        },
        "health": {
            "status": "ok",
            "backend": "quetzal",
            "provider_policy": "generated_quetzal_only",
            "resident": receipt["served_model"],
            "artifact_identity": deepcopy(receipt["artifact_identity"]),
        },
    }


def _gpt120(**overrides):
    receipt = overrides.pop("capability_receipt", _receipt())
    values = {
        "model": "gpt-oss-120b",
        "device": "p300x2",
        "task_name": "swe_bench_verified",
        "limit_samples_mode": "ci-nightly",
        "capability_receipt": receipt,
        "capability_receipt_sha256": "a" * 64,
        "endpoint_evidence": _evidence(receipt),
        "server_url": "http://qb2-120-p06t07",
        "service_port": 18091,
    }
    values.update(overrides)
    return build_contract(**values)


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return deepcopy(self.payload)


class _Session:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        key = "models" if url.endswith("/v1/models") else "health"
        payload = (
            {"object": "list", "data": [self.evidence[key]]}
            if key == "models"
            else self.evidence[key]
        )
        return _Response(payload)


def test_gpt120_plan_pins_exact_swe_shape_and_writes_quetzal_argv(tmp_path):
    contract, model_spec = _gpt120()
    assert contract.concurrency == 1
    assert contract.max_input_tokens == 92 * 1024
    assert contract.max_output_tokens == 32 * 1024
    assert contract.required_context_tokens == 124 * 1024
    assert contract.catalog_max_context_tokens == 128 * 1024
    assert contract.implementation == "quetzal"
    assert contract.serving_backend == "generated_quetzal"
    assert len(contract.instance_ids) == 5

    plan_path, runtime_path, command = write_plan(contract, model_spec, tmp_path)
    plan = json.loads(plan_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    assert plan["argv"] == command
    assert command[command.index("--model") + 1] == "gpt-oss-120b"
    assert command[command.index("--server-url") + 1] == "http://qb2-120-p06t07"
    assert runtime["runtime_config"]["limit_samples_mode"] == "ci-nightly"
    assert runtime["runtime_config"]["workflow"] == "agentic"
    assert runtime["runtime_config"]["impl"] == "quetzal"
    assert runtime["runtime_model_spec"]["impl"]["impl_id"] == "quetzal"
    assert runtime["runtime_model_spec"]["impl"]["repo_url"] == (
        "https://github.com/tenstorrent/tt-quetzalcoatlus"
    )
    assert runtime["runtime_model_spec"]["impl"]["code_path"] == (
        "serving/quetzal_vllm.py"
    )
    assert runtime["runtime_model_spec"]["docker_image"] is None
    assert runtime["runtime_model_spec"]["code_link"] is None
    assert len(plan["contract"]["endpoint_evidence_sha256"]) == 64
    assert plan["contract"]["endpoint_evidence"] == _evidence()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("max_input_tokens", 92 * 1024 - 1, "input tokens"),
        ("max_context_tokens", 124 * 1024 - 1, "total tokens"),
        ("max_concurrency", 0, "positive integer"),
    ],
)
def test_gpt120_plan_rejects_under_admitted_receipt(field, value, match):
    with pytest.raises(ContractError, match=match):
        _gpt120(capability_receipt=_receipt(**{field: value}))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("implementation", "gpt_oss", "implementation"),
        ("serving_backend", "native_ttnn", "serving_backend"),
        (
            "provider_policy",
            "generated_quetzal_plus_native_diagnostic",
            "provider_policy",
        ),
        ("model_id", "other/model", "model_id"),
    ],
)
def test_plan_rejects_non_quetzal_or_wrong_receipt(field, value, match):
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(ContractError, match=match):
        _gpt120(capability_receipt=receipt)


def test_receipt_bytes_are_bound_to_required_sha256(tmp_path):
    path = tmp_path / "receipt.json"
    payload = json.dumps(_receipt(), sort_keys=True).encode()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    receipt, actual = load_capability_receipt(path, digest)
    assert receipt == _receipt()
    assert actual == digest
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        load_capability_receipt(path, "f" * 64)


def test_endpoint_must_match_model_backend_policy_and_artifact():
    receipt = _receipt()
    evidence = _evidence(receipt)
    session = _Session(evidence)
    assert verify_external_endpoint(
        server_url="http://qb2", service_port=18091, receipt=receipt, session=session
    ) == evidence
    assert [url.rsplit("/", 1)[-1] for url, _ in session.calls] == ["models", "health"]

    for container, field, value, match in [
        ("models", "model_id", "other/model", "model_id mismatch"),
        ("models", "backend", "native_ttnn", "backend mismatch"),
        (
            "health",
            "provider_policy",
            "generated_quetzal_plus_native_diagnostic",
            "provider_policy mismatch",
        ),
        ("health", "resident", "other", "resident mismatch"),
    ]:
        bad = _evidence(receipt)
        bad[container][field] = value
        with pytest.raises(ContractError, match=match):
            verify_external_endpoint(
                server_url="http://qb2",
                service_port=18091,
                receipt=receipt,
                session=_Session(bad),
            )

    bad = _evidence(receipt)
    bad["health"]["artifact_identity"]["emit_hash"] = "9" * 64
    with pytest.raises(ContractError, match="emit_hash mismatch"):
        verify_external_endpoint(
            server_url="http://qb2",
            service_port=18091,
            receipt=receipt,
            session=_Session(bad),
        )


def test_plan_rejects_non_agentic_task():
    with pytest.raises(ContractError, match="not an agentic eval"):
        _gpt120(task_name="mmlu_generative")
