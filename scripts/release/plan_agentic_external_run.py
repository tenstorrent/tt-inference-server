#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Build a fail-closed launch contract for an external agentic endpoint.

This does not contact or start a server.  The caller must supply the capacities
admitted by the serving artifact; values smaller than the configured workload
are rejected before a long-running SWE/Terminal-Bench harness is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_module.drivers.agentic import (  # noqa: E402
    resolve_instance_ids,
    resolve_n_tasks,
)
from reference_config.evals.eval_config import EVAL_CONFIGS  # noqa: E402
from workflows.model_spec import (  # noqa: E402
    get_model_id,
    get_runtime_model_spec,
    quetzal_impl,
)
from workflows.runtime_config import RuntimeConfig  # noqa: E402
from workflows.workflow_types import WorkflowVenvType  # noqa: E402


class ContractError(ValueError):
    """The endpoint admission evidence cannot satisfy the configured eval."""


_CAPABILITY_SCHEMA = "ttis.external-generated-quetzal-capability/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QUETZAL_TARGET_MESH_BY_DEVICE = {
    # P300X2/QB2 is four physically discovered P150 chips. Keep this hardware
    # mapping independent of model IDs; the receipt still binds the exact
    # generated artifact to the endpoint's reported target_mesh.
    "p300x2": "p150x4",
}


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256")
    return value


def load_capability_receipt(path: Path, expected_sha256: str) -> tuple[dict, str]:
    """Load an immutable, caller-pinned generated-Quetzal capability receipt."""
    expected = _require_sha256(expected_sha256, "capability receipt SHA-256")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read capability receipt {path}: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ContractError(
            f"capability receipt SHA-256 mismatch: expected {expected}, got {actual}"
        )
    try:
        receipt = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContractError("capability receipt is not valid JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != _CAPABILITY_SCHEMA:
        raise ContractError(f"capability receipt schema must be {_CAPABILITY_SCHEMA!r}")
    return receipt, actual


def _validate_capability_receipt(receipt: dict, expected_model: str) -> None:
    if receipt.get("model_id") != expected_model:
        raise ContractError(f"capability receipt model_id must be {expected_model!r}")
    if receipt.get("implementation") != "quetzal":
        raise ContractError("capability receipt implementation must be 'quetzal'")
    if receipt.get("serving_backend") != "generated_quetzal":
        raise ContractError(
            "capability receipt serving_backend must be 'generated_quetzal'"
        )
    if receipt.get("provider_policy") != "generated_quetzal_only":
        raise ContractError(
            "capability receipt provider_policy must be 'generated_quetzal_only'"
        )
    served_model = receipt.get("served_model")
    if not isinstance(served_model, str) or not served_model:
        raise ContractError("capability receipt needs a non-empty served_model")
    identity = receipt.get("artifact_identity")
    if not isinstance(identity, dict):
        raise ContractError("capability receipt needs artifact_identity")
    if identity.get("model_id") != expected_model:
        raise ContractError("artifact_identity.model_id does not match the model")
    if identity.get("serving_backend") != "generated_quetzal":
        raise ContractError(
            "artifact_identity.serving_backend must be 'generated_quetzal'"
        )
    for field in (
        "codegen_fingerprint",
        "weights_fingerprint",
        "emit_hash",
        "prefill_emit_hash",
        "decode_emit_hash",
    ):
        _require_sha256(identity.get(field), f"artifact_identity.{field}")
    if identity.get("artifact_equivalence") != "exact":
        raise ContractError("artifact_identity.artifact_equivalence must be 'exact'")
    if identity.get("lossy_transformations") != []:
        raise ContractError("artifact_identity.lossy_transformations must be empty")
    if not isinstance(identity.get("target_mesh"), str) or not identity["target_mesh"]:
        raise ContractError("artifact_identity.target_mesh must be non-empty")
    if identity.get("batch_size") != 1:
        raise ContractError("artifact_identity.batch_size must be 1")
    capabilities = receipt.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ContractError("capability receipt needs capabilities")
    if capabilities.get("schema") != "ttq.serving_capabilities/v1":
        raise ContractError("capabilities.schema must be 'ttq.serving_capabilities/v1'")
    for field in (
        "max_context_tokens",
        "max_concurrency",
        "batch_size",
        "chunk_size",
        "kv_blocks",
    ):
        value = capabilities.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ContractError(f"capabilities.{field} must be a positive integer")
    if capabilities.get("batch_size") != 1:
        raise ContractError("capabilities.batch_size must be 1")
    if capabilities.get("chunked_prefill") is not True:
        raise ContractError("capabilities.chunked_prefill must be true")
    physical_capacity = capabilities["chunk_size"] * capabilities["kv_blocks"]
    if physical_capacity != capabilities["max_context_tokens"]:
        raise ContractError("capability KV geometry differs from max_context_tokens")


def _endpoint_base(server_url: str, service_port: int) -> str:
    from utils.url_helpers import build_base_url

    return build_base_url(server_url.rstrip("/"), service_port)


def _expected_endpoint_evidence(receipt: dict) -> dict:
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
            "artifact_identity": dict(receipt["artifact_identity"]),
            "serving_capabilities": dict(receipt["capabilities"]),
        },
    }


def _evidence_sha256(evidence: dict) -> str:
    payload = json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def receipt_from_contract(contract: dict) -> dict:
    """Reconstruct the already-verified receipt fields needed for drift checks."""
    evidence = contract.get("endpoint_evidence")
    if not isinstance(evidence, dict):
        raise ContractError("launch contract has no endpoint_evidence")
    expected_digest = contract.get("endpoint_evidence_sha256")
    if _evidence_sha256(evidence) != expected_digest:
        raise ContractError("launch contract endpoint evidence digest mismatch")
    health = evidence.get("health")
    if not isinstance(health, dict):
        raise ContractError("launch contract has no endpoint health evidence")
    receipt = {
        "schema": _CAPABILITY_SCHEMA,
        "model_id": contract.get("hf_model_repo"),
        "served_model": contract.get("served_model"),
        "implementation": contract.get("implementation"),
        "serving_backend": contract.get("serving_backend"),
        "provider_policy": contract.get("provider_policy"),
        "artifact_identity": contract.get("artifact_identity"),
        "capabilities": health.get("serving_capabilities"),
    }
    _validate_capability_receipt(receipt, contract.get("hf_model_repo"))
    return receipt


def verify_launch_contract_endpoint(
    contract_path: Path, timeout_sec: float = 10.0
) -> dict:
    """Re-probe an endpoint and reject any drift from a persisted launch plan."""
    try:
        document = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read launch contract {contract_path}: {exc}"
        ) from exc
    contract = document.get("contract") if isinstance(document, dict) else None
    if not isinstance(contract, dict):
        raise ContractError("launch contract document has no contract object")
    receipt = receipt_from_contract(contract)
    evidence = verify_external_endpoint(
        server_url=contract["server_url"],
        service_port=contract["service_port"],
        receipt=receipt,
        timeout_sec=timeout_sec,
    )
    if _evidence_sha256(evidence) != contract["endpoint_evidence_sha256"]:
        raise ContractError("external generated-Quetzal endpoint identity drifted")
    return evidence


def verify_external_endpoint(
    *,
    server_url: str,
    service_port: int,
    receipt: dict,
    timeout_sec: float = 10.0,
    session=requests,
) -> dict:
    """Bind a live endpoint to the exact generated artifact in ``receipt``."""
    model_id = receipt.get("model_id")
    _validate_capability_receipt(receipt, model_id)
    base = _endpoint_base(server_url, service_port)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    def get_json(path: str) -> dict:
        try:
            response = session.get(base + path, headers=headers, timeout=timeout_sec)
            response.raise_for_status()
            value = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ContractError(
                f"endpoint identity probe {path} failed: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(
                f"endpoint identity probe {path} returned non-object JSON"
            )
        return value

    models = get_json("/v1/models")
    rows = models.get("data")
    if not isinstance(rows, list):
        raise ContractError("endpoint /v1/models response has no data list")
    served_model = receipt["served_model"]
    matching = [
        row for row in rows if isinstance(row, dict) and row.get("id") == served_model
    ]
    if len(matching) != 1:
        raise ContractError(
            f"endpoint must advertise exactly one served model {served_model!r}"
        )
    model_row = matching[0]
    expected_row = {
        "owned_by": "quetzal",
        "backend": "generated_quetzal",
        "model_id": model_id,
    }
    for field, expected in expected_row.items():
        if model_row.get(field) != expected:
            raise ContractError(
                f"endpoint model {field} mismatch: expected {expected!r}, "
                f"got {model_row.get(field)!r}"
            )

    health = get_json("/health")
    expected_health = {
        "status": "ok",
        "backend": "quetzal",
        "provider_policy": "generated_quetzal_only",
        "resident": served_model,
    }
    for field, expected in expected_health.items():
        if health.get(field) != expected:
            raise ContractError(
                f"endpoint health {field} mismatch: expected {expected!r}, "
                f"got {health.get(field)!r}"
            )
    actual_identity = health.get("artifact_identity")
    if not isinstance(actual_identity, dict):
        raise ContractError("endpoint health has no artifact_identity")
    expected_identity = receipt["artifact_identity"]
    for field, expected in expected_identity.items():
        if actual_identity.get(field) != expected:
            raise ContractError(
                f"endpoint artifact_identity.{field} mismatch: expected "
                f"{expected!r}, got {actual_identity.get(field)!r}"
            )
    if health.get("serving_capabilities") != receipt["capabilities"]:
        raise ContractError(
            "endpoint serving_capabilities do not exactly match the pinned receipt"
        )
    # Persist only the closed identity fields. Runtime counters and other
    # mutable health fields must not make a post-run drift check ambiguous.
    return _expected_endpoint_evidence(receipt)


@dataclass(frozen=True)
class AgenticLaunchContract:
    model: str
    hf_model_repo: str
    device: str
    task: str
    limit_samples_mode: str
    concurrency: int
    instance_ids: list[str]
    n_tasks: int | None
    max_input_tokens: int
    max_output_tokens: int
    required_context_tokens: int
    catalog_max_context_tokens: int
    implementation: str
    serving_backend: str
    provider_policy: str
    served_model: str
    capability_receipt_sha256: str
    artifact_identity: dict
    endpoint_evidence: dict
    endpoint_evidence_sha256: str
    admitted_max_input_tokens: int
    admitted_max_context_tokens: int
    server_url: str
    service_port: int


def build_contract(
    *,
    model: str,
    device: str,
    task_name: str,
    limit_samples_mode: str,
    capability_receipt: dict,
    capability_receipt_sha256: str,
    endpoint_evidence: dict,
    server_url: str,
    service_port: int,
) -> tuple[AgenticLaunchContract, object]:
    catalog_spec, _, _ = get_runtime_model_spec(model=model, device=device)
    _validate_capability_receipt(capability_receipt, catalog_spec.hf_model_repo)
    expected_mesh = _QUETZAL_TARGET_MESH_BY_DEVICE.get(device.lower())
    if expected_mesh is None:
        raise ContractError(
            f"no generated-Quetzal target-mesh contract for device {device!r}"
        )
    actual_mesh = capability_receipt["artifact_identity"]["target_mesh"]
    if actual_mesh != expected_mesh:
        raise ContractError(
            f"artifact target_mesh must be {expected_mesh!r} for {device}, "
            f"got {actual_mesh!r}"
        )
    expected_endpoint_evidence = _expected_endpoint_evidence(capability_receipt)
    if endpoint_evidence != expected_endpoint_evidence:
        raise ContractError("closed endpoint identity evidence does not match receipt")
    capabilities = capability_receipt["capabilities"]
    admitted_max_context_tokens = capabilities["max_context_tokens"]
    model_spec = catalog_spec
    eval_config = EVAL_CONFIGS.get(model_spec.model_name)
    if eval_config is None:
        raise ContractError(f"no eval config for {model_spec.model_name!r}")
    matches = [task for task in eval_config.tasks if task.task_name == task_name]
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one task {task_name!r}, found {len(matches)}"
        )
    task = matches[0]
    if task.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC:
        raise ContractError(f"task {task_name!r} is not an agentic eval")
    cfg = task.swebench_eval_config or task.agentic_eval_config
    if cfg is None:
        raise ContractError(f"task {task_name!r} has no agentic configuration")

    max_output = cfg.max_output_tokens
    if max_output is None or max_output <= 0:
        raise ContractError(f"task {task_name!r} needs a finite positive output cap")
    if cfg.max_input_tokens <= 0:
        raise ContractError(f"task {task_name!r} needs a positive input cap")
    payload_context = cfg.max_input_tokens + max_output
    minimum_context = getattr(task, "min_context_required", None)
    if minimum_context is not None and (
        not isinstance(minimum_context, int)
        or isinstance(minimum_context, bool)
        or minimum_context <= 0
    ):
        raise ContractError(
            f"task {task_name!r} has invalid min_context_required {minimum_context!r}"
        )
    required_context = max(payload_context, minimum_context or 0)
    catalog_context = int(model_spec.device_model_spec.max_context)
    if required_context > catalog_context:
        raise ContractError(
            f"task requires {required_context} tokens but catalog declares {catalog_context}"
        )
    if admitted_max_context_tokens < required_context:
        raise ContractError(
            f"artifact admits {admitted_max_context_tokens} total tokens; "
            f"task requires {required_context}"
        )
    if cfg.n_concurrent_trials <= 0:
        raise ContractError("agentic concurrency must be positive")
    if cfg.n_concurrent_trials > model_spec.device_model_spec.max_concurrency:
        raise ContractError(
            f"task concurrency {cfg.n_concurrent_trials} exceeds catalog maximum "
            f"{model_spec.device_model_spec.max_concurrency}"
        )
    if cfg.n_concurrent_trials > capabilities["max_concurrency"]:
        raise ContractError(
            f"task concurrency {cfg.n_concurrent_trials} exceeds artifact maximum "
            f"{capabilities['max_concurrency']}"
        )

    runtime = SimpleNamespace(limit_samples_mode=limit_samples_mode)
    instance_ids = resolve_instance_ids(task, runtime)
    n_tasks = resolve_n_tasks(task, runtime)
    if task.swebench_eval_config is not None and not instance_ids and n_tasks is None:
        raise ContractError(
            f"mode {limit_samples_mode!r} selects neither fixed instances nor a task cap"
        )

    contract = AgenticLaunchContract(
        model=model_spec.model_name,
        hf_model_repo=model_spec.hf_model_repo,
        device=device,
        task=task_name,
        limit_samples_mode=limit_samples_mode,
        concurrency=cfg.n_concurrent_trials,
        instance_ids=instance_ids,
        n_tasks=n_tasks,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=max_output,
        required_context_tokens=required_context,
        catalog_max_context_tokens=catalog_context,
        implementation="quetzal",
        serving_backend="generated_quetzal",
        provider_policy="generated_quetzal_only",
        served_model=capability_receipt["served_model"],
        capability_receipt_sha256=_require_sha256(
            capability_receipt_sha256, "capability receipt SHA-256"
        ),
        artifact_identity=dict(capability_receipt["artifact_identity"]),
        endpoint_evidence=expected_endpoint_evidence,
        endpoint_evidence_sha256=_evidence_sha256(expected_endpoint_evidence),
        # Input admission is the exact workload policy, counted immediately
        # before dispatch. The endpoint independently proves total KV context;
        # no caller-authored max-input capacity is trusted.
        admitted_max_input_tokens=cfg.max_input_tokens,
        admitted_max_context_tokens=admitted_max_context_tokens,
        server_url=server_url.rstrip("/"),
        service_port=service_port,
    )
    return contract, model_spec


def write_plan(
    contract: AgenticLaunchContract, model_spec: object, output_dir: Path
) -> tuple[Path, Path, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "agentic_launch_contract.json"
    runtime = RuntimeConfig(
        model=contract.model,
        workflow="agentic",
        device=contract.device,
        service_port=str(contract.service_port),
        server_url=contract.server_url,
        limit_samples_mode=contract.limit_samples_mode,
        impl="quetzal",
        external_agentic_contract=str(plan_path),
    )
    # This spec is client-side metadata for an already-running external server.
    # Make the selected implementation explicit so the launch artifact cannot
    # silently serialize the catalog's native/default implementation.
    from dataclasses import replace

    model_spec = replace(
        model_spec,
        impl=quetzal_impl,
        model_id=get_model_id(
            quetzal_impl.impl_name, model_spec.model_name, contract.device
        ),
        tt_metal_commit=None,
        vllm_commit=None,
        version=None,
        docker_image=None,
        code_link=None,
    )
    runtime_path = runtime.to_json(
        model_spec,
        "external-agentic",
        model_spec.model_id,
        output_dir,
    )
    command = [
        sys.executable,
        str(_REPO_ROOT / "launchers" / "run_agentic.py"),
        "--model",
        contract.model,
        "--workflow",
        "agentic",
        "--device",
        contract.device,
        "--server-url",
        contract.server_url,
        "--service-port",
        str(contract.service_port),
        "--runtime-model-spec-json",
        str(runtime_path),
        "--output-dir",
        str(output_dir / "results"),
    ]
    plan_path.write_text(
        json.dumps(
            {"contract": asdict(contract), "argv": command}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path, runtime_path, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit-samples-mode", default="ci-nightly")
    parser.add_argument("--capability-receipt", type=Path, required=True)
    parser.add_argument("--capability-receipt-sha256", required=True)
    parser.add_argument("--identity-timeout-sec", type=float, default=10.0)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--service-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt, receipt_sha256 = load_capability_receipt(
            args.capability_receipt, args.capability_receipt_sha256
        )
        endpoint_evidence = verify_external_endpoint(
            server_url=args.server_url,
            service_port=args.service_port,
            receipt=receipt,
            timeout_sec=args.identity_timeout_sec,
        )
        contract, model_spec = build_contract(
            model=args.model,
            device=args.device,
            task_name=args.task,
            limit_samples_mode=args.limit_samples_mode,
            capability_receipt=receipt,
            capability_receipt_sha256=receipt_sha256,
            endpoint_evidence=endpoint_evidence,
            server_url=args.server_url,
            service_port=args.service_port,
        )
        plan_path, runtime_path, command = write_plan(
            contract, model_spec, args.output_dir
        )
    except (ContractError, ValueError) as exc:
        parser.error(str(exc))
    print(f"contract: {plan_path}")
    print(f"runtime: {runtime_path}")
    print(f"command: {shlex.join(command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
