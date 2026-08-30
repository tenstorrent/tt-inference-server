# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Fail-closed host admission for the topology-qualified GPT Quetzal lane."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

_MODEL_IDS = {"gpt-oss-120b", "openai/gpt-oss-120b"}
_EVIDENCE_SCHEMA = "quetzal.topology-evidence.v1"
_DESCRIPTOR_SHA256 = "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792"
_SMOKE_SHA256 = "bf3311c685554105cb420239467f4e5c32e294be57b2a34fc6cbf7b0b84573fa"
_SELECTION_SHA256 = "1852bfcc4a4acd234b83de0ce1b174b3334daa5f6f0361f835564a26f26291a7"
_EMIT_SHA256 = "f296b7049ad6c9bfb3876f51c5cd1e717b19ebb0a667585907779ef45019370d"
_MAX_AGE_SECONDS = 900
_REQUIRED_FIELD_PROVENANCE = {
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
}
_CLAIM_BOUNDARY = (
    "mesh lifecycle, count, shape, and degree histograms are observed; "
    "Ring and links=2 are selected qualified-artifact configuration"
)


class QuetzalTopologyAdmissionError(RuntimeError):
    """The fresh pre-weight topology admission contract was not satisfied."""


def _read_regular_file_once(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        value = os.fstat(stream.fileno())
        if not stat.S_ISREG(value.st_mode):
            raise QuetzalTopologyAdmissionError(f"not a regular file: {path}")
        return stream.read()


def _mapping(value, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise QuetzalTopologyAdmissionError(f"{field} must be an object")
    return value


def _exact(value, expected, field: str) -> None:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise QuetzalTopologyAdmissionError(
            f"{field} mismatch: expected {expected!r}, observed {value!r}"
        )


def _sha256(value, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise QuetzalTopologyAdmissionError(f"{field} must be a lowercase SHA-256")


def _absolute_path(value, field: str) -> None:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise QuetzalTopologyAdmissionError(f"{field} must be an absolute path")


def _utc(value, field: str) -> datetime:
    if not isinstance(value, str):
        raise QuetzalTopologyAdmissionError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QuetzalTopologyAdmissionError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise QuetzalTopologyAdmissionError(f"{field} must use UTC")
    return parsed.astimezone(timezone.utc)


def _inside_runner_temp(path: Path, environment: Mapping[str, str]) -> Path:
    runner_temp = environment.get("RUNNER_TEMP", "")
    if not runner_temp:
        raise QuetzalTopologyAdmissionError("RUNNER_TEMP is required")
    resolved = path.resolve()
    try:
        resolved.relative_to(Path(runner_temp).resolve())
    except ValueError as exc:
        raise QuetzalTopologyAdmissionError(
            f"admission artifacts must be under RUNNER_TEMP: {resolved}"
        ) from exc
    return resolved


def _live_identity(environment: Mapping[str, str]) -> tuple[str, int]:
    node = socket.gethostname()
    try:
        job_id = int(environment.get("SLURM_JOB_ID", ""))
    except ValueError as exc:
        raise QuetzalTopologyAdmissionError("live SLURM_JOB_ID is required") from exc
    if job_id <= 0:
        raise QuetzalTopologyAdmissionError("live SLURM_JOB_ID is required")
    slurmd_node = environment.get("SLURMD_NODENAME")
    if slurmd_node and slurmd_node != node:
        raise QuetzalTopologyAdmissionError(
            f"SLURMD_NODENAME mismatch: {slurmd_node!r} != {node!r}"
        )
    return node, job_id


def _require_current_slurm(node: str, job_id: int) -> None:
    result = subprocess.run(
        ["scontrol", "show", "job", "--oneliner", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise QuetzalTopologyAdmissionError(f"scontrol failed: {result.stderr.strip()}")
    fields = dict(
        token.split("=", 1) for token in result.stdout.split() if "=" in token
    )
    _exact(fields.get("JobState"), "RUNNING", "live Slurm JobState")
    _exact(fields.get("NodeList"), node, "live Slurm NodeList")


def _require_zero_holders() -> None:
    devices = sorted(glob.glob("/dev/tenstorrent/[0-9]*"))
    if len(devices) != 4:
        raise QuetzalTopologyAdmissionError(
            f"expected four Tenstorrent device nodes, observed {devices!r}"
        )
    fuser = shutil.which("fuser")
    if fuser is None:
        raise QuetzalTopologyAdmissionError("fuser is required")
    result = subprocess.run(
        [fuser, *devices], check=False, capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        raise QuetzalTopologyAdmissionError(
            "Tenstorrent device holder(s) appeared after topology admission: "
            + (result.stdout + result.stderr).strip()
        )
    if result.returncode != 1:
        raise QuetzalTopologyAdmissionError(
            f"fuser holder scan failed with exit {result.returncode}"
        )


def _is_target(model_spec) -> bool:
    if getattr(getattr(model_spec, "impl", None), "impl_id", None) != "quetzal":
        return False
    identities = {
        getattr(model_spec, "model_name", None),
        getattr(model_spec, "hf_model_repo", None),
        getattr(model_spec, "hf_weights_repo", None),
    }
    return bool(identities & _MODEL_IDS)


def _validate_canonical_evidence(
    evidence: Mapping,
    admission: Mapping,
    *,
    node: str,
    job_id: int,
) -> None:
    _exact(
        set(evidence),
        {
            "schema",
            "captured_at_utc",
            "node",
            "slurm_job_id",
            "slurm_state",
            "chip_count",
            "weights_loaded",
            "provenance",
            "mesh_lifecycle",
            "topology",
            "producer",
        },
        "evidence canonical fields",
    )
    for field, expected in (
        ("schema", _EVIDENCE_SCHEMA),
        ("captured_at_utc", admission.get("captured_at_utc")),
        ("node", node),
        ("slurm_job_id", job_id),
        ("slurm_state", "RUNNING"),
        ("chip_count", admission.get("chip_count")),
        ("weights_loaded", False),
    ):
        _exact(evidence.get(field), expected, f"evidence.{field}")
    _utc(evidence.get("captured_at_utc"), "evidence.captured_at_utc")

    provenance = _mapping(evidence.get("provenance"), "evidence.provenance")
    _exact(
        dict(provenance),
        _REQUIRED_FIELD_PROVENANCE,
        "evidence.provenance",
    )

    lifecycle = _mapping(evidence.get("mesh_lifecycle"), "evidence.mesh_lifecycle")
    _exact(
        dict(lifecycle),
        {
            "opened": True,
            "synchronized": True,
            "closed": True,
            "exit_code": 0,
            "device_holders_after": admission.get("device_holders_after"),
        },
        "evidence.mesh_lifecycle",
    )

    topology = _mapping(evidence.get("topology"), "evidence.topology")
    _exact(
        dict(topology),
        {
            "mesh_shape": admission.get("mesh_shape"),
            "logical_degree_histogram": admission.get("logical_degree_histogram"),
            "physical_degree_histogram": admission.get("physical_degree_histogram"),
            "descriptor_sha256": admission.get("descriptor_sha256"),
            "collective_topology": admission.get("collective_topology"),
            "collective_num_links": admission.get("collective_num_links"),
        },
        "evidence.topology",
    )

    producer = _mapping(evidence.get("producer"), "evidence.producer")
    _exact(
        set(producer),
        {
            "schema",
            "smoke_script_path",
            "smoke_script_sha256",
            "smoke_log_path",
            "smoke_log_sha256",
            "descriptor_path",
            "descriptor_sha256",
            "qualified_selection_path",
            "qualified_selection_sha256",
            "selected_model_id",
            "selected_emit_sha256",
            "claim_boundary",
        },
        "evidence.producer canonical fields",
    )
    for field, expected in (
        ("schema", "quetzal.topology-evidence-producer.v1"),
        ("smoke_script_sha256", _SMOKE_SHA256),
        ("qualified_selection_sha256", _SELECTION_SHA256),
        ("descriptor_sha256", admission.get("descriptor_sha256")),
        ("selected_model_id", "openai/gpt-oss-120b"),
        ("selected_emit_sha256", _EMIT_SHA256),
        ("claim_boundary", _CLAIM_BOUNDARY),
    ):
        _exact(producer.get(field), expected, f"evidence.producer.{field}")
    _sha256(producer.get("smoke_log_sha256"), "evidence.producer.smoke_log_sha256")
    for field in (
        "smoke_script_path",
        "smoke_log_path",
        "descriptor_path",
        "qualified_selection_path",
    ):
        _absolute_path(producer.get(field), f"evidence.producer.{field}")


def validate_gpt120_quetzal_preweight_admission(
    model_spec,
    *,
    environment: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict | None:
    """Require a fresh same-allocation receipt before host weight setup.

    Native GPT and every non-target Quetzal model are deliberately untouched.
    """
    if not _is_target(model_spec):
        return None
    environment = environment or os.environ
    raw_path = environment.get("QUETZAL_TOPOLOGY_ADMISSION_JSON", "")
    if not raw_path:
        raise QuetzalTopologyAdmissionError(
            "GPT-OSS-120B Quetzal requires QUETZAL_TOPOLOGY_ADMISSION_JSON"
        )
    admission_path = _inside_runner_temp(Path(raw_path), environment)
    admission_bytes = _read_regular_file_once(admission_path)
    try:
        admission = _mapping(json.loads(admission_bytes), "admission")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuetzalTopologyAdmissionError("admission is not JSON") from exc

    node, job_id = _live_identity(environment)
    _require_current_slurm(node, job_id)
    for field, expected in (
        ("schema", "quetzal.topology-admission-result.v1"),
        ("status", "pass"),
        ("node", node),
        ("slurm_job_id", job_id),
        ("chip_count", 4),
        ("mesh_shape", [2, 2]),
        ("logical_degree_histogram", {"2": 4}),
        ("physical_degree_histogram", {"2": 4}),
        ("descriptor_sha256", _DESCRIPTOR_SHA256),
        ("collective_topology", "Ring"),
        ("collective_num_links", 2),
        ("device_holders_after", 0),
        ("weights_loaded_at_capture", False),
    ):
        _exact(admission.get(field), expected, f"admission.{field}")

    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    captured_at = _utc(admission.get("captured_at_utc"), "admission.captured_at_utc")
    verified_at = _utc(admission.get("verified_at_utc"), "admission.verified_at_utc")
    if verified_at < captured_at:
        raise QuetzalTopologyAdmissionError("admission verification predates capture")
    age = (observed_now - captured_at).total_seconds()
    if age < -30 or age > _MAX_AGE_SECONDS:
        raise QuetzalTopologyAdmissionError(
            f"topology admission is not fresh: age={age:.3f}s"
        )

    evidence_path = _inside_runner_temp(
        Path(str(admission.get("evidence_path", ""))), environment
    )
    evidence_bytes = _read_regular_file_once(evidence_path)
    _exact(
        hashlib.sha256(evidence_bytes).hexdigest(),
        admission.get("evidence_sha256"),
        "admission.evidence_sha256",
    )
    try:
        evidence = _mapping(json.loads(evidence_bytes), "evidence")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuetzalTopologyAdmissionError("evidence is not JSON") from exc
    _validate_canonical_evidence(evidence, admission, node=node, job_id=job_id)
    _require_zero_holders()
    return {
        "status": "pass",
        "node": node,
        "slurm_job_id": job_id,
        "captured_at_utc": admission["captured_at_utc"],
        "admission_sha256": hashlib.sha256(admission_bytes).hexdigest(),
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "weights_loaded": False,
    }
