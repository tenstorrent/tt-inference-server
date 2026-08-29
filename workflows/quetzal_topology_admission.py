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
_DESCRIPTOR_SHA256 = "f4c9fb5acf307e1b320525007035ed9e75039f793e4350120365243682e37792"
_SMOKE_SHA256 = "bf3311c685554105cb420239467f4e5c32e294be57b2a34fc6cbf7b0b84573fa"
_SELECTION_SHA256 = "5ec9757ae74034c0cbc12569718c059b2b049416c736ad45a2048c5dda05b562"
_EMIT_SHA256 = "5cab85f26fe64fdea2a89c302f848a43152dcbd673133a1bfdfbf7054ba5862f"
_MAX_AGE_SECONDS = 900


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
    producer = _mapping(evidence.get("producer"), "evidence.producer")
    provenance = _mapping(evidence.get("provenance"), "evidence.provenance")
    for field, expected in (
        ("schema", "quetzal.topology-evidence-producer.v1"),
        ("smoke_script_sha256", _SMOKE_SHA256),
        ("qualified_selection_sha256", _SELECTION_SHA256),
        ("descriptor_sha256", _DESCRIPTOR_SHA256),
        ("selected_model_id", "openai/gpt-oss-120b"),
        ("selected_emit_sha256", _EMIT_SHA256),
    ):
        _exact(producer.get(field), expected, f"evidence.producer.{field}")
    _exact(
        provenance.get("physical_degree_histogram"),
        "tt_metal_topology_output",
        "evidence.provenance.physical_degree_histogram",
    )
    _exact(
        provenance.get("collective_topology"),
        "selected_qualified_artifact_configuration",
        "evidence.provenance.collective_topology",
    )
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
