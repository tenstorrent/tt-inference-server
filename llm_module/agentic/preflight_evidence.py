# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Compact, read-only disk evidence for release preflight; never infer a pass."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess


def _docker(*args):
    try:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=60
        )
        if result.returncode:
            return {
                "unavailable": "docker command failed",
                "returncode": result.returncode,
            }
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": type(exc).__name__}


def collect_disk_evidence(
    output: Path,
    phase: str,
    min_free_bytes: int | None = None,
    require_docker_root: bool = False,
    required_paths: dict[str, str] | None = None,
    min_free_inodes: int | None = None,
    required_path_min_free_bytes: dict[str, int] | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    docker_root = _docker("info", "--format", "{{.DockerRootDir}}")
    paths = {"jobs": str(output), "checkout": str(Path.cwd())}
    for key in ("STORAGE_PATH", "PERSISTENT_VOLUME_ROOT", "HOST_HF_HOME"):
        if os.environ.get(key):
            paths[key] = os.environ[key]
    if isinstance(docker_root, str) and docker_root:
        paths["docker_root"] = docker_root
    paths.update(required_paths or {})
    filesystems = {}
    for label, path in paths.items():
        try:
            usage = shutil.disk_usage(path)
            inode_usage = os.statvfs(path)
            filesystems[label] = {
                "path": path,
                "device_id": os.stat(path).st_dev,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_inodes": inode_usage.f_favail,
                "total_inodes": inode_usage.f_files,
            }
        except OSError as exc:
            filesystems[label] = {"path": path, "unavailable": type(exc).__name__}
    disk_summary = _docker("system", "df", "--format", "{{json .}}")
    if isinstance(disk_summary, str):
        try:
            disk_summary = [
                json.loads(line) for line in disk_summary.splitlines() if line
            ]
        except ValueError:
            disk_summary = {"unavailable": "unrecognized docker disk summary format"}
    data = {
        "phase": phase,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filesystems": filesystems,
        "docker_root": docker_root,
        "docker_disk_summary": disk_summary,
        "capacity_gate_passed": None,
        "limitation": "Observed storage only; selected task image/build footprint and concurrent writable reservations require separate accounting.",
    }
    if min_free_bytes is not None:
        required = ["jobs", "docker_root"] if require_docker_root else ["jobs"]
        required += list(required_paths or {})
        grouped = {}
        floors = {}
        inode_shortfalls = []
        missing = []
        for label in required:
            observation = filesystems.get(label, {})
            if "free_bytes" not in observation or "device_id" not in observation:
                missing.append(label)
                continue
            if (
                min_free_inodes is not None
                and observation.get("free_inodes", -1) < min_free_inodes
            ):
                inode_shortfalls.append(label)
            device = str(observation["device_id"])
            floor = (required_path_min_free_bytes or {}).get(label, min_free_bytes)
            floors[device] = max(floors.get(device, 0), floor)
            grouped[device] = min(
                grouped.get(device, observation["free_bytes"]),
                observation["free_bytes"],
            )
        passed = (
            not missing
            and not inode_shortfalls
            and bool(grouped)
            and all(value >= floors[device] for device, value in grouped.items())
        )
        data["reservation_check"] = {
            "minimum_free_bytes": min_free_bytes,
            "minimum_free_inodes": min_free_inodes,
            "minimum_free_bytes_by_device": floors,
            "role_specific_minimum_free_bytes": required_path_min_free_bytes or {},
            "inode_shortfalls": inode_shortfalls,
            "required_paths": required,
            "observed_free_bytes_by_device": grouped,
            "missing_paths": missing,
            "passed": passed,
            "basis": "Coordinator planning reserve, not a measured upper bound on task writes.",
        }
    (output / ("disk_" + phase + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    if min_free_bytes is not None and not data["reservation_check"]["passed"]:
        raise RuntimeError(
            "Disk reservation unavailable or insufficient; task invocation blocked"
        )
    return data


def public_tls_control(output: Path, phase: str = "before_tasks") -> dict:
    """Prove the fixed public TLS target works outside the task sandbox."""
    output.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": phase,
        "ip": "1.1.1.1",
        "port": 443,
        "server_name": "one.one.one.one",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verified_tls": False,
        "http_response": False,
    }
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=10) as raw:
            with ssl.create_default_context().wrap_socket(
                raw, server_hostname="one.one.one.one"
            ) as connection:
                data["verified_tls"] = True
                connection.sendall(
                    b"GET /cdn-cgi/trace HTTP/1.1\r\nHost: one.one.one.one\r\nConnection: close\r\n\r\n"
                )
                response = connection.recv(4096)
                data["http_response"] = response.startswith(b"HTTP/1.1 200")
    except (OSError, ValueError) as exc:
        data["error_type"] = type(exc).__name__
    (output / ("public_tls_control_" + phase + ".json")).write_text(
        json.dumps(data, indent=2) + "\n"
    )
    if not (data["verified_tls"] and data["http_response"]):
        raise RuntimeError(
            "Public TLS control unavailable; isolation smoke is inconclusive"
        )
    return data
