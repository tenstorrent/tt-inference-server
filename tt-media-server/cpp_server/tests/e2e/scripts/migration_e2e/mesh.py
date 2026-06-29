# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Mesh readiness via worker-log pattern match.

The bringup_mooncake_worker process emits two milestones we tail for:

  "CONNECTED to N peers"           PeerDiscoveryService announces the mesh
                                   is up (every worker).
  "entering KV-migration loop"     Kafka clients constructed and ready
                                   (prefill workers only; decode workers
                                   pass --no-kafka and never log this).

The mesh is "ready" when both lines are present at the expected counts.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass

from migration_e2e.config import Config


@dataclass
class MeshState:
    prefill_connected: int = 0
    decode_connected: int = 0
    prefill_kafka_ready: int = 0

    def is_ready(self, cfg: Config) -> bool:
        return (
            self.prefill_connected >= cfg.num_prefill
            and self.decode_connected >= cfg.num_decode
            and self.prefill_kafka_ready >= cfg.num_prefill
        )


def _count_occurrences(path: pathlib.Path, needle: str) -> int:
    try:
        return path.read_text(errors="replace").count(needle)
    except FileNotFoundError:
        return 0


def read_mesh_state(cfg: Config) -> MeshState:
    return MeshState(
        prefill_connected=_count_occurrences(cfg.prefill_log, "CONNECTED to"),
        decode_connected=_count_occurrences(cfg.decode_log, "CONNECTED to"),
        prefill_kafka_ready=_count_occurrences(
            cfg.prefill_log, "entering KV-migration loop"
        ),
    )


def wait_for_mesh(
    cfg: Config,
    prefill_proc: subprocess.Popen[bytes],
    decode_proc: subprocess.Popen[bytes],
) -> MeshState:
    """Poll worker logs until the mesh is ready or one mpirun exits early.

    A live mpirun that hasn't yet finished bringup keeps the poll going;
    an exited mpirun is a hard failure signal — the caller should fall
    through to RESULT: FAIL and tail the logs.
    """
    print(
        f"Waiting up to {cfg.discovery_timeout_sec}s for "
        f"{cfg.num_workers} CONNECTED + {cfg.num_prefill} KV-migration-loop "
        f"lines..."
    )
    deadline = time.monotonic() + cfg.discovery_timeout_sec + 10
    while time.monotonic() < deadline:
        state = read_mesh_state(cfg)
        if state.is_ready(cfg):
            return state
        if prefill_proc.poll() is not None or decode_proc.poll() is not None:
            print(
                "ERROR: one mpirun exited before mesh was up.", file=sys.stderr
            )
            break
        time.sleep(1.0)
    return read_mesh_state(cfg)


def tail(path: pathlib.Path, lines: int = 40) -> None:
    """Dump last N lines of a log to stderr, for failure debugging."""
    if not path.exists():
        return
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    print(f"--- last {lines} lines of {path} ---", file=sys.stderr)
    for line in content[-lines:]:
        print(line, file=sys.stderr)
