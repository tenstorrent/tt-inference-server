# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""mpirun-based worker launch + cleanup.

The split-launch topology (one mpirun per role) lives in
migration_worker_rank_launch.sh; this module just wires the right env into
the right Popen for each role and provides cleanup primitives.
"""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess

from migration_e2e.config import Config


def _common_worker_env(cfg: Config, metadata_uri: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MC_TCP_BIND_ADDRESS": cfg.mc_bind_address,
            "WORKER_BIN": str(cfg.worker_bin),
            "METADATA": metadata_uri,
            "HOST_DRAM_BYTES": str(cfg.host_dram_bytes),
            "DISCOVERY_TIMEOUT_SEC": str(cfg.discovery_timeout_sec),
            "NUM_PREFILL": str(cfg.num_prefill),
            "NUM_DECODE": str(cfg.num_decode),
            "KAFKA_BROKERS": cfg.kafka_brokers,
        }
    )
    return env


def launch_role(
    cfg: Config,
    role: str,
    num_workers: int,
    log_path: pathlib.Path,
    metadata_uri: str,
) -> subprocess.Popen[bytes]:
    """Spawn `num_workers` ranks of one role under a fresh mpirun.

    start_new_session=True makes the mpirun (and its descendants) a new
    process group so `terminate()` can SIGTERM everything in one shot.
    """
    env = _common_worker_env(cfg, metadata_uri)
    env["WORKER_ROLE"] = role
    log_path.write_bytes(b"")  # truncate previous run's log
    log_fh = log_path.open("wb")

    # --oversubscribe: tests pack 20 ranks onto hosts with fewer cores.
    # --tag-output: every stdout line carries its MPI rank prefix.
    cmd = [
        "mpirun",
        "--oversubscribe",
        "--tag-output",
        "-np",
        str(num_workers),
        "bash",
        str(cfg.rank_launch_script),
    ]
    print(f"Launching {num_workers} {role} workers via mpirun...")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def terminate(proc: subprocess.Popen[bytes] | None) -> None:
    """SIGTERM a child started with start_new_session=True (whole group)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()


def sweep_stragglers() -> None:
    """Belt-and-braces sweep of bringup workers that outlived their mpirun.

    Matching on `--metadata` (which every worker invocation passes) avoids
    hitting launcher shells whose argv merely mentions the binary path.
    """
    subprocess.run(
        ["pkill", "-f", "bringup_mooncake_worker --metadata"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
