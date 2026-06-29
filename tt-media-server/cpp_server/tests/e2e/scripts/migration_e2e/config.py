# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Single source of truth for test configuration.

Centralizes the env-var contract used by the MPI + Kafka e2e tests. New tests
should call `load_config()` (or construct a `Config` directly) instead of
reading `os.environ` ad hoc, so the contract stays in one place.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from migration_e2e import SCRIPTS_DIR


@dataclass
class Config:
    """Test knobs with dev-friendly defaults.

    CMake injects WORKER_BIN via ctest's ENVIRONMENT property; everything
    else mirrors the legacy shell orchestrator's defaults.
    """

    num_prefill: int
    num_decode: int
    worker_bin: pathlib.Path
    kafka_brokers: str
    http_port: int
    mc_bind_address: str
    host_dram_bytes: int
    discovery_timeout_sec: int
    ack_timeout_sec: float
    metadata_override: str | None
    prefill_log: pathlib.Path
    decode_log: pathlib.Path
    meta_log: pathlib.Path

    @property
    def num_workers(self) -> int:
        return self.num_prefill + self.num_decode

    @property
    def rank_launch_script(self) -> pathlib.Path:
        return SCRIPTS_DIR / "migration_worker_rank_launch.sh"

    @property
    def metadata_server_script(self) -> pathlib.Path:
        # tests/e2e/scripts/ -> tests/e2e/ -> tests/integration/...
        return (
            SCRIPTS_DIR.parent.parent
            / "integration"
            / "run_mooncake_metadata_server.sh"
        )


def load_config() -> Config:
    def _int(key: str, default: int) -> int:
        return int(os.environ.get(key, str(default)))

    def _float(key: str, default: float) -> float:
        return float(os.environ.get(key, str(default)))

    worker_bin_raw = os.environ.get("WORKER_BIN", "./build/bringup_mooncake_worker")
    return Config(
        num_prefill=_int("NUM_PREFILL", 4),
        num_decode=_int("NUM_DECODE", 16),
        worker_bin=pathlib.Path(worker_bin_raw).resolve(),
        kafka_brokers=os.environ.get("KAFKA_BROKERS", "kafka:9092"),
        http_port=_int("HTTP_PORT", 18083),
        mc_bind_address=os.environ.get("MC_BIND_ADDRESS", "127.0.0.1"),
        host_dram_bytes=_int("HOST_DRAM_BYTES", 1 << 20),
        discovery_timeout_sec=_int("DISCOVERY_TIMEOUT_SEC", 60),
        ack_timeout_sec=_float("ACK_TIMEOUT_SEC", 30.0),
        metadata_override=os.environ.get("METADATA") or None,
        prefill_log=pathlib.Path(
            os.environ.get("PREFILL_LOG", "/tmp/tt_mc_mpi_prefill.log")
        ),
        decode_log=pathlib.Path(
            os.environ.get("DECODE_LOG", "/tmp/tt_mc_mpi_decode.log")
        ),
        meta_log=pathlib.Path(
            os.environ.get("META_LOG", "/tmp/tt_mc_metadata_kafka_mpi.log")
        ),
    )
