# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Pre-launch environment checks.

Anything that can fail fast (worker binary missing, mpirun not on PATH, Kafka
unreachable or topics missing) is verified here before we spend time launching
workers. PreflightError is the single error type callers map to a misconfig
exit code (e.g. 2).
"""

from __future__ import annotations

import os
import shutil

from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient

from migration_e2e import ACK_TOPIC, REQUEST_TOPIC
from migration_e2e.config import Config


class PreflightError(RuntimeError):
    """Environment is not ready to run the test."""


def preflight(cfg: Config) -> None:
    if not cfg.worker_bin.is_file() or not os.access(cfg.worker_bin, os.X_OK):
        raise PreflightError(
            f"{cfg.worker_bin} not found/executable. "
            "Build it: ./build.sh --blaze --kafka --mooncake"
        )
    if shutil.which("mpirun") is None:
        raise PreflightError("mpirun not found; install Open MPI to run this test.")
    if not cfg.rank_launch_script.is_file():
        raise PreflightError(f"rank launcher missing: {cfg.rank_launch_script}")
    if not cfg.metadata_server_script.is_file():
        raise PreflightError(
            f"metadata server script missing: {cfg.metadata_server_script}"
        )

    print(f"Probing Kafka broker at {cfg.kafka_brokers}...")
    admin = AdminClient({"bootstrap.servers": cfg.kafka_brokers})
    try:
        metadata = admin.list_topics(timeout=5)
    except KafkaException as exc:
        raise PreflightError(
            f"Kafka at {cfg.kafka_brokers} not reachable: {exc}. "
            "Hint: bash scripts/dev-kafka.sh up"
        ) from exc

    missing = {REQUEST_TOPIC, ACK_TOPIC} - set(metadata.topics)
    if missing:
        raise PreflightError(
            f"missing Kafka topics: {sorted(missing)}. "
            "Hint: python scripts/migration_cli.py setup"
        )
    print("Kafka topics OK.")
