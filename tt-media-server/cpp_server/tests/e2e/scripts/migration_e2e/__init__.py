# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""migration_e2e: shared building blocks for cpp_server MPI + Kafka e2e tests.

Each module owns one slice of the test lifecycle so new e2e tests can
reuse them à la carte:

    config           env-var contract -> typed Config
    preflight        worker_bin / mpirun / Kafka topic checks
    metadata_server  Mooncake HTTP metadata service lifecycle
    workers          mpirun launch + cleanup for prefill / decode roles
    mesh             readiness detection via worker-log pattern match
    acks             one-shot request producer + ack counter

Importing the package triggers a venv re-exec when confluent_kafka isn't
available under the current interpreter (see _bootstrap). The bootstrap runs
BEFORE any submodule top-level imports, so a fresh ctest invocation can pick
up cpp_server/.venv transparently.
"""

from __future__ import annotations

import pathlib

from ._bootstrap import ensure_confluent_kafka

ensure_confluent_kafka()

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = PACKAGE_DIR.parent

# App Kafka topic names. Match what `scripts/migration_cli.py setup`
# provisions and what the cpp_server workers produce/consume.
REQUEST_TOPIC = "kv-migration-requests"
ACK_TOPIC = "kv-migration-acks"

__all__ = ["PACKAGE_DIR", "SCRIPTS_DIR", "REQUEST_TOPIC", "ACK_TOPIC"]
