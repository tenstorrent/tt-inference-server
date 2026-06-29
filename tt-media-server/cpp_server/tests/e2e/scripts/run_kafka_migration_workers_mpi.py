#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""
MooncakeKafkaMigration single-host e2e test.

End-to-end exercise of the Kafka request path on top of the #4294 Mooncake
mesh. The test:

  1. Pre-flights the dev Kafka broker and topic provisioning.
  2. Starts the Mooncake HTTP metadata service.
  3. Launches NUM_PREFILL prefill workers (with Kafka) and NUM_DECODE decode
     workers (--no-kafka) via two `mpirun` invocations.
  4. Waits for all NUM_PREFILL + NUM_DECODE "CONNECTED to" lines and all
     NUM_PREFILL "entering KV-migration loop" lines.
  5. Produces ONE migration request and verifies NUM_PREFILL SUCCESSFUL acks
     arrive within ACK_TIMEOUT_SEC.

Each prefill worker is launched in its own Kafka consumer group (see
migration_worker_rank_launch.sh), so a single request fans out to all
prefills — one ack per worker is the contract this test asserts.

Configuration is read from env vars (see migration_e2e.config). Exit codes:
0 PASS, 1 FAIL (mesh/ack mismatch), 2 misconfig/preflight.

This file is intentionally thin: each lifecycle stage lives in its own
migration_e2e.* module so new tests can reuse the building blocks.
"""

from __future__ import annotations

import pathlib
import sys
from contextlib import ExitStack

# Ensure the package next to this script is on sys.path even when ctest invokes
# us with a CWD elsewhere (build/). Importing migration_e2e also runs the venv
# bootstrap, so this MUST precede all other migration_e2e imports below.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from migration_e2e.acks import produce_and_count_acks  # noqa: E402
from migration_e2e.config import load_config  # noqa: E402
from migration_e2e.mesh import tail, wait_for_mesh  # noqa: E402
from migration_e2e.metadata_server import start_metadata_server  # noqa: E402
from migration_e2e.preflight import PreflightError, preflight  # noqa: E402
from migration_e2e.workers import (  # noqa: E402
    launch_role,
    sweep_stragglers,
    terminate,
)


def main() -> int:
    cfg = load_config()
    try:
        preflight(cfg)
    except PreflightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    with ExitStack() as stack:
        # LIFO cleanup: this runs LAST. It sweeps any straggler bringup
        # workers that survived the per-process termination above.
        stack.callback(sweep_stragglers)

        try:
            meta_proc, metadata_uri = start_metadata_server(cfg)
        except PreflightError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if meta_proc is not None:
            stack.callback(terminate, meta_proc)

        prefill_proc = launch_role(
            cfg, "prefill", cfg.num_prefill, cfg.prefill_log, metadata_uri
        )
        stack.callback(terminate, prefill_proc)
        decode_proc = launch_role(
            cfg, "decode", cfg.num_decode, cfg.decode_log, metadata_uri
        )
        stack.callback(terminate, decode_proc)

        state = wait_for_mesh(cfg, prefill_proc, decode_proc)
        print("-" * 40)
        print(
            f"connected: prefill={state.prefill_connected}/{cfg.num_prefill} "
            f"decode={state.decode_connected}/{cfg.num_decode} "
            f"total={state.prefill_connected + state.decode_connected}"
            f"/{cfg.num_workers}"
        )
        print(
            f"prefill in KV-migration loop: "
            f"{state.prefill_kafka_ready}/{cfg.num_prefill}"
        )
        if not state.is_ready(cfg):
            print("RESULT: FAIL (mesh not ready)", file=sys.stderr)
            tail(cfg.prefill_log)
            tail(cfg.decode_log)
            return 1

        if not produce_and_count_acks(cfg):
            print("RESULT: FAIL", file=sys.stderr)
            tail(cfg.prefill_log)
            return 1

        print("RESULT: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
