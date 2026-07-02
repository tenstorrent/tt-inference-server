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
  4. Produces ONE migration request and verifies NUM_PREFILL SUCCESSFUL acks
     arrive within ACK_TIMEOUT_SEC.

Each prefill worker is launched in its own Kafka consumer group (see
migration_worker_rank_launch.sh), so a single request fans out to all
prefills — one ack per worker is the contract this test asserts.

Configuration is read from env vars (see migration_e2e.config). Exit codes:
0 PASS, 1 FAIL (ack mismatch), 2 misconfig/preflight.

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

from migration_e2e.acks import count_acks, produce_migration_request  # noqa: E402
from migration_e2e.config import load_config  # noqa: E402
from migration_e2e.metadata_server import start_metadata_server  # noqa: E402
from migration_e2e.verify_test_environment import (  # noqa: E402
    TestEnvironmentError,
    verify_test_environment,
)
from migration_e2e.workers import (  # noqa: E402
    launch_role,
    sweep_stragglers,
    terminate,
)


def main() -> int:
    cfg = load_config()
    try:
        verify_test_environment(cfg)
    except TestEnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    with ExitStack() as stack:
        # LIFO cleanup: this runs LAST. It sweeps any straggler bringup
        # workers that survived the per-process termination above.
        stack.callback(sweep_stragglers)

        try:
            meta_proc, metadata_uri = start_metadata_server(cfg)
        except TestEnvironmentError as exc:
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

        migration_id, ack_consumer = produce_migration_request(cfg)
        stack.callback(ack_consumer.close)

        if not count_acks(cfg, migration_id, ack_consumer):
            print("RESULT: FAIL", file=sys.stderr)
            return 1

        print("RESULT: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
