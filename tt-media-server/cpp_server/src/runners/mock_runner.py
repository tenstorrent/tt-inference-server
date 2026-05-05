#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mock runner for testing shared memory IPC without ttnn dependencies.

This script echoes back input tokens one by one to simulate token generation.
Useful for testing the C++ inference server integration without hardware.

Launch directly with:
    python mock_runner.py
"""

import os
import signal
import sys
import time

from shared_memory import DECODE_MAX_TOKEN_IDS, PREFILL_MAX_TOKEN_IDS, SharedMemory

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _run_mock_bridge() -> None:
    """Open shared-memory buffers and run the mock echo bridge."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")

    if not (c2p_name and p2c_name):
        print(
            "Error: TT_IPC_SHM_C2P or TT_IPC_SHM_P2C not set",
            file=sys.stderr,
        )
        print("Example usage:", file=sys.stderr)
        print("  export TT_IPC_SHM_C2P=tt_ipc_c2p_12345", file=sys.stderr)
        print("  export TT_IPC_SHM_P2C=tt_ipc_p2c_12345", file=sys.stderr)
        print("  python mock_runner.py", file=sys.stderr)
        sys.exit(1)

    print(
        f"Mock runner: Opening shared memory C2P={c2p_name}, P2C={p2c_name}",
        file=sys.stderr,
    )

    def is_shutdown() -> bool:
        return _shutdown

    try:
        with SharedMemory(
            c2p_name, max_token_ids=PREFILL_MAX_TOKEN_IDS, is_shutdown=is_shutdown
        ) as c2p, SharedMemory(
            p2c_name, max_token_ids=DECODE_MAX_TOKEN_IDS, is_shutdown=is_shutdown
        ) as p2c:
            print("Mock runner: SHM bridge started successfully", file=sys.stderr)

            while not _shutdown:
                msg = c2p.read()
                if msg is None:
                    break

                tokens_to_generate = (
                    msg.max_tokens if msg.max_tokens > 0 else len(msg.token_ids)
                )

                print(
                    f"Mock runner: Received task_id={msg.task_id}, "
                    f"max_tokens={msg.max_tokens}, "
                    f"num_tokens={len(msg.token_ids)}, "
                    f"tokens={msg.token_ids[:5]}{'...' if len(msg.token_ids) > 5 else ''}",
                    file=sys.stderr,
                )

                # DeepSeek R1 reasoning token IDs
                THINK_START_TOKEN = 128798  # <think>
                THINK_END_TOKEN = 128799  # </think>

                # First 21 tokens: reasoning sequence <think> ... </think> + start of answer
                reasoning_and_start_sequence = [
                    THINK_START_TOKEN,  # <think>
                    # Reasoning tokens (12 tokens)
                    2810,  # "Let"
                    502,  # " me"
                    1781,  # " think"
                    922,  # " about"
                    420,  # " this"
                    281,  # "."
                    720,  # " The"
                    3575,  # " problem"
                    7612,  # " requires"
                    8954,  # " careful"
                    6685,  # " analysis"
                    281,  # "."
                    THINK_END_TOKEN,  # </think>
                    # Start of answer tokens (7 tokens)
                    791,  # "The"
                    4320,  # " answer"
                    374,  # " is"
                    551,  # ":"
                    220,  # " "
                    2983,  # "42"
                    281,  # "."
                ]

                # After reasoning sequence, repeat answer tokens
                continuation_token = 281  # "."

                for i in range(tokens_to_generate):
                    start_time = time.perf_counter()

                    # Use reasoning sequence for first 21 tokens, then repeat continuation
                    if i < len(reasoning_and_start_sequence):
                        token_id = reasoning_and_start_sequence[i]
                    else:
                        token_id = continuation_token

                    p2c.write_token(msg.task_id, token_id)

                    print(
                        f"Mock runner: Sent token {i + 1}/{tokens_to_generate}: {token_id}",
                        file=sys.stderr,
                    )

                    # Sleep for remaining time to reach 50 microseconds total
                    elapsed = time.perf_counter() - start_time
                    remaining = 0.00002 - elapsed  # 20 microseconds total
                    if remaining > 0:
                        time.sleep(remaining)

                print(
                    f"Mock runner: Finished generating {tokens_to_generate} tokens for task {msg.task_id}",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print("\nMock runner: Interrupted by user", file=sys.stderr)

    print("Mock runner: Shutdown complete", file=sys.stderr)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    print("Mock runner: Starting (no ttnn dependencies)", file=sys.stderr)
    _run_mock_bridge()


if __name__ == "__main__":
    main()
