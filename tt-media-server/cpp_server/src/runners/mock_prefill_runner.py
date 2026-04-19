#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Mock pipeline runner with multi-rank support for testing prefill.

This runner uses shared memory for the coordinator (rank 0) and sockets
for distributing work to 3 worker processes (ranks 1-3). Returns exactly
one token per request (prefill only).

Environment variables required:
    TT_IPC_SHM_C2P: Name of input  SHM segment (e.g., tt_ipc_c2p_12345)
    TT_IPC_SHM_P2C: Name of output SHM segment (e.g., tt_ipc_p2c_12345)
    OMPI_COMM_WORLD_RANK or RANK: Rank ID (0-3)

Usage:
    # Rank 0 (coordinator):
    export RANK=0
    export TT_IPC_SHM_C2P=tt_ipc_c2p_12345
    export TT_IPC_SHM_P2C=tt_ipc_p2c_12345
    python mock_pipeline_runner.py

    # Ranks 1-3 (workers):
    export RANK=1  # or 2, 3
    python mock_pipeline_runner.py
"""

import os
import pickle
import signal
import socket
import struct
import sys
import time
from typing import Optional

from shared_memory import PREFILL_MAX_TOKEN_IDS, PrefillMessage, SharedMemory

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


def _rank() -> int:
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


RANK_CONFIG = {
    0: {"ip": "127.0.0.1", "port": 10000},
    1: {"ip": "127.0.0.1", "port": 10001},
    2: {"ip": "127.0.0.1", "port": 10002},
    3: {"ip": "127.0.0.1", "port": 10003},
}


def _send_via_socket(sock: socket.socket, msg: PrefillMessage) -> None:
    """Send PrefillMessage via socket using pickle."""
    data = pickle.dumps(msg)
    sock.sendall(struct.pack("<I", len(data)))
    sock.sendall(data)


def _recv_via_socket(conn: socket.socket) -> Optional[PrefillMessage]:
    """Receive PrefillMessage via socket using pickle."""
    try:
        length_data = conn.recv(4)
        if not length_data:
            return None
        length = struct.unpack("<I", length_data)[0]

        data = b""
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), 4096))
            if not chunk:
                return None
            data += chunk

        return pickle.loads(data)
    except Exception as e:
        print(f"Error receiving request: {e}", file=sys.stderr)
        return None


def _connect_to_workers() -> dict[int, socket.socket]:
    """Connect to worker ranks 1-3, returning sockets for those that accepted."""
    worker_sockets: dict[int, socket.socket] = {}
    for rank in [1, 2, 3]:
        config = RANK_CONFIG[rank]
        print(
            f"Rank 0: Connecting to rank {rank} at {config['ip']}:{config['port']}",
            file=sys.stderr,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((config["ip"], config["port"]))
            worker_sockets[rank] = sock
            print(f"Rank 0: Connected to rank {rank}", file=sys.stderr)
        except Exception as e:
            print(f"Rank 0: Failed to connect to rank {rank}: {e}", file=sys.stderr)
            sock.close()

    if worker_sockets:
        print(f"Rank 0: Connected to {len(worker_sockets)} workers", file=sys.stderr)
    else:
        print(
            "Rank 0: No workers connected, running in standalone mode", file=sys.stderr
        )
    return worker_sockets


def run_rank0_coordinator() -> None:
    """Rank 0: Read from input SHM, send to workers, return one token to output SHM."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")

    if not (c2p_name and p2c_name):
        print(
            "Error: TT_IPC_SHM_C2P or TT_IPC_SHM_P2C not set",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Rank 0: Opening shared memory C2P={c2p_name}, P2C={p2c_name}",
        file=sys.stderr,
    )

    try:
        with SharedMemory(
            c2p_name, max_token_ids=PREFILL_MAX_TOKEN_IDS, is_shutdown=_is_shutdown
        ) as c2p, SharedMemory(
            p2c_name,
            max_token_ids=1,
            is_shutdown=_is_shutdown,  # Only 1 token output
        ) as p2c:
            print("Rank 0: SHM bridge started successfully", file=sys.stderr)

            print("Rank 0: Waiting 2s for workers to start...", file=sys.stderr)
            time.sleep(2)
            worker_sockets = _connect_to_workers()

            print("Rank 0: Waiting for prefill requests...", file=sys.stderr)

            while not _shutdown:
                msg = c2p.read()
                if msg is None:
                    break

                task_id = msg.task_id  # Now an int (uint32_t)

                print(
                    f"Rank 0: Received prefill request task_id={task_id}, "
                    f"num_tokens={len(msg.token_ids)}, "
                    f"tokens={msg.token_ids[:5]}{'...' if len(msg.token_ids) > 5 else ''}",
                    file=sys.stderr,
                )

                # Send to all workers
                for rank, sock in worker_sockets.items():
                    try:
                        _send_via_socket(sock, msg)
                        print(f"Rank 0: Sent request to worker {rank}", file=sys.stderr)
                    except Exception as e:
                        print(
                            f"Rank 0: Failed to send to rank {rank}: {e}",
                            file=sys.stderr,
                        )

                # Return exactly one token (first token from reasoning sequence)
                prefill_token = 128798  # <think> token

                p2c.write_token(task_id, prefill_token)

                print(
                    f"Rank 0: Sent prefill token {prefill_token} for task {task_id}",
                    file=sys.stderr,
                )

    except KeyboardInterrupt:
        print("Rank 0: Interrupted by user", file=sys.stderr)
    finally:
        for rank, sock in worker_sockets.items():
            sock.close()

    print("Rank 0: Shutdown complete", file=sys.stderr)


def run_worker_rank(rank: int) -> None:
    """Ranks 1-3: Listen on socket, receive requests from rank 0, and print them."""
    config = RANK_CONFIG[rank]

    print(
        f"Rank {rank}: Starting worker on {config['ip']}:{config['port']}",
        file=sys.stderr,
    )
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((config["ip"], config["port"]))
    server_sock.listen(1)

    print(f"Rank {rank}: Listening for connections...", file=sys.stderr)

    try:
        conn, addr = server_sock.accept()
        print(f"Rank {rank}: Accepted connection from {addr}", file=sys.stderr)

        while not _shutdown:
            msg = _recv_via_socket(conn)
            if msg is None:
                break

            task_id_str = msg.task_id.decode("utf-8", errors="ignore").rstrip("\x00")

            print(
                f"Rank {rank}: Received prefill request task_id={task_id_str}, "
                f"num_tokens={len(msg.token_ids)}, "
                f"tokens={msg.token_ids[:10]}{'...' if len(msg.token_ids) > 10 else ''}",
                file=sys.stderr,
            )

    except KeyboardInterrupt:
        print(f"Rank {rank}: Interrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"Rank {rank}: Error: {e}", file=sys.stderr)
    finally:
        server_sock.close()

    print(f"Rank {rank}: Shutdown complete", file=sys.stderr)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    rank = _rank()
    print(f"Starting mock pipeline runner with rank={rank}", file=sys.stderr)

    if rank == 0:
        run_rank0_coordinator()
    elif rank in [1, 2, 3]:
        run_worker_rank(rank)
    else:
        print(f"Invalid rank: {rank}. Must be 0-3", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
