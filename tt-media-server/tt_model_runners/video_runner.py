#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared memory video generation runner with multi-rank support.

Reads VideoRequest from input VideoShm, processes with a DiT runner
(Mochi, Wan, etc.), and streams generated frames back through output VideoShm.

This is the external runner process that pairs with SPRunner on the server side.
SPRunner runs inside the standard device_worker and communicates via VideoShm.

Multi-rank mode:
- Rank 0: Reads from SHM, runs DiT inference, writes frames to output SHM, distributes to ranks 1-3
- Ranks 1-3: Listen on sockets, receive requests from rank 0, and run DiT inference on their devices

Environment variables required:
    TT_VIDEO_SHM_INPUT:  Name of input  SHM segment (default ``tt_video_in``)
    TT_VIDEO_SHM_OUTPUT: Name of output SHM segment (default ``tt_video_out``)
    TT_VISIBLE_DEVICES:  Device ID(s) to use
    MODEL_RUNNER:        DiT runner to use (e.g., tt_mochi_1, tt_wan_2_2)
    OMPI_COMM_WORLD_RANK or RANK: Rank ID (0-3)

Usage:
    # Rank 0 (coordinator with inference):
    export RANK=0
    export TT_VIDEO_SHM_INPUT=tt_video_in
    export TT_VIDEO_SHM_OUTPUT=tt_video_out
    export TT_VISIBLE_DEVICES=0
    export MODEL_RUNNER=tt_mochi_1
    python -m tt_model_runners.video_runner

    # Ranks 1-3 (workers with inference):
    export RANK=1  # or 2, 3
    export TT_VISIBLE_DEVICES=1  # or 2, 3
    export MODEL_RUNNER=tt_mochi_1
    python -m tt_model_runners.video_runner
"""

import asyncio
import os
import pickle
import signal
import socket
import struct
import sys
import time
import traceback
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import ModelRunners
from domain.video_generate_request import VideoGenerateRequest
from ipc.video_shm import (
    VideoRequest,
    VideoResponse,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_video_files,
)
from utils.logger import TTLogger

if TYPE_CHECKING:
    from tt_model_runners.dit_runners import TTMochi1Runner, TTWan22Runner

    DitRunner = TTMochi1Runner | TTWan22Runner

_log = TTLogger("video_runner")

# Defaults (overridden by TT_VIDEO_SHM_INPUT / TT_VIDEO_SHM_OUTPUT when set)
DEFAULT_TT_VIDEO_SHM_INPUT = "tt_video_in"
DEFAULT_TT_VIDEO_SHM_OUTPUT = "tt_video_out"

# Rank 0 waits for worker ranks to bind listening sockets before connecting.
WORKER_STARTUP_WAIT_SECONDS = 5

# Inter-rank socket bridge (rank 0 ↔ ranks 1–3) on loopback.
WORKER_LOOPBACK_HOST = "127.0.0.1"
RANK_SOCKET_BASE_PORT = 9000

RANK_CONFIG = {
    rank: {"ip": WORKER_LOOPBACK_HOST, "port": RANK_SOCKET_BASE_PORT + rank}
    for rank in range(4)
}

COORDINATOR_RANK = 0
WORKER_RANKS = (1, 2, 3)
MAX_RANK = 3

# Rank 0 passes no device id into the DiT runner ctor (coordinator); workers use TT_VISIBLE_DEVICES.
COORDINATOR_RUNNER_DEVICE_ID = ""

SOCKET_RECV_CHUNK_BYTES = 4096
LOG_PROMPT_PREVIEW_CHARS = 50


def video_request_to_generate_request(req: VideoRequest) -> VideoGenerateRequest:
    """Map SHM :class:`VideoRequest` to :class:`VideoGenerateRequest` for DiT runners.

    Uses the intersection of field names so we never pass SHM-only fields (e.g.
    ``height``, ``width``) unless they exist on ``VideoGenerateRequest``. Today
    that matches the historical explicit mapping: prompt, negative_prompt,
    ``num_inference_steps``, seed — same as ``SPRunner`` / API usage.
    """
    shm_names = {f.name for f in dc_fields(VideoRequest)}
    gen_names = set(VideoGenerateRequest.model_fields.keys())
    common = shm_names & gen_names
    return VideoGenerateRequest(**{name: getattr(req, name) for name in common})


_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


def _rank() -> int:
    r = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("RANK")
    return int(r) if r is not None else 0


def _send_via_socket(sock: socket.socket, request: VideoRequest) -> None:
    """Send VideoRequest via socket using pickle."""
    data = pickle.dumps(request)
    sock.sendall(struct.pack("<I", len(data)))
    sock.sendall(data)


def _recv_via_socket(conn: socket.socket) -> Optional[VideoRequest]:
    """Receive VideoRequest via socket using pickle."""
    try:
        length_data = conn.recv(4)
        if not length_data:
            return None
        length = struct.unpack("<I", length_data)[0]

        data = b""
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), SOCKET_RECV_CHUNK_BYTES))
            if not chunk:
                return None
            data += chunk

        return pickle.loads(data)
    except Exception as e:
        _log.warning(f"Error receiving request: {e}")
        return None


def create_dit_runner(model_runner: str, device_id: str):
    """Create the appropriate DiT runner (lazy import to avoid loading ttnn globally)."""
    from tt_model_runners.dit_runners import TTMochi1Runner, TTWan22Runner

    runner_map = {
        ModelRunners.TT_MOCHI_1.value: TTMochi1Runner,
        ModelRunners.TT_WAN_2_2.value: TTWan22Runner,
    }

    runner_class = runner_map.get(model_runner)
    if not runner_class:
        raise ValueError(
            f"Unsupported MODEL_RUNNER: {model_runner}. "
            f"Supported: {list(runner_map.keys())}"
        )

    _log.info(f"Creating {runner_class.__name__} for device {device_id}")
    return runner_class(device_id)


def _bootstrap_dit_runner(rank: int, runner_device_id: str) -> "DitRunner":
    """Create DiT runner from ``MODEL_RUNNER``, set device, load weights, warmup.

    ``runner_device_id`` is the constructor argument: use
    :data:`COORDINATOR_RUNNER_DEVICE_ID` for rank 0 coordinator; workers pass
    ``TT_VISIBLE_DEVICES``.
    """
    model_runner = os.environ.get("MODEL_RUNNER", "")
    if not model_runner:
        _log.error(f"Rank {rank}: MODEL_RUNNER environment variable not set")
        sys.exit(1)

    visible = os.environ.get("TT_VISIBLE_DEVICES", "")
    _log.info(f"Rank {rank}: model={model_runner}, device={visible}")

    runner = create_dit_runner(model_runner, runner_device_id)
    _log.info(f"Rank {rank}: Setting up device...")
    runner.set_device()
    _log.info(f"Rank {rank}: Loading weights...")
    runner.load_weights()
    _log.info(f"Rank {rank}: Running warmup...")
    asyncio.run(runner.warmup())
    _log.info(f"Rank {rank}: Model ready for inference")
    return runner


def _write_response_to_shm(
    output_shm: VideoShm,
    task_id: str,
    mp4_path: str,
) -> None:
    """Send the final mp4 path through SHM (rank 0 encodes before calling this)."""
    _log.info(f"Rank 0: Writing mp4 path to SHM: {mp4_path}")
    output_shm.write_response(
        VideoResponse(
            task_id=task_id,
            status=VideoStatus.SUCCESS,
            file_path=mp4_path,
            error_message="",
        )
    )
    _log.info("Rank 0: Response written to SHM successfully")


def _write_error_to_shm(output_shm: VideoShm, task_id: str, error: str = "") -> None:
    output_shm.write_response(
        VideoResponse(
            task_id=task_id,
            status=VideoStatus.ERROR,
            file_path="",
            error_message=error[: VideoShm.MAX_ERROR_LEN],
        )
    )


def _connect_to_workers() -> dict[int, socket.socket]:
    """Connect to worker ranks 1-3, returning sockets for those that accepted."""
    worker_sockets: dict[int, socket.socket] = {}
    for rank in WORKER_RANKS:
        config = RANK_CONFIG[rank]
        _log.info(
            f"Rank 0: Connecting to rank {rank} at {config['ip']}:{config['port']}"
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((config["ip"], config["port"]))
            worker_sockets[rank] = sock
            _log.info(f"Rank 0: Connected to rank {rank}")
        except Exception as e:
            _log.warning(f"Rank 0: Failed to connect to rank {rank}: {e}")
            sock.close()

    if worker_sockets:
        _log.info(f"Rank 0: Connected to {len(worker_sockets)} workers")
    else:
        _log.info("Rank 0: No workers connected, running in standalone mode")
    return worker_sockets


def run_rank0_coordinator() -> None:
    """Rank 0: Read from input SHM, run DiT inference, write MP4 path to output SHM."""
    input_name = os.environ.get("TT_VIDEO_SHM_INPUT", DEFAULT_TT_VIDEO_SHM_INPUT)
    output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", DEFAULT_TT_VIDEO_SHM_OUTPUT)

    runner = _bootstrap_dit_runner(
        rank=COORDINATOR_RANK, runner_device_id=COORDINATOR_RUNNER_DEVICE_ID
    )
    _log.info(f"Rank 0: SHM in={input_name}, out={output_name}")

    input_shm = VideoShm(input_name, mode="input", is_shutdown=_is_shutdown)
    output_shm = VideoShm(output_name, mode="output", is_shutdown=_is_shutdown)
    input_shm.open(create=True)
    output_shm.open(create=True)

    _log.info(f"Rank 0: Waiting {WORKER_STARTUP_WAIT_SECONDS}s for workers to start...")
    time.sleep(WORKER_STARTUP_WAIT_SECONDS)
    worker_sockets = _connect_to_workers()

    _log.info("Rank 0: SHM bridge started, waiting for requests...")

    try:
        while not _shutdown:
            req = input_shm.read_request()
            if req is None:
                break

            _log.info(
                f"Rank 0: task_id={req.task_id}, "
                f"prompt='{req.prompt[:LOG_PROMPT_PREVIEW_CHARS]}...', "
                f"steps={req.num_inference_steps}, seed={req.seed}"
            )

            for rank, sock in worker_sockets.items():
                try:
                    _send_via_socket(sock, req)
                except Exception as e:
                    _log.warning(f"Rank 0: Failed to send to rank {rank}: {e}")

            try:
                video_gen_req = video_request_to_generate_request(req)

                _log.info(f"Rank 0: Starting inference for task {req.task_id}")
                frames = runner.run([video_gen_req])
                _log.info("Rank 0: Inference done")

                from utils.video_manager import VideoManager

                mp4_path = VideoManager().export_to_mp4(frames)
                _log.info(f"Rank 0: Encoded mp4 at {mp4_path}")

                _write_response_to_shm(output_shm, req.task_id, mp4_path)
                _log.info(f"Rank 0: Response written for task {req.task_id}")

            except Exception as e:
                _log.error(
                    f"Rank 0: ERROR - Inference/write failed for task {req.task_id}: {e}\n"
                    f"{traceback.format_exc()}"
                )
                _write_error_to_shm(output_shm, req.task_id, str(e))

    except KeyboardInterrupt:
        _log.info("Rank 0: Interrupted by user")
    finally:
        for rank, sock in worker_sockets.items():
            sock.close()
        input_shm.close()
        output_shm.close()
        runner.close_device()
        removed = cleanup_orphaned_video_files()
        if removed:
            _log.info(f"Rank 0: Cleaned up {removed} orphaned video file(s)")

    _log.info("Rank 0: Shutdown complete")


def run_worker_rank(rank: int) -> None:
    """Ranks 1-3: Listen on socket, receive requests from rank 0, and run DiT inference."""
    config = RANK_CONFIG[rank]
    device_id = os.environ.get("TT_VISIBLE_DEVICES", "")

    runner = _bootstrap_dit_runner(rank=rank, runner_device_id=device_id)

    _log.info(f"Rank {rank}: Starting worker on {config['ip']}:{config['port']}")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((config["ip"], config["port"]))
    server_sock.listen(1)

    _log.info(f"Rank {rank}: Listening for connections...")

    conn: socket.socket | None = None
    try:
        conn, addr = server_sock.accept()
        _log.info(f"Rank {rank}: Accepted connection from {addr}")

        while not _shutdown:
            req = _recv_via_socket(conn)
            if req is None:
                break

            _log.info(
                f"Rank {rank}: task_id={req.task_id}, "
                f"prompt='{req.prompt[:LOG_PROMPT_PREVIEW_CHARS]}...', "
                f"steps={req.num_inference_steps}, seed={req.seed}"
            )

            try:
                video_gen_req = video_request_to_generate_request(req)

                _log.info(f"Rank {rank}: Starting inference for task {req.task_id}")
                _frames = runner.run([video_gen_req])
                _log.info(f"Rank {rank}: Inference done for task {req.task_id}")
            except Exception as e:
                _log.error(f"Rank {rank}: Inference failed for task {req.task_id}: {e}")

    except KeyboardInterrupt:
        _log.info(f"Rank {rank}: Interrupted by user")
    except Exception as e:
        _log.error(f"Rank {rank}: Error: {e}")
    finally:
        if conn is not None:
            conn.close()
        runner.close_device()
        server_sock.close()

    _log.info(f"Rank {rank}: Shutdown complete")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    rank = _rank()
    _log.info(f"Starting video runner with rank={rank}")

    if rank == COORDINATOR_RANK:
        run_rank0_coordinator()
    elif rank in WORKER_RANKS:
        run_worker_rank(rank)
    else:
        _log.error(f"Invalid rank: {rank}. Must be 0-{MAX_RANK}")
        sys.exit(1)


if __name__ == "__main__":
    main()
