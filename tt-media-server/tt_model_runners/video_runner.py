#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MPI-based video generation runner for a unified 4x32 mesh across 4 machines.

All 4 ranks are launched simultaneously by tt-run. Every rank participates in
collective inference on a single unified mesh — there is no coordinator/worker
split and no TCP socket coordination.

Rank 0 owns the SHM bridge (reads VideoRequest, writes VideoResponse). All ranks
receive each request via MPI broadcast, run inference together on the shared mesh,
and rank 0 writes the result back.

Launch command (run from tt-media-server directory on the primary host):
    tt-run \
      --rank-binding <RANK_BINDING_YAML> \
      --mpi-args "--host <HOST0>,<HOST1>,<HOST2>,<HOST3> \
                  --rankfile <RANKFILE> \
                  --bind-to none --tag-output" \
      bash -c "source <METAL_ENV_SH> && \
               source <PYTHON_ENV>/bin/activate && \
               SP_MESH_4X32=true python -m tt_model_runners.video_runner"

Example:
    tt-run \
      --rank-binding /data/sadesoye/tt-metal/tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
      --mpi-args "--host bh-glx-c03u02,bh-glx-c03u08,bh-glx-c04u02,bh-glx-c04u08 \
                  --rankfile /data/cglagovich/test_c03_c04_rankfile \
                  --bind-to none --tag-output" \
      bash -c "source /data/sadesoye/tt-inference-server/tt-media-server/scripts/sp_env_sample.sh && \
               source /data/sadesoye/tt-inference-server/tt-media-server/python_env/bin/activate && \
               SP_MESH_4X32=true python -m tt_model_runners.video_runner"

Environment variables:
    MODEL_RUNNER:        DiT runner to use (e.g., tt-wan2.2)
    TT_VIDEO_SHM_INPUT:  Input SHM segment name — rank 0 only (default: tt_video_in). This must be the same used for SPPRunner.
    TT_VIDEO_SHM_OUTPUT: Output SHM segment name — rank 0 only (default: tt_video_out). This must be the same used for SPPRunner.
    SP_MESH_4X32:        Set to "true" to enable (4, 32) mesh shape override
    OMPI_COMM_WORLD_RANK: Set automatically by MPI
"""

import asyncio
import os
import pickle
import signal
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


def _attach_mpi_comm():
    """Attach mpi4py to the MPI context already initialized by tt-metal.

    Must be called AFTER set_device() — tt-metal calls MPI_Init_thread internally
    during device initialization. Setting rc.initialize=False prevents mpi4py from
    calling MPI_Init a second time, which would crash with 'MPI initialized twice'.
    Setting rc.finalize=False lets tt-metal own MPI_Finalize on shutdown.
    """
    try:
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except ImportError as e:
        raise RuntimeError(
            "mpi4py is required for multi-rank operation. "
            "Install with: pip install mpi4py"
        ) from e


def _create_dit_runner(model_runner: str, rank: int):
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


def _broadcast_request(comm, req: Optional[VideoRequest]) -> Optional[VideoRequest]:
    """Collective: rank 0 provides the request, all ranks receive it.

    Returns None when rank 0 signals shutdown (read_request returned None).
    All ranks must call this on every iteration so MPI stays in sync.
    """
    return comm.bcast(req, root=0)


def _run_inference_loop(
    comm, runner, input_shm: Optional[VideoShm], output_shm: Optional[VideoShm]
) -> None:
    rank = comm.Get_rank()

    while not _shutdown:
        raw_req: Optional[VideoRequest] = None
        if rank == 0:
            raw_req = input_shm.read_request()

        req = _broadcast_request(comm, raw_req)

        if req is None:
            print(f"Rank {rank}: Shutdown signal received, exiting loop")
            break

        if rank == 0:
            print(
                f"Rank 0: task_id={req.task_id}, "
                f"prompt='{req.prompt[:LOG_PROMPT_PREVIEW_CHARS]}...', "
                f"steps={req.num_inference_steps}, seed={req.seed}"
            )

        try:
            from domain.video_generate_request import VideoGenerateRequest

            video_gen_req = VideoGenerateRequest(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.num_inference_steps,
                seed=req.seed,
            )

            print(f"Rank {rank}: Starting inference for task {req.task_id}")
            video = runner.run([video_gen_req])
            print(f"Rank {rank}: Inference done for task {req.task_id}")

            if rank == 0:
                _write_response_to_shm(output_shm, req.task_id, video)
                print(f"Rank 0: Response written for task {req.task_id}")

        except Exception as e:
            import traceback

            print(f"Rank {rank}: ERROR for task {req.task_id}: {e}")
            traceback.print_exc()
            if rank == 0:
                _write_error_to_shm(output_shm, req.task_id, str(e))


def run_all_ranks() -> None:
    # Use env var for rank before MPI is initialized — tt-metal initializes MPI
    # during set_device(), so mpi4py cannot be initialized before that point.
    rank = _rank()

    model_runner = os.environ.get("MODEL_RUNNER", "")
    if not model_runner:
        print(f"Rank {rank}: MODEL_RUNNER environment variable not set")
        sys.exit(1)

    print(f"Rank {rank}: model={model_runner}")

    runner = _create_dit_runner(model_runner, rank)

    print(f"Rank {rank}: Setting up device...")
    runner.set_device()

    # tt-metal has now called MPI_Init_thread — safe to attach mpi4py
    comm = _attach_mpi_comm()
    print(f"Rank {comm.Get_rank()}/{comm.Get_size()}: MPI attached, loading weights...")

    runner.load_weights()

    print(f"Rank {rank}: Running warmup...")
    asyncio.run(runner.warmup())
    print(f"Rank {rank}: Model ready for inference")

    input_shm: Optional[VideoShm] = None
    output_shm: Optional[VideoShm] = None

    if rank == 0:
        input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
        output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")
        input_shm = VideoShm(input_name, mode="input", is_shutdown=_is_shutdown)
        output_shm = VideoShm(output_name, mode="output", is_shutdown=_is_shutdown)
        input_shm.open(create=True)
        output_shm.open(create=True)
        print("Rank 0: SHM bridge ready, waiting for requests...")

    conn: socket.socket | None = None
    try:
        _run_inference_loop(comm, runner, input_shm, output_shm)
    except KeyboardInterrupt:
        print(f"Rank {rank}: Interrupted by user")
    finally:
        if rank == 0:
            if input_shm:
                input_shm.close()
            if output_shm:
                output_shm.close()
        runner.close_device()

    _log.info(f"Rank {rank}: Shutdown complete")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    rank = _rank()
    print(f"Rank {rank}: Starting video runner (MPI 4x32 mode)")
    run_all_ranks()


if __name__ == "__main__":
    main()
