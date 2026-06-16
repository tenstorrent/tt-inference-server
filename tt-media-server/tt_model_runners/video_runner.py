#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MPI-based video generation runner for a unified 4x32 mesh across 4 machines.

All ranks are launched simultaneously by tt-run. Every rank participates in
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
    TT_VIDEO_SHM_INPUT:  Input SHM segment name — rank 0 only (default: tt_video_in).
    TT_VIDEO_SHM_OUTPUT: Output SHM segment name — rank 0 only (default: tt_video_out).
    SP_MESH_4X32:        Set to "true" to enable (4, 32) mesh shape override
    OMPI_COMM_WORLD_RANK: Set automatically by MPI
"""

import asyncio
import json
import os
import queue
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import (
    CANARY_DEEP_TASK_ID,
    CANARY_TASK_ID,
    CANARY_TASK_IDS,
    ModelRunners,
)
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from ipc.video_shm import (
    SP_WARMUP_TASK_ID,
    VideoRequest,
    VideoResponse,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_video_files,
)
from utils.logger import TTLogger

_log = TTLogger("video_runner")

LOG_PROMPT_PREVIEW_CHARS = 50

# Bounded encoder backlog: up to 2 jobs queued + 1 in-flight encode. Any deeper
# holds onto frame buffers (RAM) for no benefit — model latency >> encode
# latency, so the backlog stays at 0–1 in steady state. Rank 0 blocking on
# `put` if the encoder falls behind is the correct back-pressure signal.
ENCODER_QUEUE_MAXSIZE = 2
# Per-job wall-clock bound used to derive the shutdown drain timeout. Wan 2.2
# at `ultrafast` typically encodes in ~1–3s; 10s is a deliberately loose cap.
# If we hit it, something (e.g. a wedged ffmpeg child) is actually broken.
_PER_ENCODE_BOUND_S = 10.0
# Drain bound = (in-flight + queued) × per-job bound. Scales with MAXSIZE so
# future tuning can't silently underdimension the shutdown path.
ENCODER_JOIN_TIMEOUT_S = (ENCODER_QUEUE_MAXSIZE + 1) * _PER_ENCODE_BOUND_S

_shutdown = False


@dataclass
class _EncodeJob:
    """Work item handed off from rank-0 inference to the encoder thread.

    The encoder thread is the *sole* writer of ``output_shm``, for both success
    and failure paths. Exactly one of ``frames`` or ``error`` is set:

    - ``frames`` set → run ffmpeg, write SUCCESS response.
    - ``error``  set → skip ffmpeg, write ERROR response directly.

    This single-writer invariant eliminates out-of-order responses that would
    otherwise occur when an inference failure short-circuited the queue while
    a prior successful job was still waiting to encode.
    """

    task_id: str
    frames: Optional[Any] = None
    error: Optional[str] = None


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
    from tt_model_runners.dit_runners import (
        TTMochi1Runner,
        TTWan22I2VAniSoraRunner,
        TTWan22I2VDistillRunner,
        TTWan22I2VLoRARunner,
        TTWan22I2VProdiaRunner,
        TTWan22I2VRunner,
        TTWan22Runner,
    )

    runner_map = {
        ModelRunners.TT_MOCHI_1.value: TTMochi1Runner,
        ModelRunners.TT_WAN_2_2.value: TTWan22Runner,
        ModelRunners.TT_WAN_2_2_I2V.value: TTWan22I2VRunner,
        ModelRunners.TT_WAN_2_2_I2V_PRODIA.value: TTWan22I2VProdiaRunner,
        ModelRunners.TT_WAN_2_2_I2V_ANISORA.value: TTWan22I2VAniSoraRunner,
        ModelRunners.TT_WAN_2_2_I2V_DISTILL.value: TTWan22I2VDistillRunner,
        ModelRunners.TT_WAN_2_2_I2V_LORA.value: TTWan22I2VLoRARunner,
    }
    runner_class = runner_map.get(model_runner)
    if not runner_class:
        raise ValueError(
            f"Unsupported MODEL_RUNNER: {model_runner}. "
            f"Supported: {list(runner_map.keys())}"
        )
    _log.info(f"Rank {rank}: Creating {runner_class.__name__}")
    return runner_class("")


def _read_image_prompts_side_file(path: str, task_id: str) -> Optional[List[dict]]:
    """Load the I2V image_prompts list written by ``SPRunner._write_image_side_file``.

    The cross-process contract is a JSON array of ``{"image", "frame_pos"}``
    dicts (see ``ipc.video_shm.image_prompts_path`` and the SPRunner helper).
    Returns the parsed list on success, or ``None`` on any failure (missing
    file, parse error, unexpected top-level shape).
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        _log.warning(
            f"Rank 0: failed to read I2V side-file {path!r} for task {task_id}: {e}"
        )
        return None
    if not isinstance(data, list):
        _log.warning(
            f"Rank 0: I2V side-file {path!r} for task {task_id} is not a "
            f"JSON list (got {type(data).__name__})"
        )
        return None
    return data


def _enqueue_rank0_error(
    encode_queue: "queue.Queue[Optional[_EncodeJob]]",
    task_id: str,
    err: str,
) -> tuple:
    """Surface a rank-0-detected I2V error: log it, enqueue an ERROR job
    for the encoder thread to write back, and return the
    ``([], skip=True)`` tuple ``_rank0_load_image_prompts`` callers expect
    so ranks 1..N no-op in lockstep on this iteration.
    """
    _log.error(f"Rank 0: {err}")
    encode_queue.put(_EncodeJob(task_id=task_id, error=err))
    return [], True


def _rank0_load_image_prompts(
    raw_req: Optional[VideoRequest],
    encode_queue: "queue.Queue[Optional[_EncodeJob]]",
) -> tuple:
    """
    Rank-0-only: resolve ``image_prompts`` for one inference iteration.
    Returns ``(image_prompts, skip)`` where ``skip=True`` means rank 0 has
    already enqueued an error response on the caller's behalf.

    """
    if raw_req is None or not raw_req.image_path:
        return None, False
    prompts = _read_image_prompts_side_file(raw_req.image_path, raw_req.task_id)
    if prompts is None:
        return _enqueue_rank0_error(
            encode_queue,
            raw_req.task_id,
            f"I2V conditioning side-file unreadable: "
            f"{raw_req.image_path!r} for task {raw_req.task_id}",
        )
    if not prompts:
        return _enqueue_rank0_error(
            encode_queue,
            raw_req.task_id,
            f"I2V conditioning side-file is an empty list: "
            f"{raw_req.image_path!r} for task {raw_req.task_id}",
        )
    return prompts, False


def video_request_to_generate_request(
    req: VideoRequest,
    image_prompts: Optional[List[dict]] = None,
) -> VideoGenerateRequest:
    """Map SHM VideoRequest (+ optional broadcast image_prompts) to a runner request.

    Uses the intersection of field names so we never pass SHM-only fields (e.g.
    height, width, image_path) unless they exist on the target request schema.
    """
    shm_names = {f.name for f in dc_fields(VideoRequest)}
    gen_names = set(VideoGenerateRequest.model_fields.keys())
    common = shm_names & gen_names
    base_kwargs = {name: getattr(req, name) for name in common}

    if image_prompts:
        return VideoI2VGenerateRequest(
            **base_kwargs,
            image_prompts=[ImagePromptEntry(**entry) for entry in image_prompts],
        )
    return VideoGenerateRequest(**base_kwargs)


def _write_response_to_shm(
    output_shm: VideoShm,
    task_id: str,
    mp4_path: str,
) -> None:
    _log.info(f"Rank 0: Writing mp4 path to SHM: {mp4_path}")
    output_shm.write_response(
        VideoResponse(
            task_id=task_id,
            status=VideoStatus.SUCCESS,
            file_path=mp4_path,
            error_message="",
        )
    )


def _write_error_to_shm(output_shm: VideoShm, task_id: str, error: str = "") -> None:
    output_shm.write_response(
        VideoResponse(
            task_id=task_id,
            status=VideoStatus.ERROR,
            file_path="",
            error_message=error[: VideoShm.MAX_ERROR_LEN],
        )
    )


def _broadcast_request(
    comm,
    req: Optional[VideoRequest],
    image_prompts: Optional[List[dict]] = None,
    skip: bool = False,
) -> tuple:
    """Collective: rank 0 provides ``(req, image_prompts, skip)``; all ranks
    receive it.

    Returns ``(None, None, False)`` when rank 0 signals shutdown
    (``read_request`` returned None). All ranks must call this on every
    iteration so MPI stays in sync.

    ``skip=True`` signals that rank 0 has already produced an error response
    for this iteration (e.g. the I2V side-file was unreadable) and the
    inference call must be skipped on every rank in lockstep — otherwise
    ranks 1..N would block on collective ops with no rank-0 peer.
    """
    return comm.bcast((req, image_prompts, skip), root=0)


def _encoder_loop(
    output_shm: VideoShm,
    encode_queue: "queue.Queue[Optional[_EncodeJob]]",
) -> None:
    """Rank-0 background thread: sole writer of ``output_shm``.

    FIFO by construction (single consumer of the queue, single writer of the
    output SHM), so response order matches submission order for both success
    and error paths. Sentinel ``None`` signals shutdown.

    """
    from utils.video_manager import VideoManager

    video_manager = VideoManager()
    _log.info("Encoder thread: started")

    while True:
        job = encode_queue.get()
        if job is None:
            _log.info("Encoder thread: shutdown sentinel received, exiting")
            return

        # Warmup ping and canary probes (shallow + deep): respond SUCCESS with
        # empty file_path. No ffmpeg, no frames — the round-trip itself is the
        # signal. (The collective — bare barrier for shallow, replayed warmup
        # forward for deep — already ran in the inference loop; here rank 0 just
        # acks.)
        if job.task_id == SP_WARMUP_TASK_ID or job.task_id in CANARY_TASK_IDS:
            try:
                _write_response_to_shm(output_shm, job.task_id, "")
                _log.info(f"Encoder thread: replied to {job.task_id}")
            except Exception as write_err:
                _log.error(
                    f"Encoder thread: failed to write {job.task_id} response: "
                    f"{write_err}"
                )
            continue

        if job.error is not None:
            try:
                _write_error_to_shm(output_shm, job.task_id, job.error)
            except Exception as write_err:
                _log.error(
                    f"Encoder thread: failed to write upstream-error response "
                    f"for task {job.task_id}: {write_err}"
                )
            continue

        try:
            mp4_path = video_manager.export_to_mp4(job.frames)
            _log.info(
                f"Encoder thread: encoded mp4 for task {job.task_id} at {mp4_path}"
            )
            _write_response_to_shm(output_shm, job.task_id, mp4_path)
        except Exception as encode_err:
            _log.error(
                f"Encoder thread: encode failure for task {job.task_id}: "
                f"{encode_err}\n{traceback.format_exc()}"
            )
            try:
                _write_error_to_shm(output_shm, job.task_id, str(encode_err))
            except Exception as write_err:
                _log.error(
                    f"Encoder thread: failed to write error response for task "
                    f"{job.task_id}: {write_err}"
                )


def _run_inference_loop(
    comm,
    runner,
    input_shm: Optional[VideoShm],
    encode_queue: "Optional[queue.Queue[Optional[_EncodeJob]]]",
) -> None:
    """Collective inference loop. Rank 0 owns SHM + encoder queue; other ranks
    receive broadcasts and run inference only.

    Rank 0 never writes to ``output_shm`` directly — all responses (success
    and error) are handed off to the encoder thread so ordering matches
    submission (see ``_encoder_loop`` for the single-writer invariant).
    """
    rank = comm.Get_rank()

    while not _shutdown:
        raw_req: Optional[VideoRequest] = None
        rank0_image_prompts: Optional[List[dict]] = None
        rank0_skip = False
        if rank == 0:
            raw_req = input_shm.read_request()
            # Server-side readiness ping. Reuses the existing ``skip`` lockstep
            # path so ranks 1..N don't block on a collective with no peer:
            # rank 0 hands a SUCCESS response to the encoder thread directly
            # (single-writer invariant preserved) and we broadcast skip=True
            # so every rank no-ops this iteration. No inference, no MPI
            # broadcast of the dummy request body — round-trip latency equals
            # the pipeline's own cold-start time, which is exactly the signal
            # SPRunner wants.
            if raw_req is not None and raw_req.task_id == SP_WARMUP_TASK_ID:
                _log.info("Rank 0: received SP warmup ping, replying READY")
                encode_queue.put(
                    _EncodeJob(task_id=SP_WARMUP_TASK_ID, frames=None, error=None)
                )
                rank0_skip = True
            elif raw_req is not None and raw_req.task_id in CANARY_TASK_IDS:
                # Canary probe (shallow or deep): unlike warmup, do NOT skip. We
                # broadcast the request (skip=False) so every rank joins the
                # collective below in lockstep — that is what lets the probe
                # catch a wedged sub-rank. No image-prompt loading for canaries.
                _log.info(
                    f"Rank 0: received canary probe ({raw_req.task_id}), "
                    f"broadcasting to all ranks"
                )
            else:
                rank0_image_prompts, rank0_skip = _rank0_load_image_prompts(
                    raw_req, encode_queue
                )

        req, image_prompts, skip = _broadcast_request(
            comm,
            raw_req,
            image_prompts=rank0_image_prompts,
            skip=rank0_skip,
        )

        if req is None:
            _log.info(f"Rank {rank}: Shutdown signal received, exiting loop")
            break

        if skip:
            if rank == 0:
                _log.info(
                    f"Rank 0: skipping inference for task {req.task_id} "
                    f"(rank 0 already submitted response)"
                )
            continue

        if req.task_id == CANARY_TASK_ID:
            # Canary probe: every rank joins one bare MPI collective to prove
            # the whole job's host loops are responsive, then rank 0 acks. The
            # broadcast above (skip=False) guarantees all ranks reach this
            # barrier in lockstep, so a wedged sub-rank stalls the collective
            # and fails the probe — which the warmup short-circuit cannot do.
            #
            # INTENTIONAL TRADE-OFF: a bare collective proves host-side rank liveness, NOT
            # device responsiveness.
            comm.barrier()
            if rank == 0:
                encode_queue.put(
                    _EncodeJob(task_id=CANARY_TASK_ID, frames=None, error=None)
                )
            continue

        if req.task_id == CANARY_DEEP_TASK_ID:
            # Deep canary: replay the runner's compiled warmup forward on EVERY
            # rank (a real collective forward), so the ack proves the device can
            # still compute — not just that hosts reach a barrier. We reuse the
            # exact warmup request so the shape is already compiled: a novel
            # shape would trigger a recompile and evict real-request programs
            # from the cache, degrading the very thing we monitor. Frames are
            # discarded; rank 0 just acks (no ffmpeg). An exception on any rank
            # becomes an ERROR ack → a probe miss, which is the correct signal.
            try:
                runner.run([runner._build_warmup_video_request()])
                if rank == 0:
                    encode_queue.put(
                        _EncodeJob(task_id=CANARY_DEEP_TASK_ID, frames=None, error=None)
                    )
            except Exception as e:
                _log.error(
                    f"Rank {rank}: deep canary forward failed: {e}\n"
                    f"{traceback.format_exc()}"
                )
                if rank == 0:
                    encode_queue.put(
                        _EncodeJob(task_id=CANARY_DEEP_TASK_ID, error=str(e))
                    )
            continue

        if rank == 0:
            _log.info(
                f"Rank 0: task_id={req.task_id}, "
                f"prompt='{req.prompt[:LOG_PROMPT_PREVIEW_CHARS]}...', "
                f"steps={req.num_inference_steps}, seed={req.seed}, "
                f"image_prompts={len(image_prompts) if image_prompts else 0}"
            )

        try:
            video_gen_req = video_request_to_generate_request(req, image_prompts)

            _log.info(f"Rank {rank}: Starting inference for task {req.task_id}")
            frames = runner.run([video_gen_req])
            _log.info(f"Rank {rank}: Inference done for task {req.task_id}")

            if rank == 0:
                # Hand the frames off to the encoder thread and immediately
                # loop back to read_request — this frees the mesh while
                # ffmpeg runs in parallel. `put` is bounded; if the encoder
                # falls behind we block here, naturally back-pressuring
                # inference (no memory blowup).
                encode_queue.put(_EncodeJob(task_id=req.task_id, frames=frames))

        except Exception as e:
            _log.error(
                f"Rank {rank}: ERROR for task {req.task_id}: {e}\n"
                f"{traceback.format_exc()}"
            )
            if rank == 0:
                encode_queue.put(_EncodeJob(task_id=req.task_id, error=str(e)))


def run_all_ranks() -> None:
    rank = _rank()

    model_runner = os.environ.get("MODEL_RUNNER", "")
    if not model_runner:
        _log.error(f"Rank {rank}: MODEL_RUNNER environment variable not set")
        sys.exit(1)

    _log.info(f"Rank {rank}: model={model_runner}")

    runner = _create_dit_runner(model_runner, rank)
    # Force out-of-runner MP4 export across all ranks: the encoder thread on
    # rank 0 is the sole writer to ``output_shm`` and must receive raw frames,
    # not a path-string. ``hasattr`` keeps this a no-op for runners that don't
    # opt into the in-worker export pattern.
    if hasattr(runner, "export_in_runner"):
        runner.export_in_runner = False

    _log.info(f"Rank {rank}: Setting up device...")
    runner.set_device()

    comm = _attach_mpi_comm()
    _log.info(
        f"Rank {comm.Get_rank()}/{comm.Get_size()}: MPI attached, loading weights..."
    )

    runner.load_weights()

    _log.info(f"Rank {rank}: Running warmup...")
    asyncio.run(runner.warmup())
    _log.info(f"Rank {rank}: Model ready for inference")

    input_shm: Optional[VideoShm] = None
    output_shm: Optional[VideoShm] = None
    encode_queue: Optional[queue.Queue] = None
    encoder_thread: Optional[threading.Thread] = None

    if rank == 0:
        input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
        output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")
        input_shm = VideoShm(input_name, mode="input", is_shutdown=_is_shutdown)
        output_shm = VideoShm(output_name, mode="output", is_shutdown=_is_shutdown)
        # Create-or-attach; whichever side comes up first owns creation. Ring
        # position survives runner restarts via the <name>_state segment.
        input_shm.open()
        output_shm.open()
        # Self-heal any gap left by a previous runner instance that crashed
        # mid-read (on input) or mid-write (on output). Scoped to this
        # process's own role, so safe to run with a live server peer.
        in_repair = input_shm.recover(side="reader")
        out_repair = output_shm.recover(side="writer")
        if any(in_repair.values()) or any(out_repair.values()):
            _log.warning(
                f"Rank 0: crash-recovery repaired prior inconsistency: "
                f"input={in_repair} output={out_repair}"
            )

        # Start the encoder thread AFTER output_shm is open and recovered,
        # so its first `_write_response_to_shm` call sees a usable segment.
        encode_queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
        encoder_thread = threading.Thread(
            target=_encoder_loop,
            args=(output_shm, encode_queue),
            name="video-encoder",
            daemon=False,
        )
        encoder_thread.start()
        _log.info("Rank 0: SHM bridge ready, waiting for requests...")

    try:
        _run_inference_loop(comm, runner, input_shm, encode_queue)
    except KeyboardInterrupt:
        _log.info(f"Rank {rank}: Interrupted by user")
    finally:
        if rank == 0:
            # Order matters: stop the encoder BEFORE closing output_shm so any
            # remaining encodes can write their responses on the way out.
            if encode_queue is not None and encoder_thread is not None:
                encode_queue.put(None)
                encoder_thread.join(timeout=ENCODER_JOIN_TIMEOUT_S)
                if encoder_thread.is_alive():
                    _log.warning(
                        f"Rank 0: encoder thread did not drain within "
                        f"{ENCODER_JOIN_TIMEOUT_S}s "
                        f"(remaining queued={encode_queue.qsize()}); "
                        f"continuing shutdown"
                    )
            if input_shm:
                input_shm.close()
            if output_shm:
                output_shm.close()
            # Mirror mock_video_runner_base + sp_runner: a crash mid-encode
            # can leave mp4 files on tmpfs. Sweeping here bounds orphan growth
            # to one runner lifetime instead of relying on the next SP_RUNNER
            # restart to clean up.
            removed = cleanup_orphaned_video_files()
            if removed:
                _log.info(f"Rank 0: cleaned up {removed} orphaned video file(s)")
        runner.close_device()

    _log.info(f"Rank {rank}: Shutdown complete")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    rank = _rank()
    _log.info(f"Rank {rank}: Starting video runner (MPI 4x32 mode)")
    run_all_ranks()


if __name__ == "__main__":
    main()
