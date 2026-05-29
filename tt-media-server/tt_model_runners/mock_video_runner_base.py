# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared scaffolding for the concurrent SHM mock runners.

Two thin variants build on this base:

* ``mock_video_runner_concurrent.py``     — T2V (no per-request validation)
* ``mock_video_runner_concurrent_i2v.py`` — I2V (validates the side-file
  contract end-to-end before the simulated inference sleep)

Both inherit the same producer/consumer split that ``video_runner.py``
uses post-Phase-1, so the encode cost is hidden behind the next request's
inference (in steady state) instead of stacking on top:

  * **reader thread** drains the input SHM ring as fast as requests arrive.
  * **inference pool** (``MOCK_CONCURRENCY`` workers) runs the optional
    per-request validator, then ``time.sleep(MOCK_LATENCY_S)`` to fake
    inference, then hands the (task_id, frames) tuple to the encoder
    queue and IMMEDIATELY frees the worker.
  * **encoder thread** (single consumer, bounded FIFO queue) sleeps
    ``MOCK_ENCODE_S`` to fake mp4 encode, runs the real ``export_to_mp4``
    placeholder, then writes the response to the output SHM ring.

Env knobs (shared by both variants)::

    TT_VIDEO_SHM_INPUT       (default tt_video_in)
    TT_VIDEO_SHM_OUTPUT      (default tt_video_out)
    MOCK_CONCURRENCY         (default 8)   max concurrent in-flight requests
    MOCK_LATENCY_S           (default 2.0) simulated inference time per request
    MOCK_ENCODE_S            (default 1.0) simulated mp4 encode time per request
    MOCK_FRAME_HEIGHT        (default 8)   placeholder frame height
    MOCK_FRAME_WIDTH         (default 8)   placeholder frame width
    MOCK_NUM_FRAMES          (default 4)   placeholder frame count
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

_DEFAULT_CONCURRENCY = 8
_DEFAULT_LATENCY_S = 2.0
_DEFAULT_ENCODE_S = 1.0
_DEFAULT_HEIGHT = 8
_DEFAULT_WIDTH = 8
_DEFAULT_NUM_FRAMES = 4

ENCODER_QUEUE_MAXSIZE = 2
ENCODER_JOIN_TIMEOUT_S = 30.0

_shutdown = False


@dataclass
class _EncodeJob:
    taskId: str
    frames: Any
    enqueuedAt: float


@dataclass
class _BridgeConfig:
    label: str
    inputName: str
    outputName: str
    concurrency: int
    latencyS: float
    encodeS: float
    height: int
    width: int
    numFrames: int
    perRequestValidator: Optional[Callable[[Any], None]]


def handleSignal(signum, frame):
    """SIGTERM/SIGINT handler — flips the module-level shutdown flag."""
    global _shutdown
    _shutdown = True


def _readEnvInt(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _readEnvFloat(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _isShutdown() -> bool:
    return _shutdown


def _readBridgeConfig(
    label: str,
    perRequestValidator: Optional[Callable[[Any], None]],
) -> _BridgeConfig:
    return _BridgeConfig(
        label=label,
        inputName=os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in"),
        outputName=os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out"),
        concurrency=_readEnvInt("MOCK_CONCURRENCY", _DEFAULT_CONCURRENCY),
        latencyS=_readEnvFloat("MOCK_LATENCY_S", _DEFAULT_LATENCY_S),
        encodeS=_readEnvFloat("MOCK_ENCODE_S", _DEFAULT_ENCODE_S),
        height=_readEnvInt("MOCK_FRAME_HEIGHT", _DEFAULT_HEIGHT),
        width=_readEnvInt("MOCK_FRAME_WIDTH", _DEFAULT_WIDTH),
        numFrames=_readEnvInt("MOCK_NUM_FRAMES", _DEFAULT_NUM_FRAMES),
        perRequestValidator=perRequestValidator,
    )


def runMockBridge(
    label: str,
    perRequestValidator: Optional[Callable[[Any], None]] = None,
) -> None:
    """Run the SHM bridge mock loop until SIGTERM/SIGINT.

    The optional ``perRequestValidator`` callable runs INSIDE each worker
    BEFORE the simulated inference sleep. Any exception raised is caught
    by the worker's error path and surfaces as an ERROR response on SHM
    output — same shape that the real runner peer would produce on a
    contract violation. The T2V mock passes ``None`` (no validation);
    the I2V mock passes a validator that opens + parses the side-file.
    """
    from config.constants import GAS_PROBE_TASK_ID
    from ipc.video_shm import (
        SP_WARMUP_TASK_ID,
        VideoResponse,
        VideoShm,
        VideoStatus,
        cleanup_orphaned_video_files,
    )
    from utils.logger import TTLogger
    from utils.video_manager import VideoManager

    logger = TTLogger()
    cfg = _readBridgeConfig(label, perRequestValidator)

    inputShm = VideoShm(cfg.inputName, mode="input", is_shutdown=_isShutdown)
    outputShm = VideoShm(cfg.outputName, mode="output", is_shutdown=_isShutdown)
    inputShm.open()
    outputShm.open()

    inRepair = inputShm.recover(side="reader")
    outRepair = outputShm.recover(side="writer")
    if any(inRepair.values()) or any(outRepair.values()):
        logger.warning(
            f"{cfg.label} crash-recovery repaired prior inconsistency: "
            f"input={inRepair} output={outRepair}"
        )

    placeholderFrames = np.random.randint(
        0, 256, (cfg.numFrames, cfg.height, cfg.width, 3), dtype=np.uint8
    )

    logger.info(
        f"{cfg.label} ready  concurrency={cfg.concurrency}  "
        f"latency={cfg.latencyS}s  encode={cfg.encodeS}s  "
        f"frames={cfg.numFrames}x{cfg.height}x{cfg.width}  "
        f"input={cfg.inputName}  output={cfg.outputName}  "
        f"validator={'yes' if cfg.perRequestValidator else 'none'}"
    )

    writeLock = threading.Lock()
    pool = ThreadPoolExecutor(
        max_workers=cfg.concurrency, thread_name_prefix="mock-infer"
    )
    encodeQueue: queue.Queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
    inFlight = {"v": 0}
    inFlightLock = threading.Lock()

    def adjustInFlight(delta: int) -> int:
        with inFlightLock:
            inFlight["v"] += delta
            return inFlight["v"]

    def writeError(taskId: str, err: BaseException) -> None:
        try:
            with writeLock:
                outputShm.write_response(
                    VideoResponse(
                        task_id=taskId,
                        status=VideoStatus.ERROR,
                        file_path="",
                        error_message=str(err)[: VideoShm.MAX_ERROR_LEN],
                    )
                )
        except Exception as writeErr:
            logger.error(
                f"{cfg.label} failed writing error response for {taskId}: {writeErr}"
            )

    def encoderLoop() -> None:
        # Single-consumer encode thread mirroring video_runner.py post-Phase-1.
        # Sleeps encodeS to fake mp4 encode cost, then writes the response.
        while True:
            job = encodeQueue.get()
            if job is None:
                encodeQueue.task_done()
                return
            try:
                time.sleep(cfg.encodeS)
                mp4Path = VideoManager().export_to_mp4(job.frames)
                respondedAt = time.monotonic()
                with writeLock:
                    outputShm.write_response(
                        VideoResponse(
                            task_id=job.taskId,
                            status=VideoStatus.SUCCESS,
                            file_path=mp4Path,
                            error_message="",
                        )
                    )
                depth = adjustInFlight(-1)
                logger.info(
                    f"{cfg.label} DONE  task={job.taskId[:8]}  in_flight={depth}  "
                    f"t_resp={respondedAt:.3f}  "
                    f"encode_wait={respondedAt - job.enqueuedAt:.3f}s"
                )
            except Exception as e:
                adjustInFlight(-1)
                logger.error(f"{cfg.label} ENCODE-ERROR task={job.taskId}: {e}")
                writeError(job.taskId, e)
            finally:
                encodeQueue.task_done()

    def handleRequest(req) -> None:
        # Inference-only path; encode is offloaded to encoderThread so this
        # worker can immediately accept the next request from the pool.
        taskId = req.task_id
        receivedAt = time.monotonic()
        depth = adjustInFlight(+1)
        logger.info(
            f"{cfg.label} START task={taskId[:8]}  in_flight={depth}  "
            f"t_recv={receivedAt:.3f}"
        )
        try:
            if cfg.perRequestValidator is not None:
                cfg.perRequestValidator(req)
            time.sleep(cfg.latencyS)
            inferDoneAt = time.monotonic()
            encodeQueue.put(
                _EncodeJob(
                    taskId=taskId,
                    frames=placeholderFrames,
                    enqueuedAt=inferDoneAt,
                )
            )
            logger.info(
                f"{cfg.label} INFER task={taskId[:8]}  "
                f"infer_dur={inferDoneAt - receivedAt:.3f}s  "
                f"qdepth={encodeQueue.qsize()}"
            )
        except Exception as e:
            adjustInFlight(-1)
            logger.error(f"{cfg.label} INFER-ERROR task={taskId}: {e}")
            writeError(taskId, e)

    encoderThread = threading.Thread(
        target=encoderLoop, name="mock-encoder", daemon=False
    )
    encoderThread.start()

    warmupAckCount = [0]  # mutable counter captured by the closure

    def replyWarmupPing(taskId: str) -> None:
        """SP readiness ping: respond SUCCESS immediately. No fake inference,
        no encoder hand-off, no mp4 leak. Mirrors the rank-0 short-circuit
        in ``video_runner.py`` so end-to-end testing of the
        ``SP_REQUIRE_WARMUP_PING`` contract works against the mock too.

        Includes a sequence counter in the log line so that consecutive
        acks (multi-worker boot, server restart against same mock) are
        distinguishable in a single log tail. Without this all the lines
        read identically and you can't tell server-1's ack from server-2's."""
        try:
            with writeLock:
                outputShm.write_response(
                    VideoResponse(
                        task_id=taskId,
                        status=VideoStatus.SUCCESS,
                        file_path="",
                        error_message="",
                    )
                )
            warmupAckCount[0] += 1
            logger.info(
                f"{cfg.label} replied to control ping {taskId} (#{warmupAckCount[0]})"
            )
        except Exception as err:
            logger.error(f"{cfg.label} failed writing control-ping response: {err}")

    try:
        while not _shutdown:
            req = inputShm.read_request()
            if req is None:
                break
            if req.task_id in (SP_WARMUP_TASK_ID, GAS_PROBE_TASK_ID):
                # Single-process mock has no MPI ranks, so the gas probe has no
                # collective to run — an immediate SUCCESS ack is the right
                # mock of a healthy pipeline.
                replyWarmupPing(req.task_id)
                continue
            pool.submit(handleRequest, req)
    finally:
        logger.info(f"{cfg.label} shutting down — draining inference workers...")
        pool.shutdown(wait=True)
        logger.info(f"{cfg.label} draining encoder queue...")
        encodeQueue.put(None)
        encoderThread.join(timeout=ENCODER_JOIN_TIMEOUT_S)
        if encoderThread.is_alive():
            logger.warning(
                f"{cfg.label} encoder thread did not drain within "
                f"{ENCODER_JOIN_TIMEOUT_S}s; abandoning"
            )
        inputShm.close()
        outputShm.close()
        removed = cleanup_orphaned_video_files()
        if removed:
            logger.info(f"{cfg.label} cleaned up {removed} orphaned video file(s)")
        logger.info(f"{cfg.label} shut down")
