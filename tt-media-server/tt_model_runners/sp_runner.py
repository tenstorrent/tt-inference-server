# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared-memory pipeline runner (server-side proxy).

Forwards requests through :class:`VideoShm` to an external runner process
(``video_runner.py``/``mock_video_runner.py``) instead of loading a model.

Async API: :meth:`submit` writes to the input ring and registers a
``task_id``-keyed future; a lazy single drainer thread reads the output
ring and resolves the matching future (out-of-order safe);
:meth:`await_result` awaits with a per-request timeout. :meth:`_run_async`
is the submit+await combo used by the device_worker's continuous fan-out
(opt-in via ``supports_continuous_fan_out``). Sync :meth:`run` kept for
legacy callers.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import Future as CFuture

from ipc.video_shm import (
    VideoRequest,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_video_files,
)
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time

DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_NUM_FRAMES = 81
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.0
DEFAULT_VIDEO_GUIDANCE_SCALE_2 = 4.0

# Drainer poll cadence on the output SHM. Read_response itself spin-waits
# internally; this is just the granularity at which the drainer notices
# shutdown.
_DRAINER_READ_TIMEOUT_S = 1.0
# Bound the join() at shutdown so a wedged read never holds up close_device.
_DRAINER_JOIN_TIMEOUT_S = 2.0


class SPRunner(BaseDeviceRunner):
    """Proxy runner that bridges the device-worker to an external video runner via SHM."""

    # Opt in to ``device_worker.continuousFanOut``. The SHM input ring (8
    # slots) plus the encoder thread on the external runner pipeline
    # inference + MP4 encode; continuous fan-out keeps the ring primed
    # across batch boundaries instead of draining at the tail of every
    # gather. Per-request deadlines are enforced inside ``await_result``
    # via ``video_request_timeout_seconds`` (the per-batch
    # ``threading.Timer`` is cancelled by the worker when this is True).
    supports_continuous_fan_out: bool = True

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self._input_shm: VideoShm | None = None
        self._output_shm: VideoShm | None = None
        self._shutdown = False

        # task_id -> Future[VideoResponse]. Owned jointly by submit() (insert),
        # the drainer thread (resolve+pop), and await_result (timeout-pop).
        self._pending: dict[str, CFuture] = {}
        self._pending_lock = threading.Lock()

        self._drainer: threading.Thread | None = None
        self._drainer_lock = threading.Lock()

    def _is_shutdown(self) -> bool:
        return self._shutdown

    def set_device(self):
        input_name = os.environ.get("TT_VIDEO_SHM_INPUT", "tt_video_in")
        output_name = os.environ.get("TT_VIDEO_SHM_OUTPUT", "tt_video_out")

        self._input_shm = VideoShm(
            input_name, mode="input", is_shutdown=self._is_shutdown
        )
        self._output_shm = VideoShm(
            output_name, mode="output", is_shutdown=self._is_shutdown
        )
        self._input_shm.open()
        self._output_shm.open()
        # Self-heal any gap left by a previous server instance that crashed
        # mid-write (on input) or mid-read (on output). Scoped to this
        # process's own role, so safe to run with a live runner peer.
        in_repair = self._input_shm.recover(side="writer")
        out_repair = self._output_shm.recover(side="reader")
        if any(in_repair.values()) or any(out_repair.values()):
            self.logger.warning(
                f"SPRunner {self.device_id}: crash-recovery repaired prior "
                f"inconsistency: input={in_repair} output={out_repair}"
            )
        # Any responses sitting in the output ring at startup are addressed to
        # tasks whose requester (the previous SPRunner process) is gone. Leaving
        # them in place would desync ridx by N and every future request would
        # silently receive the previous task's file_path. Drain + unlink now.
        self._drain_stale_responses()
        self.logger.info(
            f"SPRunner {self.device_id}: SHM opened (in={input_name}, out={output_name})"
        )
        return {}

    def _drain_stale_responses(self) -> None:
        depth = self._output_shm.queue_depth()
        drained = 0
        for _ in range(depth):
            resp = self._output_shm.read_response(timeout_s=1.0)
            if resp is None:
                break
            self._try_unlink(resp.file_path)
            drained += 1
        if drained:
            self.logger.warning(
                f"SPRunner {self.device_id}: drained {drained} stale "
                f"response(s) left by prior session"
            )

    def close_device(self):
        self._shutdown = True

        if self._drainer is not None:
            # _is_shutdown() is wired into VideoShm.read_response, so the
            # blocking spin will exit on the next poll tick (≤ ~_POLL_INTERVAL_S
            # plus our own DRAINER_READ_TIMEOUT_S in the worst case).
            self._drainer.join(timeout=_DRAINER_JOIN_TIMEOUT_S)
            if self._drainer.is_alive():
                self.logger.warning(
                    f"SPRunner {self.device_id}: drainer did not exit within "
                    f"{_DRAINER_JOIN_TIMEOUT_S}s; leaving as daemon"
                )
            self._drainer = None

        # Cancel any callers still parked on a future so they don't hang on shutdown.
        with self._pending_lock:
            stragglers = list(self._pending.items())
            self._pending.clear()
        for task_id, fut in stragglers:
            if not fut.done():
                fut.cancel()
            self.logger.warning(
                f"SPRunner {self.device_id}: cancelled pending future for "
                f"task_id={task_id!r} during shutdown"
            )

        if self._input_shm:
            self._input_shm.close()
            self._input_shm = None
        if self._output_shm:
            self._output_shm.close()
            self._output_shm = None
        removed = cleanup_orphaned_video_files()
        if removed:
            self.logger.info(
                f"SPRunner {self.device_id}: cleaned up {removed} orphaned video file(s)"
            )
        self.logger.info(f"SPRunner {self.device_id}: SHM cleaned up")
        return True

    def load_weights(self):
        return True

    async def warmup(self) -> bool:
        self.logger.info("Skipping warmup since SHM runner has a warm start")
        return True

    # ── Async submit / await flow ─────────────────────────────────────────

    def _ensure_drainer(self) -> None:
        """Lazily start the single output-SHM drainer thread."""
        if self._drainer is not None:
            return
        with self._drainer_lock:
            if self._drainer is not None:
                return
            t = threading.Thread(
                target=self._drain_loop,
                name=f"sp-drainer-{self.device_id}",
                daemon=True,
            )
            t.start()
            self._drainer = t
            self.logger.info(f"SPRunner {self.device_id}: drainer thread started")

    def _drain_loop(self) -> None:
        """Consume responses from output SHM and resolve matching futures.
        The drainer **does not pop** from ``_pending`` — it just resolves the
        future.
        """
        while not self._shutdown:
            try:
                resp = self._output_shm.read_response(timeout_s=_DRAINER_READ_TIMEOUT_S)
            except Exception as e:
                self.logger.error(f"[SP] drainer read error: {e}")
                time.sleep(_DRAINER_READ_TIMEOUT_S)
                continue
            if resp is None:
                continue

            with self._pending_lock:
                fut = self._pending.get(resp.task_id)

            if fut is None:
                self.logger.warning(
                    f"[SP] orphan response task_id={resp.task_id!r}; unlinking"
                )
                self._try_unlink(resp.file_path)
                continue

            if fut.done():
                self._try_unlink(resp.file_path)
                continue

            fut.set_result(resp)

    def submit(self, request) -> str:
        """
        Pack request, register a future, and write it to the input SHM ring.
        Returns the ``task_id`` so callers can pair with :meth:`await_result`.
        Does **not** block on a response. If the input ring is full, the
        underlying ``write_request`` spin-waits on an EMPTY slot (natural
        back-pressure to the producer).
        """
        task_id = request._task_id

        fut: CFuture = CFuture()
        with self._pending_lock:
            if task_id in self._pending:
                raise RuntimeError(
                    f"SPRunner: duplicate task_id {task_id!r} (already in flight)"
                )
            self._pending[task_id] = fut

        try:
            self._input_shm.write_request(self._build_video_req(request, task_id))
        except Exception:
            with self._pending_lock:
                self._pending.pop(task_id, None)
            raise

        self._ensure_drainer()

        self.logger.info(f"[SP] Request {task_id} submitted to SHM input")
        return task_id

    async def await_result(self, task_id: str) -> str:
        """Await the response for ``task_id`` and return the mp4 path.
        Owns pop semantics on ``_pending``: removes the entry on success,
        error, or timeout — whichever happens first. The drainer leaves the
        entry in place so this lookup always finds the future even when the
        drainer races ahead and resolves it before we look.
        """
        with self._pending_lock:
            fut = self._pending.get(task_id)
        if fut is None:
            raise RuntimeError(
                f"SPRunner: no pending future for task_id={task_id!r} "
                "(submit() was not called or already resolved)"
            )

        timeout_s = self.settings.video_request_timeout_seconds
        try:
            resp = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout_s)
        except asyncio.TimeoutError:
            with self._pending_lock:
                self._pending.pop(task_id, None)
            # If the response arrives after this, the drainer will see no
            # entry in _pending and treat it as an orphan (unlink + warn).
            raise TimeoutError(
                f"REQUEST_TIMEOUT: response exceeded {timeout_s}s for task {task_id}"
            )

        with self._pending_lock:
            self._pending.pop(task_id, None)

        if resp.status == VideoStatus.ERROR:
            self._try_unlink(resp.file_path)
            raise RuntimeError(f"Runner error for task {task_id}: {resp.error_message}")

        mp4_path = resp.file_path
        exists = os.path.exists(mp4_path)
        size_bytes = os.path.getsize(mp4_path) if exists else None
        size_part = f"{size_bytes:,} bytes" if size_bytes is not None else "n/a"
        self.logger.info(
            f"[SP] Received mp4 path from SHM: {mp4_path} "
            f"(exists={exists}, size={size_part})"
        )
        return mp4_path

    @log_execution_time(
        "SP-Runner inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def _run_async(self, requests):
        """Async entry point used by ``device_worker`` to fan out via gather.

        One request per call — ``device_worker`` calls this concurrently for
        each request in the batch so the input ring stays primed.
        """
        request = requests[0]
        task_id = self.submit(request)
        mp4_path = await self.await_result(task_id)
        # List so device_worker's responses[i] matches one path per request.
        return [mp4_path]

    def run(self, requests):
        """Synchronous one-shot wrapper for callers without an event loop.

        Kept for backward compatibility (and unit tests). The async pipelining
        win comes from device_worker calling :meth:`_run_async` directly via
        ``asyncio.gather`` on the worker's own event loop.
        """
        return asyncio.run(self._run_async(requests))

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_video_req(request, task_id: str) -> VideoRequest:
        return VideoRequest(
            task_id=task_id,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            num_inference_steps=request.num_inference_steps or 20,
            seed=int(request.seed or 0),
            height=getattr(request, "height", DEFAULT_VIDEO_HEIGHT),
            width=getattr(request, "width", DEFAULT_VIDEO_WIDTH),
            num_frames=getattr(request, "num_frames", DEFAULT_VIDEO_NUM_FRAMES),
            guidance_scale=getattr(
                request, "guidance_scale", DEFAULT_VIDEO_GUIDANCE_SCALE
            ),
            guidance_scale_2=getattr(
                request, "guidance_scale_2", DEFAULT_VIDEO_GUIDANCE_SCALE_2
            ),
        )

    @staticmethod
    def _try_unlink(path: str) -> None:
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass
