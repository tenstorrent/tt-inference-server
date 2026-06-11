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
import json
import os
import tempfile
import threading
import time
from concurrent.futures import Future as CFuture
from concurrent.futures import TimeoutError as FuturesTimeoutError

from config.constants import CANARY_DEEP_TASK_ID, CANARY_TASK_ID, CANARY_TASK_IDS
from ipc.video_shm import (
    MAX_IMAGE_PATH_LEN,
    SP_WARMUP_TASK_ID,
    VideoRequest,
    VideoResponse,
    VideoShm,
    VideoStatus,
    cleanup_orphaned_image_files,
    cleanup_orphaned_video_files,
    image_prompts_path,
)
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time

# Interval (seconds) between "still waiting" heartbeat lines while
# SPRunner.warmup() is blocked on the pipeline's ack.
_WARMUP_HEARTBEAT_SECONDS: float = 7.0

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


def _is_warmup_ping_enabled() -> bool:
    """Feature flag for the SHM warmup round-trip.

    Default ``False`` so the server-side change can land independently of the
    matching pipeline-side handler in ``video_runner.py``. Once both sides are
    deployed, flip ``SP_REQUIRE_WARMUP_PING=true`` to get the DS-style
    eventually-consistent readiness contract (``/health`` stays 503 until the
    pipeline has finished its own warmup and replied to the ping).
    """
    return os.environ.get("SP_REQUIRE_WARMUP_PING", "false").lower() in (
        "true",
        "1",
        "yes",
    )


class SPRunner(BaseDeviceRunner):
    """Proxy runner that bridges the device-worker to an external video runner via SHM."""

    # SHM-proxy runner: never reads weights from HF; suppresses the
    # spurious "HF_TOKEN missing" warning in BaseDeviceRunner.
    requires_weights = False

    # Opt in to ``device_worker._continuous_fan_out``. The SHM input ring
    # (8 slots) plus the encoder thread on the external runner pipeline
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
        # the drainer thread (resolve, does NOT pop), and await_result (pop).
        self._pending: dict[str, CFuture] = {}
        # task_id -> I2V side-file path (empty for T2V). Lifecycle is identical
        # to _pending and the same lock guards both maps so cleanup is atomic.
        self._pending_image_paths: dict[str, str] = {}
        self._pending_lock = threading.Lock()

        self._drainer: threading.Thread | None = None
        self._drainer_lock = threading.Lock()

    def _is_shutdown(self) -> bool:
        return self._shutdown

    @property
    def _log_id(self) -> str:
        """Stable prefix for every SPRunner log line."""
        return f"SPRunner[{self.device_id or 'default'}]"

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
                f"{self._log_id}: crash-recovery repaired prior "
                f"inconsistency: input={in_repair} output={out_repair}"
            )
        # Any responses sitting in the output ring at startup are addressed to
        # tasks whose requester (the previous SPRunner process) is gone. Leaving
        # them in place would desync ridx by N and every future request would
        # silently receive the previous task's file_path. Drain + unlink now.
        self._drain_stale_responses()
        # Sweep any I2V side-files left behind by a prior crashed session.
        removed_imgs = cleanup_orphaned_image_files()
        if removed_imgs:
            self.logger.info(
                f"{self._log_id}: cleaned up {removed_imgs} orphaned image side-file(s)"
            )
        self.logger.info(
            f"{self._log_id}: SHM opened (in={input_name}, out={output_name})"
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
                f"{self._log_id}: drained {drained} stale "
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
            straggler_image_paths = list(self._pending_image_paths.items())
            self._pending_image_paths.clear()
        for task_id, fut in stragglers:
            if not fut.done():
                fut.cancel()
            self.logger.warning(
                f"SPRunner {self.device_id}: cancelled pending future for "
                f"task_id={task_id!r} during shutdown"
            )
        # Unlink any I2V side-files that the runner peer never got to consume.
        for task_id, path in straggler_image_paths:
            self._try_unlink(path)
            self.logger.warning(
                f"SPRunner {self.device_id}: unlinked orphan I2V side-file "
                f"for task_id={task_id!r} during shutdown ({path!r})"
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
                f"{self._log_id}: cleaned up {removed} orphaned video file(s)"
            )
        removed_imgs = cleanup_orphaned_image_files()
        if removed_imgs:
            self.logger.info(
                f"{self._log_id}: cleaned up {removed_imgs} orphaned image side-file(s)"
            )
        self.logger.info(f"{self._log_id}: SHM cleaned up")
        return True

    def load_weights(self):
        return True

    @staticmethod
    def _build_canary_request(task_id: str) -> VideoRequest:
        """Reserved zero-cost VideoRequest for the canary round-trip.

        The body is empty for both shallow and deep probes — the pipeline keys
        off ``task_id`` to decide whether to run a bare collective or replay its
        compiled warmup forward, so the server never has to know the shape.
        """
        return VideoRequest(
            task_id=task_id,
            prompt="",
            negative_prompt="",
            num_inference_steps=0,
            seed=0,
            height=0,
            width=0,
            num_frames=0,
            guidance_scale=0.0,
            guidance_scale_2=0.0,
            image_path="",
        )

    def _round_trip_canary(
        self, task_id: str, timeout_s: float
    ) -> VideoResponse | None:
        """Write the canary probe and read its ack within a single ``timeout_s`` budget.

        Routes through the same future/drainer machinery real requests use: the
        drainer thread owns every output-ring read and demultiplexes responses to
        ``_pending`` futures by ``task_id``. The canary MUST NOT read the ring
        itself — a second reader would race the drainer and one of them would
        miss the ack. Returns ``None`` on ring-full, timeout, or a probe that is
        already in flight (all treated as a miss by the caller).
        """
        fut: CFuture = CFuture()
        with self._pending_lock:
            if task_id in self._pending:
                self.logger.warning(
                    f"{self._log_id} canary probe: {task_id!r} already in flight; "
                    "treating as miss"
                )
                return None
            self._pending[task_id] = fut

        try:
            deadline = time.monotonic() + timeout_s
            wrote = self._input_shm.write_request(
                self._build_canary_request(task_id), timeout_s
            )
            if not wrote:
                self.logger.warning(
                    f"{self._log_id} canary probe: input ring full; treating as miss"
                )
                return None
            # Start the drainer if no real request has yet (an idle server still
            # needs the ack routed to our future instead of nobody reading it).
            self._ensure_drainer()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            return fut.result(timeout=remaining)
        except FuturesTimeoutError:
            return None
        finally:
            with self._pending_lock:
                self._pending.pop(task_id, None)

    def health_check(self, deep: bool = False) -> bool:
        """End-to-end liveness probe for the multihost video pipeline.

        Round-trips a reserved canary ``VideoRequest`` over the SAME
        single-writer SHM input ring real requests use, and waits for the
        pipeline's ack. ``video_runner`` handles ``CANARY_TASK_ID`` by
        broadcasting to all ranks and running a bare MPI collective
        (``skip=False``), so a wedged sub-rank fails this probe — unlike the
        warmup ping, which short-circuits the collective and cannot see it.

        When ``deep`` is set the probe uses ``CANARY_DEEP_TASK_ID``: the pipeline
        replays its compiled warmup forward across all ranks, so the ack also
        proves the device can still compute (not just that hosts are looping).
        Deep probes get the larger ``canary_deep_probe_timeout_seconds`` budget
        since a real forward is seconds, not the ~ms of a barrier.
        """
        if self._input_shm is None or self._output_shm is None:
            self.logger.error(f"{self._log_id} health_check called before set_device()")
            return False

        task_id = CANARY_DEEP_TASK_ID if deep else CANARY_TASK_ID
        timeout_s = (
            self.settings.canary_deep_probe_timeout_seconds
            if deep
            else self.settings.canary_probe_timeout_seconds
        )
        try:
            resp = self._round_trip_canary(task_id, timeout_s)
        except Exception:
            self.logger.exception(f"{self._log_id} canary probe: probe failed")
            return False

        if resp is None:
            return False
        self._try_unlink(resp.file_path)
        return resp.status != VideoStatus.ERROR

    async def _await_ping_ack_with_heartbeat(
        self, timeout_s: float
    ) -> VideoResponse | None:
        """Block on the output ring up to ``timeout_s``, logging progress
        every ``sp_warmup_heartbeat_seconds``. Returns the VideoResponse or
        None on timeout / shutdown / read error."""
        heartbeat_s = max(1.0, float(_WARMUP_HEARTBEAT_SECONDS))
        deadline = time.monotonic() + timeout_s
        start_t = time.monotonic()
        self.logger.debug(
            f"{self._log_id}: entering warmup-ack wait loop "
            f"(budget={timeout_s:.0f}s, heartbeat={heartbeat_s:.0f}s)"
        )

        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return None
            wait_chunk = min(heartbeat_s, remaining)

            try:
                resp = await asyncio.to_thread(
                    self._output_shm.read_response, wait_chunk
                )
            except Exception:
                # Use .exception so the full traceback surfaces — without it,
                # debugging a corrupt SHM slot in prod is needlessly painful.
                self.logger.exception(
                    f"{self._log_id} warmup: failed to read response from output ring"
                )
                return None

            if resp is not None:
                # A leftover canary ack can only be stale here: this session's
                # canary monitor starts only AFTER warmup flips the worker ready.
                # The one-shot drain in set_device() races with canary requests
                # still queued in the input ring from a prior session, which the
                # peer answers into the output ring after the drain. Discarding
                # them (instead of failing as "desynced") keeps the warmup
                # handshake robust across restarts; the deadline still bounds us.
                if resp.task_id in CANARY_TASK_IDS:
                    self.logger.warning(
                        f"{self._log_id} warmup: discarding stale canary ack "
                        f"left by a prior session, still awaiting warmup ack"
                    )
                    self._try_unlink(resp.file_path)
                    continue
                return resp

            elapsed = time.monotonic() - start_t
            self.logger.info(
                f"{self._log_id}: still waiting for pipeline warmup ack "
                f"({elapsed:.0f}s elapsed / {timeout_s:.0f}s budget)"
            )

    async def warmup(self) -> bool:
        """Block until the video pipeline is ready to serve.

        When ``SP_REQUIRE_WARMUP_PING=true``, sends a sentinel ``VideoRequest``
        through the SHM input ring and waits for the pipeline's response on
        the output ring. The pipeline only enters its SHM read loop after its
        own ``runner.warmup()`` completes, so the round-trip latency equals
        the pipeline's full cold-start time (weight load + compile across
        every MPI rank). This is what gates ``/health`` from 503 → 200.

        When the flag is off (default), we keep the legacy no-op behaviour so
        the server-side change can be deployed before the pipeline-side ping
        handler is in production.
        """
        if not _is_warmup_ping_enabled():
            self.logger.warning(
                f"{self._log_id}: SP_REQUIRE_WARMUP_PING is OFF. /health will flip to "
                "READY as soon as SHM is attached, BEFORE the video pipeline "
                "has loaded weights or compiled kernels. The first inference "
                "request will block in SHM read_response until the pipeline "
                "catches up (or times out at "
                f"{self.settings.video_request_timeout_seconds:.0f}s). Set "
                "SP_REQUIRE_WARMUP_PING=true once the matching pipeline-side "
                "handler is deployed to get truthful readiness reporting."
            )
            return True

        if self._input_shm is None or self._output_shm is None:
            self.logger.error(
                f"{self._log_id} warmup called before set_device(); cannot ping pipeline"
            )
            return False

        timeout_s = self.settings.sp_warmup_timeout_seconds
        self.logger.info(
            f"{self._log_id}: sending warmup ping to pipeline "
            f"(timeout={timeout_s:.0f}s)"
        )

        ping = VideoRequest(
            task_id=SP_WARMUP_TASK_ID,
            prompt="",
            negative_prompt="",
            num_inference_steps=0,
            seed=0,
            height=0,
            width=0,
            num_frames=0,
            guidance_scale=0.0,
            guidance_scale_2=0.0,
            image_path="",
        )

        # If the input ring already holds unconsumed pings from prior server
        # sessions, the pipeline will eventually consume one and reply — that
        # same READY response works for us (all pings share the sentinel
        # task_id). Skip the write entirely when the ring is full, so this
        # worker never spin-blocks on write_request and the pod can be
        # restarted cleanly by the orchestrator.
        pending = self._input_shm.queue_depth()
        ring_capacity = self._input_shm.INPUT_SLOTS
        if pending >= ring_capacity:
            self.logger.warning(
                f"{self._log_id} warmup: input ring full "
                f"({pending}/{ring_capacity}); waiting on an existing ping "
                f"instead of writing a new one"
            )
        else:
            # Short timeout: if write blocks, ring is degenerately full and
            # we'd rather fail fast and let the orchestrator restart us than
            # silently hang the worker for an hour.
            wrote = await asyncio.to_thread(self._input_shm.write_request, ping, 5.0)
            if not wrote:
                self.logger.error(
                    f"{self._log_id} warmup: write_request timed out "
                    f"(input ring stuck full); aborting warmup"
                )
                return False

        resp = await self._await_ping_ack_with_heartbeat(timeout_s)
        if resp is None:
            self.logger.error(
                f"{self._log_id} warmup: pipeline did not respond within {timeout_s:.0f}s"
            )
            return False

        if resp.task_id != SP_WARMUP_TASK_ID:
            # A stale response from a prior session — drop it and treat warmup
            # as failed. The scheduler will restart the worker, which will run
            # _drain_stale_responses on set_device() and try again.
            self.logger.error(
                f"{self._log_id} warmup: unexpected response task_id={resp.task_id!r} "
                f"(expected {SP_WARMUP_TASK_ID!r}); pipeline state is desynced"
            )
            self._try_unlink(resp.file_path)
            return False

        if resp.status == VideoStatus.ERROR:
            self.logger.error(
                f"{self._log_id} warmup: pipeline reported ERROR: {resp.error_message}"
            )
            return False

        self.logger.info(f"{self._log_id}: pipeline ready (warmup ping ack'd)")
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
                # Reserved control acks (canary / warmup) are routinely orphaned:
                # a probe that timed out leaves its future popped, and stale acks
                # from a prior session get flushed on recovery. Log those at debug
                # so recovery doesn't drown the logs; a real UUID orphan is unusual
                # and stays at warning.
                is_control = (
                    resp.task_id in CANARY_TASK_IDS or resp.task_id == SP_WARMUP_TASK_ID
                )
                log = self.logger.debug if is_control else self.logger.warning
                log(f"[SP] orphan response task_id={resp.task_id!r}; unlinking")
                self._try_unlink(resp.file_path)
                continue

            if fut.done():
                self._try_unlink(resp.file_path)
                continue

            fut.set_result(resp)

    def submit(self, request) -> str:
        """
        Pack request, write the optional I2V side-file, register a future, and
        write the request to the input SHM ring.

        Returns the ``task_id`` so callers can pair with :meth:`await_result`.
        Does **not** block on a response. If the input ring is full, the
        underlying ``write_request`` spin-waits on an EMPTY slot (natural
        back-pressure to the producer).

        Cleanup contract for the I2V side-file:
          * On any failure before the SHM write completes, the side-file is
            unlinked here and ``_pending_image_paths`` is left untouched.
          * Once the SHM write succeeds, the side-file path is parked under
            ``_pending_image_paths[task_id]`` and ownership transfers to
            :meth:`_run_async`'s ``finally`` (or to ``close_device`` on
            shutdown).
        """
        task_id = request._task_id
        image_path = self._write_image_side_file(request, task_id)

        fut: CFuture = CFuture()
        with self._pending_lock:
            if task_id in self._pending:
                if image_path:
                    self._try_unlink(image_path)
                raise RuntimeError(
                    f"SPRunner: duplicate task_id {task_id!r} (already in flight)"
                )
            self._pending[task_id] = fut

        try:
            self._input_shm.write_request(
                self._build_video_req(request, task_id, image_path)
            )
        except Exception:
            with self._pending_lock:
                self._pending.pop(task_id, None)
            if image_path:
                self._try_unlink(image_path)
            raise

        # Park the path for the matching await_result/close_device cleanup.
        if image_path:
            with self._pending_lock:
                self._pending_image_paths[task_id] = image_path

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

        One request per call — ``device_worker._continuous_fan_out`` invokes
        this once per in-flight slot via ``asyncio.create_task`` so the SHM
        input ring stays primed across batch boundaries. The drainer thread
        demultiplexes responses by ``task_id``, so the N concurrent calls do
        not need to share completion order.
        """
        request = requests[0]
        task_id = self.submit(request)
        try:
            mp4_path = await self.await_result(task_id)
        finally:
            # Reliable I2V side-file cleanup on every exit (success, error,
            # timeout, asyncio.CancelledError). The runner peer has finished
            # with the file by the time the response (or error / timeout)
            # surfaces, so unlinking here is always safe.
            self._pop_and_unlink_image_path(task_id)
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

    def _pop_and_unlink_image_path(self, task_id: str) -> None:
        """Remove and unlink the I2V side-file for ``task_id`` (if any).

        Idempotent: callable from success / error / timeout / shutdown paths
        without coordinating who "owns" the cleanup.
        """
        with self._pending_lock:
            path = self._pending_image_paths.pop(task_id, "")
        if path:
            self._try_unlink(path)

    @staticmethod
    def _build_video_req(request, task_id: str, image_path: str) -> VideoRequest:
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
            image_path=image_path,
        )

    @staticmethod
    def _write_image_side_file(request, task_id: str) -> str:
        """Spill ``request.image_prompts`` to a JSON side-file on tmpfs.

        Atomic publish: write to a temp file in the same directory and then
        ``os.rename`` to the final path. The runner peer never observes a
        partially-written file at the final name — it sees either no file or
        the fully-written JSON.
        """
        image_prompts = getattr(request, "image_prompts", None)
        if not image_prompts:
            return ""

        final_path = image_prompts_path(task_id)
        path_bytes = len(final_path.encode("utf-8"))
        if path_bytes > MAX_IMAGE_PATH_LEN:
            raise RuntimeError(
                f"image_prompts side-file path exceeds SHM cap "
                f"({path_bytes} > {MAX_IMAGE_PATH_LEN} bytes); "
                f"reduce TT_VIDEO_FILE_DIR length"
            )

        payload = [
            {"image": entry.image, "frame_pos": entry.frame_pos}
            for entry in image_prompts
        ]
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"tt_img_{task_id}.",
            suffix=".json.tmp",
            dir=os.path.dirname(final_path),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.rename(tmp_path, final_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return final_path

    @staticmethod
    def _try_unlink(path: str) -> None:
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass
