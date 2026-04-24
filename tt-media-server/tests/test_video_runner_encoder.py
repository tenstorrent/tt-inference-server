# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the rank-0 background encoder thread in ``video_runner``.

These tests pin three behaviors the production loop relies on:

1. **FIFO + completeness** — N enqueued jobs produce N ``write_response`` calls
   in the same order they were submitted (single consumer of the queue, single
   writer of the output SHM).
2. **Error isolation** — a failing ``export_to_mp4`` reports ``VideoStatus.ERROR``
   for that ``task_id`` and the encoder keeps running for subsequent jobs (one
   bad clip must never wedge the pipeline).
3. **Bounded-time shutdown** — the ``None`` sentinel drains queued jobs and the
   thread exits within a reasonable bound.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ipc.video_shm import VideoStatus
from tt_model_runners.video_runner import (
    ENCODER_QUEUE_MAXSIZE,
    _EncodeJob,
    _encoder_loop,
)

_DRAIN_BOUND_S = 5.0


class _FakeVideoManager:
    """Drop-in for ``utils.video_manager.VideoManager`` in the encoder loop.

    Records every call, optionally sleeps to simulate real ffmpeg latency, and
    raises for explicitly-listed task ids so we can hit the error-handling
    branch without touching ffmpeg.
    """

    def __init__(
        self,
        sleep_s: float = 0.0,
        fail_task_ids: set[str] | None = None,
    ):
        self.sleep_s = sleep_s
        self.fail_task_ids = fail_task_ids or set()
        self.calls: list[str] = []

    def export_to_mp4(self, frames, fps: int = 16) -> str:  # noqa: ARG002
        # Pull task_id off the frames object for fakeability — see job.frames
        # construction in the tests below.
        task_id = frames["task_id"] if isinstance(frames, dict) else str(frames)
        self.calls.append(task_id)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if task_id in self.fail_task_ids:
            raise RuntimeError(f"simulated encode failure for {task_id}")
        return f"/tmp/{task_id}.mp4"


def _make_job(task_id: str) -> _EncodeJob:
    # frames is dict so the fake manager can recover task_id without numpy.
    return _EncodeJob(task_id=task_id, frames={"task_id": task_id})


def _start_encoder(output_shm, encode_queue) -> threading.Thread:
    t = threading.Thread(
        target=_encoder_loop,
        args=(output_shm, encode_queue),
        name="test-video-encoder",
        daemon=True,
    )
    t.start()
    return t


def _shutdown_and_join(encode_queue: queue.Queue, thread: threading.Thread) -> None:
    encode_queue.put(None)
    thread.join(timeout=_DRAIN_BOUND_S)
    assert not thread.is_alive(), (
        f"encoder thread failed to exit within {_DRAIN_BOUND_S}s; "
        f"qsize={encode_queue.qsize()}"
    )


class TestEncoderHappyPath:
    def test_n_jobs_produce_n_responses_in_order(self):
        """5 enqueued jobs → 5 SUCCESS responses, in submission order."""
        output_shm = MagicMock()
        encode_queue: queue.Queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
        fake_vm = _FakeVideoManager(sleep_s=0.01)

        with patch("utils.video_manager.VideoManager", return_value=fake_vm):
            thread = _start_encoder(output_shm, encode_queue)
            try:
                task_ids = [f"task-{i}" for i in range(5)]
                for tid in task_ids:
                    encode_queue.put(_make_job(tid))
            finally:
                _shutdown_and_join(encode_queue, thread)

        assert fake_vm.calls == task_ids
        assert output_shm.write_response.call_count == 5
        seen_order = [
            c.args[0].task_id for c in output_shm.write_response.call_args_list
        ]
        assert seen_order == task_ids
        for call in output_shm.write_response.call_args_list:
            resp = call.args[0]
            assert resp.status == VideoStatus.SUCCESS
            assert resp.file_path == f"/tmp/{resp.task_id}.mp4"
            assert resp.error_message == ""


class TestEncoderErrorIsolation:
    def test_one_failure_does_not_kill_encoder(self):
        """Failing job → ERROR response for it; subsequent jobs still succeed."""
        output_shm = MagicMock()
        encode_queue: queue.Queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
        fake_vm = _FakeVideoManager(fail_task_ids={"task-bad"})

        with patch("utils.video_manager.VideoManager", return_value=fake_vm):
            thread = _start_encoder(output_shm, encode_queue)
            try:
                for tid in ("task-ok-0", "task-bad", "task-ok-1", "task-ok-2"):
                    encode_queue.put(_make_job(tid))
            finally:
                _shutdown_and_join(encode_queue, thread)

        responses = {
            c.args[0].task_id: c.args[0]
            for c in output_shm.write_response.call_args_list
        }

        assert set(responses) == {
            "task-ok-0",
            "task-bad",
            "task-ok-1",
            "task-ok-2",
        }, "every enqueued job must produce exactly one response"

        assert responses["task-bad"].status == VideoStatus.ERROR
        assert "simulated encode failure" in responses["task-bad"].error_message
        assert responses["task-bad"].file_path == ""

        for tid in ("task-ok-0", "task-ok-1", "task-ok-2"):
            assert responses[tid].status == VideoStatus.SUCCESS, (
                f"{tid} must succeed even though task-bad failed"
            )
            assert responses[tid].file_path == f"/tmp/{tid}.mp4"


class TestEncoderBackPressure:
    def test_put_blocks_when_queue_full_and_encoder_slow(self):
        """The ``maxsize=ENCODER_QUEUE_MAXSIZE`` bound is the load-bearing
        memory guarantee (frame buffers are hundreds of MB). If a slow
        encoder lets the queue fill, producers MUST block — not silently
        grow the backlog. ``queue.Full`` on a non-blocking put proves the
        bound is active.
        """
        output_shm = MagicMock()
        encode_queue: queue.Queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
        fake_vm = _FakeVideoManager(sleep_s=0.5)

        with patch("utils.video_manager.VideoManager", return_value=fake_vm):
            thread = _start_encoder(output_shm, encode_queue)
            try:
                # Fill the queue: 1 in-flight (held by slow encoder) plus
                # ENCODER_QUEUE_MAXSIZE pending. The very next put must fail
                # immediately — proves maxsize is enforced end-to-end.
                for i in range(ENCODER_QUEUE_MAXSIZE + 1):
                    encode_queue.put(_make_job(f"bp-{i}"), timeout=1.0)

                with pytest.raises(queue.Full):
                    encode_queue.put(_make_job("bp-overflow"), block=False)
            finally:
                _shutdown_and_join(encode_queue, thread)


class TestEncoderUnifiedWriter:
    """Error jobs (`job.error` set) must be written as ERROR responses by the
    encoder thread itself — this is what keeps it the *single writer* of
    ``output_shm`` and preserves submission-order FIFO across success/error.
    """

    def test_error_job_writes_error_without_calling_ffmpeg(self):
        output_shm = MagicMock()
        encode_queue: queue.Queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)
        fake_vm = _FakeVideoManager()

        with patch("utils.video_manager.VideoManager", return_value=fake_vm):
            thread = _start_encoder(output_shm, encode_queue)
            try:
                encode_queue.put(
                    _EncodeJob(task_id="inference-fail", error="upstream boom")
                )
            finally:
                _shutdown_and_join(encode_queue, thread)

        assert fake_vm.calls == [], "export_to_mp4 must not run for error jobs"
        assert output_shm.write_response.call_count == 1
        resp = output_shm.write_response.call_args.args[0]
        assert resp.task_id == "inference-fail"
        assert resp.status == VideoStatus.ERROR
        assert resp.error_message == "upstream boom"
        assert resp.file_path == ""

    def test_mixed_success_and_error_preserve_fifo(self):
        """Interleaved success + error jobs produce responses in submission
        order. This is the property the unified-writer refactor exists to
        guarantee (prior design wrote error responses directly, bypassing
        the queue and allowing them to overtake in-flight encodes).
        """
        output_shm = MagicMock()
        encode_queue: queue.Queue = queue.Queue(maxsize=8)
        fake_vm = _FakeVideoManager(sleep_s=0.02)

        submitted = [
            _make_job("ok-0"),
            _EncodeJob(task_id="fail-1", error="upstream 1"),
            _make_job("ok-2"),
            _EncodeJob(task_id="fail-3", error="upstream 3"),
            _make_job("ok-4"),
        ]

        with patch("utils.video_manager.VideoManager", return_value=fake_vm):
            thread = _start_encoder(output_shm, encode_queue)
            try:
                for job in submitted:
                    encode_queue.put(job)
            finally:
                _shutdown_and_join(encode_queue, thread)

        seen = [c.args[0].task_id for c in output_shm.write_response.call_args_list]
        assert seen == [j.task_id for j in submitted]

        seen_statuses = {
            c.args[0].task_id: c.args[0].status
            for c in output_shm.write_response.call_args_list
        }
        assert seen_statuses == {
            "ok-0": VideoStatus.SUCCESS,
            "fail-1": VideoStatus.ERROR,
            "ok-2": VideoStatus.SUCCESS,
            "fail-3": VideoStatus.ERROR,
            "ok-4": VideoStatus.SUCCESS,
        }
        # Error jobs must bypass ffmpeg entirely.
        assert fake_vm.calls == ["ok-0", "ok-2", "ok-4"]


class TestEncoderShutdown:
    def test_sentinel_drains_pending_jobs_then_exits(self):
        """Enqueue 3 jobs, push sentinel, expect all 3 written before exit."""
        output_shm = MagicMock()
        # Use a deeper queue so we can pre-load all jobs + sentinel without
        # the slow encoder back-pressuring our test thread.
        encode_queue: queue.Queue = queue.Queue(maxsize=8)
        # Slow enough that the sentinel is queued well behind the jobs but
        # fast enough that the test stays under the drain bound.
        fake_vm = _FakeVideoManager(sleep_s=0.05)

        with patch("utils.video_manager.VideoManager", return_value=fake_vm):
            thread = _start_encoder(output_shm, encode_queue)
            try:
                for i in range(3):
                    encode_queue.put(_make_job(f"shutdown-{i}"))
            finally:
                _shutdown_and_join(encode_queue, thread)

        assert output_shm.write_response.call_count == 3
        seen = [c.args[0].task_id for c in output_shm.write_response.call_args_list]
        assert seen == ["shutdown-0", "shutdown-1", "shutdown-2"]

    def test_sentinel_alone_exits_immediately(self):
        """No pending jobs → sentinel → encoder exits without writing anything."""
        output_shm = MagicMock()
        encode_queue: queue.Queue = queue.Queue(maxsize=ENCODER_QUEUE_MAXSIZE)

        with patch(
            "utils.video_manager.VideoManager", return_value=_FakeVideoManager()
        ):
            thread = _start_encoder(output_shm, encode_queue)
            _shutdown_and_join(encode_queue, thread)

        output_shm.write_response.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
