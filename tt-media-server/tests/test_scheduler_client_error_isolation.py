# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Client-input rejections must never count as worker execution errors (#4811).

Issue #4811 was re-opened because PR #4817 patched one request signature
(text-only ``POST /generations`` on an I2V-only deployment) rather than the
mechanism its own title names: ``Scheduler.error_listener`` incremented
``worker_info[...]["error_count"]`` for EVERY entry on ``error_queue``, with no
distinction between "the device is dying" and "the client sent junk". Six
bad-input requests therefore walked a perfectly healthy worker past
``max_worker_restart_count`` into a restart, and with it ``/health`` 503 and a
pod shutdown.

The chain under test spans a process boundary in production:

    runner raises  ->  device_worker  ->  error_queue  ->  error_listener
                                          (mp.Queue)       -> worker_health_monitor

``TestRogueRequestChain`` drives that whole chain in one process against a real
``multiprocessing.Queue`` — so a payload that cannot pickle fails the test
rather than silently taking the error listener down in production. The control
case (a genuine device fault still restarts the worker) is asserted alongside
every isolation case: the point is to stop miscounting client errors, not to
blind the watchdog.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import struct
import sys
from multiprocessing import Queue
from unittest.mock import Mock, patch

import pytest

# Mirror ``test_scheduler.py``'s settings stub so this module is self-sufficient
# regardless of collection order. ``max_worker_restart_count`` is deliberately
# the production default (5) so the counts below read like the issue's log.
_mock_settings = Mock()
_mock_settings.device_ids = "(0)"
_mock_settings.max_queue_size = 10
_mock_settings.max_batch_size = 1
_mock_settings.use_queue_per_worker = False
_mock_settings.use_dynamic_batcher = False
_mock_settings.queue_for_multiprocessing = "default"
_mock_settings.max_worker_restart_count = 5
_mock_settings.allow_deep_reset = False
_mock_settings.worker_check_sleep_timeout = 0.01
sys.modules["config.settings"] = Mock()
sys.modules["config.settings"].get_settings = Mock(return_value=_mock_settings)
sys.modules["config.settings"].settings = _mock_settings
sys.modules["config.settings"].Settings = Mock()

from device_workers.device_worker import _continuous_fan_out  # noqa: E402
from domain.errors import (  # noqa: E402
    ClientRequestError,
    classify_worker_error,
    parse_peer_status,
)
from model_services.scheduler import Scheduler  # noqa: E402

WORKER_ID = "0"

# Verbatim from the issue's error log — the peer stringifies an
# ``HTTPException(400, ...)``, which is where the ``400: `` prefix comes from.
PEER_DECODE_ERROR = (
    "400: Could not decode image (804 bytes): cannot identify image file "
    "<_io.BytesIO object at 0x78810f8022a0>"
)

# Rogue but plausible: valid data URL, valid base64, bytes are not an image.
# Same shape as the ``curl`` in the issue, sized to match its 804 bytes.
ROGUE_IMAGE = "data:image/png;base64," + base64.b64encode(b"x" * 804).decode("ascii")


def _make_scheduler() -> Scheduler:
    with patch(
        "model_services.scheduler.get_settings", return_value=_mock_settings
    ), patch("model_services.scheduler.TTLogger", return_value=Mock()):
        scheduler = Scheduler()
    scheduler.worker_info[WORKER_ID] = {
        "process": Mock(is_alive=Mock(return_value=True)),
        "start_time": 0.0,
        "restart_count": 0,
        "is_ready": True,
        "error_count": 0,
        "queue_index": 0,
    }
    return scheduler


async def _drain(scheduler: Scheduler, payloads: list) -> None:
    """Feed *payloads* through the real ``error_listener`` and let it exit."""
    for payload in payloads:
        scheduler.error_queue.put((WORKER_ID, payload[0], payload[1]))
    scheduler.error_queue.put((WORKER_ID, None, None))  # shutdown sentinel
    await asyncio.wait_for(scheduler.error_listener(), timeout=10.0)


async def _run_one_monitor_pass(scheduler: Scheduler) -> Mock:
    """Run a single ``worker_health_monitor`` iteration; return the restart mock."""
    scheduler.is_ready = True
    scheduler.monitor_running = True
    first_sleep = True

    async def stop_after_one(_timeout):
        nonlocal first_sleep
        if first_sleep:
            first_sleep = False
            scheduler.monitor_running = False

    with patch.object(scheduler, "restart_worker") as restart, patch(
        "model_services.scheduler.asyncio.sleep", side_effect=stop_after_one
    ):
        await asyncio.wait_for(scheduler.worker_health_monitor(), timeout=5.0)
    return restart


class TestClientRequestErrorContract:
    """``ClientRequestError`` is the marker that survives the process hop."""

    @pytest.mark.parametrize(
        "error,expected_status",
        [
            (ClientRequestError("bad image", status_code=422), 422),
            (ClientRequestError("bad image"), 400),  # default status
        ],
    )
    def test_survives_the_multiprocessing_queue(self, error, expected_status):
        """The error crosses an ``mp.Queue``, so it MUST serialise.

        Starlette's ``HTTPException`` does not call ``super().__init__``, so its
        ``args`` are empty and a naive subclass fails to reconstruct — which would
        kill ``error_listener`` instead of reporting the error. Asserted through a
        real queue rather than by calling a serialiser directly, so the test
        exercises the production transport.
        """
        queue: Queue = Queue()
        queue.put(error)

        revived = queue.get(timeout=5.0)

        assert isinstance(revived, ClientRequestError)
        assert revived.status_code == expected_status
        assert revived.detail == "bad image"

    def test_reduce_carries_both_constructor_arguments(self):
        """Pin the ``__reduce__`` hook the queue above depends on.

        The queue test proves the round trip works; this one says *why*, so a
        regression points straight at the hook instead of at the transport.
        """
        factory, args = ClientRequestError("nope", status_code=413).__reduce__()

        assert factory is ClientRequestError
        assert args == ("nope", 413)

        rebuilt = factory(*args)
        assert rebuilt.status_code == 413
        assert rebuilt.detail == "nope"

    def test_str_matches_the_peer_wire_format(self):
        """Rank 0 writes ``str(exc)`` into SHM, and the peer's ``400: `` prefix
        is what ``parse_peer_status`` reads back on the server side. Keeping
        ``__str__`` inherited from HTTPException closes that loop."""
        assert str(ClientRequestError("Could not decode image")) == (
            "400: Could not decode image"
        )

    def test_is_an_http_exception(self):
        """So ``_submit_video_request``'s ``except HTTPException: raise`` hands
        the client a 4xx without any per-endpoint plumbing."""
        from fastapi import HTTPException

        assert isinstance(ClientRequestError("x"), HTTPException)


class TestParsePeerStatus:
    """Translate the multihost peer's stringified error back into a status."""

    def test_reads_the_status_from_the_issues_error_message(self):
        assert parse_peer_status(PEER_DECODE_ERROR) == 400

    @pytest.mark.parametrize("status", [400, 413, 422, 499])
    def test_accepts_any_4xx(self, status):
        assert parse_peer_status(f"{status}: something the client sent") == status

    @pytest.mark.parametrize(
        "message",
        [
            "500: internal pipeline failure",
            "503: peer unavailable",
            "ttnn device hang",
            "Runner error for task abc: watchdog timeout",
            "",
            "400 no colon so not a status prefix",
            "40: too short to be a status",
        ],
    )
    def test_rejects_everything_that_is_not_a_4xx_prefix(self, message):
        """Anything unrecognised stays a worker fault — the conservative
        direction. A missed classification costs a false restart; a wrong one
        would blind the watchdog."""
        assert parse_peer_status(message) is None

    def test_tolerates_a_non_string(self):
        assert parse_peer_status(None) is None


class TestClassifyWorkerError:
    """The type-based safety net for rogue params nobody validated yet."""

    def test_client_request_error_is_passed_through_unchanged(self):
        original = ClientRequestError("bad base64", status_code=422)

        assert classify_worker_error(WORKER_ID, original) is original

    @pytest.mark.parametrize(
        "exc",
        [
            binascii.Error("Invalid base64-encoded string"),
            struct.error("argument out of range"),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ],
    )
    def test_input_shape_failures_classify_as_client_errors(self, exc):
        """``struct.error`` is the live example: ``seed`` outside int64 blows up
        in ``VideoShm.write_request`` inside the worker."""
        classified = classify_worker_error(WORKER_ID, exc)

        assert isinstance(classified, ClientRequestError)
        assert classified.status_code == 400

    def test_pydantic_validation_error_classifies_as_client_error(self):
        from pydantic import ValidationError

        from domain.video_generate_request import VideoGenerateRequest

        try:
            VideoGenerateRequest()  # missing required prompt
        except ValidationError as e:
            exc = e

        assert isinstance(classify_worker_error(WORKER_ID, exc), ClientRequestError)

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("ttnn device hang"),
            ValueError("tensor shape mismatch on device"),
            OSError("SHM segment gone"),
            TimeoutError("REQUEST_TIMEOUT: response exceeded 600s"),
        ],
    )
    def test_device_faults_stay_worker_faults(self, exc):
        classified = classify_worker_error(WORKER_ID, exc, prefix="Worker 0 error: ")

        assert not isinstance(classified, ClientRequestError)
        assert isinstance(classified, str)
        assert "Worker 0 error: " in classified

    def test_peer_4xx_runtime_error_is_reclassified(self):
        """``SPRunner`` raises ``ClientRequestError`` for a 4xx peer message, but
        classify by message as well so an unconverted path still can't cost the
        worker its life."""
        classified = classify_worker_error(
            WORKER_ID, RuntimeError(f"Runner error for task abc: {PEER_DECODE_ERROR}")
        )

        assert isinstance(classified, ClientRequestError)
        assert classified.status_code == 400


class TestErrorAccounting:
    """``error_listener`` is where #4811 actually lived."""

    @pytest.mark.asyncio
    async def test_client_error_does_not_count_toward_worker_death(self):
        scheduler = _make_scheduler()

        await _drain(scheduler, [("task-a", ClientRequestError("bad image"))])

        assert scheduler.worker_info[WORKER_ID]["error_count"] == 0

    @pytest.mark.asyncio
    async def test_client_error_is_still_counted_for_observability(self):
        scheduler = _make_scheduler()

        await _drain(scheduler, [("task-a", ClientRequestError("bad image"))])

        assert scheduler.worker_info[WORKER_ID]["client_error_count"] == 1

    @pytest.mark.asyncio
    async def test_device_fault_still_counts(self):
        """The watchdog must keep working — this is the control case."""
        scheduler = _make_scheduler()

        await _drain(scheduler, [("task-a", "Worker 0 execution error: device hang")])

        assert scheduler.worker_info[WORKER_ID]["error_count"] == 1

    @pytest.mark.asyncio
    async def test_client_error_reaches_the_caller_as_a_4xx(self):
        """Previously the caller got a bare ``Exception`` and the endpoint turned
        it into a 500 for a request the client got wrong."""
        scheduler = _make_scheduler()
        scheduler.result_queues = {"task-a": asyncio.Queue()}

        await _drain(
            scheduler, [("task-a", ClientRequestError(PEER_DECODE_ERROR, 400))]
        )

        surfaced = await scheduler.result_queues["task-a"].get()
        assert surfaced.status_code == 400
        assert "Could not decode image" in str(surfaced.detail)

    @pytest.mark.asyncio
    async def test_client_error_on_a_chunked_task_id_routes_to_the_task(self):
        """Streaming keys arrive as ``<task_id>_chunk_<n>``; the 4xx has to land
        on the parent task's queue like the string path already does."""
        scheduler = _make_scheduler()
        scheduler.result_queues = {"task-a": asyncio.Queue()}

        await _drain(scheduler, [("task-a_chunk_0", ClientRequestError("bad image"))])

        surfaced = await scheduler.result_queues["task-a"].get()
        assert surfaced.status_code == 400

    def test_restart_resets_error_count(self):
        """A fresh process has no errors. Carrying ``old - 1`` forward left a
        restarted worker one bad request away from the next restart."""
        scheduler = _make_scheduler()
        scheduler.result_queues_by_worker = {0: Mock()}
        scheduler.worker_info[WORKER_ID]["error_count"] = 6
        scheduler.worker_info[WORKER_ID]["process"] = Mock(
            is_alive=Mock(return_value=False)
        )

        with patch("model_services.scheduler.Process"), patch(
            "model_services.scheduler.mark_worker_dead"
        ):
            scheduler.restart_worker(WORKER_ID)

        assert scheduler.worker_info[WORKER_ID]["error_count"] == 0
        assert scheduler.worker_info[WORKER_ID]["restart_count"] == 1


class TestRogueRequestChain:
    """End to end, from the runner's failure to the watchdog's decision."""

    class _Request:
        """What the worker sees of a ``VideoI2VGenerateRequest``."""

        def __init__(self, task_id: str, image: str) -> None:
            self._task_id = task_id
            self.image_prompts = [Mock(image=image, frame_pos=0)]

    class _DecodingRunner:
        """Fails the way the I2V runners fail on a bad conditioning image.

        ``dit_runners._build_image_prompt`` calls ``base64_to_pil_image``
        directly, so going through the real ``ImageManager`` reproduces the
        production failure instead of asserting against a hand-made exception.
        """

        supports_continuous_fan_out = True

        async def _run_async(self, requests):
            from utils.image_manager import ImageManager

            entry = requests[0].image_prompts[0]
            return [ImageManager().base64_to_pil_image(entry.image)]

    class _DyingDeviceRunner:
        supports_continuous_fan_out = True

        async def _run_async(self, requests):
            raise RuntimeError("ttnn: device watchdog fired, mesh unresponsive")

    async def _run_requests(self, runner, count: int) -> tuple:
        """Push *count* requests through the real fan-out and error listener."""
        error_queue: Queue = Queue()
        requests = [
            TestRogueRequestChain._Request(f"task-{i}", ROGUE_IMAGE)
            for i in range(count)
        ]

        task_queue = Mock()
        task_queue.get_many = Mock(return_value=[])

        await _continuous_fan_out(
            device_runner=runner,
            initial_requests=requests,
            worker_id=WORKER_ID,
            result_queue=Mock(),
            error_queue=error_queue,
            task_queue=task_queue,
            max_inflight=count,
            logger=Mock(),
        )

        scheduler = _make_scheduler()
        scheduler.error_queue = error_queue
        scheduler.result_queues = {r._task_id: asyncio.Queue() for r in requests}
        error_queue.put((WORKER_ID, None, None))
        await asyncio.wait_for(scheduler.error_listener(), timeout=10.0)
        return scheduler, requests

    @pytest.mark.asyncio
    async def test_rogue_image_requests_never_restart_the_worker(self):
        """The issue verbatim: 6 bad-input requests used to log
        ``Worker 0 has too many errors (6), restarting``."""
        over_threshold = _mock_settings.max_worker_restart_count + 1

        scheduler, _ = await self._run_requests(self._DecodingRunner(), over_threshold)
        restart = await _run_one_monitor_pass(scheduler)

        assert scheduler.worker_info[WORKER_ID]["error_count"] == 0
        assert scheduler.worker_info[WORKER_ID]["client_error_count"] == over_threshold
        restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_rogue_image_request_answers_the_client_with_a_400(self):
        scheduler, requests = await self._run_requests(self._DecodingRunner(), 1)

        surfaced = await scheduler.result_queues[requests[0]._task_id].get()
        assert surfaced.status_code == 400

    @pytest.mark.asyncio
    async def test_genuine_device_faults_still_restart_the_worker(self):
        """Control case. If this ever goes green-by-accident the watchdog is
        blind and #4811's fix has overshot."""
        over_threshold = _mock_settings.max_worker_restart_count + 1

        scheduler, _ = await self._run_requests(
            self._DyingDeviceRunner(), over_threshold
        )
        restart = await _run_one_monitor_pass(scheduler)

        assert scheduler.worker_info[WORKER_ID]["error_count"] == over_threshold
        restart.assert_called_once_with(WORKER_ID)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
