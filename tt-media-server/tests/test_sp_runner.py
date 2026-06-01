# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""I2V-specific behaviour of :class:`tt_model_runners.sp_runner.SPRunner`.

The server-side proxy (``SPRunner``) is the single point that has to
serialise the conditioning images to a side-file on tmpfs before handing
the request off through SHM to the runner peer. The runner peer then
reads + broadcasts that file to all MPI ranks (see ``video_runner.py``).

These tests pin the side-file lifecycle invariants that the multi-host
I2V path depends on:

  - the JSON layout of the side-file matches what ``video_runner._run_inference_loop``
    expects (``{"image", "frame_pos"}`` per entry),
  - ``image_path`` on the SHM ``VideoRequest`` points to that exact file,
  - the file is ALWAYS unlinked when ``run()`` returns (success OR error),
    so a busy server does not leak hundreds of MB per failed request on tmpfs.

T2V backward-compat (no ``image_prompts`` on the upstream request) is also
asserted: no side-file is created and ``image_path`` stays empty so the
runner peer takes its existing T2V code path unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ipc.video_shm import VideoRequest, VideoResponse, VideoStatus
from tt_model_runners.sp_runner import SPRunner

_mock_settings = MagicMock()
_mock_settings.device_mesh_shape = (1, 1)
_mock_settings.use_dynamic_batcher = False
# Concrete numeric: ``_read_response_for`` does ``time.monotonic() + timeout``.
_mock_settings.video_request_timeout_seconds = 60.0


@pytest.fixture(autouse=True)
def _patch_base_runner():
    """Stub out ``BaseDeviceRunner.__init__`` so SPRunner can be instantiated
    in a test process with no devices, no settings file, and no logger backend."""
    with patch(
        "tt_model_runners.base_device_runner.get_settings",
        return_value=_mock_settings,
    ), patch("tt_model_runners.base_device_runner.setup_runner_environment"), patch(
        "tt_model_runners.base_device_runner.TTLogger",
        return_value=MagicMock(),
    ):
        yield


@pytest.fixture
def tmp_video_dir(tmp_path, monkeypatch):
    """Redirect ``image_prompts_path`` and the cleanup glob away from /dev/shm
    so tests don't touch the real tmpfs (which would race with a live runner).
    """
    monkeypatch.setattr("ipc.video_shm.VIDEO_FILE_DIR", str(tmp_path))
    return tmp_path


class _ImagePromptStub:
    """Duck-typed stand-in for ``ImagePromptEntry`` — the SPRunner only uses
    ``.image`` and ``.frame_pos`` attributes, not the pydantic model itself.
    Decoupling here keeps the test import surface minimal and avoids dragging
    in the full domain layer."""

    def __init__(self, image: str, frame_pos: int):
        self.image = image
        self.frame_pos = frame_pos


class _MockI2VRequest:
    """Minimal duck-typed I2V request matching the attribute surface the
    SPRunner reads from."""

    def __init__(
        self,
        task_id: str = "i2v-task-id-000000000000000000",
        prompt: str = "a sunset on a beach",
        image_prompts: Optional[List[_ImagePromptStub]] = None,
        negative_prompt: str = "blurry",
        num_inference_steps: int = 20,
        seed: int = 42,
    ):
        self._task_id = task_id
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.image_prompts = image_prompts


class _MockT2VRequest:
    """T2V request without ``image_prompts`` — used to assert the I2V
    side-file branch is correctly skipped on the T2V hot path."""

    def __init__(self, task_id: str = "t2v-task-id-000000000000000000"):
        self._task_id = task_id
        self.prompt = "a forest at dawn"
        self.negative_prompt = ""
        self.num_inference_steps = 20
        self.seed = 7


def _touch_mp4_file() -> str:
    """Empty .mp4 file simulating a successful runner output."""
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="tt_video_test_")
    os.close(fd)
    return path


def _install_shm_factory(MockVideoShm):
    """Wire the patched ``VideoShm`` constructor to return mode-specific mocks.
    Pins ``queue_depth`` to 0 so ``set_device``'s startup drain is a no-op
    (matches the pattern in ``test_device_worker_video_shm.py``)."""
    mock_input = MagicMock()
    mock_output = MagicMock()
    mock_output.queue_depth.return_value = 0

    def factory(*_args, **kwargs):
        return mock_input if kwargs.get("mode") == "input" else mock_output

    MockVideoShm.side_effect = factory
    return mock_input, mock_output


class TestWriteImageSideFile:
    """``_write_image_side_file`` is the single helper that turns the inbound
    ``image_prompts`` (rich Pydantic objects) into the on-disk JSON side-file
    that the runner peer deserialises. Pin the on-disk schema explicitly —
    runner-side ``json.loads`` reconstructs ``ImagePromptEntry`` from these
    keys, so any rename here silently breaks the cross-process contract."""

    def test_returns_empty_string_for_t2v_request(self, tmp_video_dir):
        """T2V backward-compat: no ``image_prompts`` attribute → no side-file
        written, no path returned. The downstream ``image_path == ""`` triggers
        the T2V code path on the runner peer."""
        request = _MockT2VRequest(task_id="t2v-1")

        path = SPRunner._write_image_side_file(request, "t2v-1")

        assert path == ""
        assert list(tmp_video_dir.iterdir()) == []

    def test_returns_empty_string_for_empty_image_prompts(self, tmp_video_dir):
        """An I2V request that arrived with an empty list (e.g. validator
        permitted it via a future change) must NOT create an empty side-file
        — the runner peer would then try to ``json.loads`` an empty list and
        fall through to T2V, which is correct, but writing the file at all
        would still leak it on tmpfs in the I2V error path."""
        request = _MockI2VRequest(task_id="empty-1", image_prompts=[])

        path = SPRunner._write_image_side_file(request, "empty-1")

        assert path == ""
        assert list(tmp_video_dir.iterdir()) == []

    def test_writes_json_with_image_and_frame_pos_keys(self, tmp_video_dir):
        """The on-disk schema is the cross-process contract. Pin the field
        names (``image``, ``frame_pos``) so a refactor on either side is
        forced to update the other."""
        prompts = [
            _ImagePromptStub(image="b64-frame-0", frame_pos=0),
            _ImagePromptStub(image="b64-frame-40", frame_pos=40),
        ]
        request = _MockI2VRequest(task_id="task-abc", image_prompts=prompts)

        path = SPRunner._write_image_side_file(request, "task-abc")

        assert path.endswith("tt_img_task-abc.json")
        assert os.path.exists(path)
        with open(path) as f:
            payload = json.load(f)
        assert payload == [
            {"image": "b64-frame-0", "frame_pos": 0},
            {"image": "b64-frame-40", "frame_pos": 40},
        ]

    def test_path_is_under_video_file_dir(self, tmp_video_dir):
        """Side-file location must be configurable via ``TT_VIDEO_FILE_DIR``
        for tests + multi-tenant deployments. The helper uses
        ``image_prompts_path`` which honours the env var."""
        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(task_id="env-1", image_prompts=prompts)

        path = SPRunner._write_image_side_file(request, "env-1")

        assert os.path.dirname(path) == str(tmp_video_dir)

    def test_atomic_publish_leaves_no_tmp_file_on_success(self, tmp_video_dir):
        """The temp file used to stage the JSON write must be renamed away
        (not left as a sibling) once publish succeeds. Lingering
        ``*.json.tmp`` files would accumulate on tmpfs across thousands of
        I2V requests and eventually trip the operator's bootstrap sweep."""
        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(task_id="atomic-ok", image_prompts=prompts)

        path = SPRunner._write_image_side_file(request, "atomic-ok")

        assert os.path.exists(path)
        leftovers = [
            p
            for p in tmp_video_dir.iterdir()
            if p.name.startswith("tt_img_atomic-ok.") and p.name.endswith(".json.tmp")
        ]
        assert leftovers == [], (
            f"atomic publish must rename the temp file away, got: {leftovers}"
        )

    def test_atomic_publish_no_partial_file_on_dump_failure(self, tmp_video_dir):
        """If ``json.dump`` raises mid-write (disk full, OOM, etc.), the
        FINAL path must not exist. Without atomic publish the runner peer
        would see a half-written file at the final name and either
        ``json.load``-fail or parse truncated bytes."""
        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(task_id="atomic-fail", image_prompts=prompts)

        with patch(
            "tt_model_runners.sp_runner.json.dump",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                SPRunner._write_image_side_file(request, "atomic-fail")

        assert not (tmp_video_dir / "tt_img_atomic-fail.json").exists()
        leftovers = list(tmp_video_dir.glob("tt_img_atomic-fail.*.json.tmp"))
        assert leftovers == [], (
            f"failed write must clean up its own temp file, got: {leftovers}"
        )

    def test_path_exceeding_shm_cap_raises_before_write(self, tmp_path, monkeypatch):
        """``image_prompts_path`` overrunning ``MAX_IMAGE_PATH_LEN`` must
        fail-fast at the writer instead of relying on ``_pack_string`` to
        silently truncate the path on the SHM input slot — the runner peer
        would hit ``ENOENT`` on the truncated path and silently degrade I2V
        to T2V, which is the failure mode the precondition exists to avoid.
        """
        from ipc.video_shm import MAX_IMAGE_PATH_LEN

        # Pad the directory so the resulting tt_img_<task>.json path is just
        # over the SHM cap. Any task_id wider than 0 chars now overflows.
        long_dir = tmp_path / ("x" * (MAX_IMAGE_PATH_LEN - 10))
        long_dir.mkdir()
        monkeypatch.setattr("ipc.video_shm.VIDEO_FILE_DIR", str(long_dir))

        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(
            task_id="long-path-task-id-000000000000",
            image_prompts=prompts,
        )

        with pytest.raises(RuntimeError, match="exceeds SHM cap"):
            SPRunner._write_image_side_file(request, "long-path-task-id-000000000000")

        # And the precondition must fire BEFORE any file is written.
        assert list(long_dir.iterdir()) == []


class TestI2VRunSideFileLifecycle:
    """End-to-end: ``SPRunner.run()`` must (1) write the side-file, (2) set
    ``image_path`` on the SHM request to that file, and (3) unlink the file
    when run returns — independent of success / error / timeout outcome.
    The unlink invariant is the leak-prevention contract the I2V path
    relies on, so each terminal outcome gets its own assertion."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_success_path_writes_side_file_and_sets_image_path(
        self, MockVideoShm, tmp_video_dir
    ):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        mp4_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "i2v-task-id-000000000000000000",
            VideoStatus.SUCCESS,
            mp4_path,
            "",
        )

        prompts = [
            _ImagePromptStub(image="b64-1", frame_pos=0),
            _ImagePromptStub(image="b64-2", frame_pos=40),
        ]
        request = _MockI2VRequest(image_prompts=prompts)

        runner = SPRunner("dev0")
        runner.set_device()

        # Capture the on-disk JSON BEFORE run() returns (the finally clause
        # unlinks it) — proves both file existence and content in one pass.
        captured: dict = {}

        def capture_payload(req):
            captured["image_path"] = req.image_path
            with open(req.image_path) as f:
                captured["payload"] = json.load(f)

        mock_input.write_request.side_effect = capture_payload

        out = runner.run([request])

        assert out == [mp4_path]
        assert captured["image_path"].endswith(
            "tt_img_i2v-task-id-000000000000000000.json"
        )
        assert captured["payload"] == [
            {"image": "b64-1", "frame_pos": 0},
            {"image": "b64-2", "frame_pos": 40},
        ]
        assert not os.path.exists(captured["image_path"])

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_error_path_still_unlinks_side_file(self, MockVideoShm, tmp_video_dir):
        """The ``finally`` cleanup must fire on the error path too, otherwise
        a flaky model would leak ~810 MB per failed I2V request on tmpfs."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        mock_output.read_response.return_value = VideoResponse(
            "i2v-task-id-000000000000000000",
            VideoStatus.ERROR,
            "",
            "pipeline crashed",
        )

        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(image_prompts=prompts)

        captured_path: dict = {}

        def capture_path_at_write(req):
            captured_path["path"] = req.image_path
            assert os.path.exists(req.image_path)

        mock_input.write_request.side_effect = capture_path_at_write

        runner = SPRunner("dev0")
        runner.set_device()

        with pytest.raises(RuntimeError, match="pipeline crashed"):
            runner.run([request])

        assert captured_path["path"]
        assert not os.path.exists(captured_path["path"])

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_timeout_path_still_unlinks_side_file(self, MockVideoShm, tmp_video_dir):
        """``read_response`` returning None → TimeoutError. The side-file
        must STILL be unlinked, otherwise a slow runner peer leaks 810 MB
        per stuck request."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_output.read_response.return_value = None

        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(image_prompts=prompts)

        captured_path: dict = {}

        def capture_path_at_write(req):
            captured_path["path"] = req.image_path

        mock_input.write_request.side_effect = capture_path_at_write

        runner = SPRunner("dev0")
        runner.set_device()

        with pytest.raises(TimeoutError, match="REQUEST_TIMEOUT"):
            runner.run([request])

        assert captured_path["path"]
        assert not os.path.exists(captured_path["path"])

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_t2v_request_does_not_create_side_file(self, MockVideoShm, tmp_video_dir):
        """T2V backward compat: ``image_path`` on the SHM request stays
        empty so the runner peer skips the side-file read entirely. Verifies
        the I2V branch is gated on the upstream ``image_prompts`` attribute,
        not unconditionally entered."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        mp4_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "t2v-task-id-000000000000000000",
            VideoStatus.SUCCESS,
            mp4_path,
            "",
        )

        captured: dict = {}

        def capture_image_path(req):
            captured["image_path"] = req.image_path

        mock_input.write_request.side_effect = capture_image_path

        runner = SPRunner("dev0")
        runner.set_device()

        out = runner.run([_MockT2VRequest()])

        assert out == [mp4_path]
        assert captured["image_path"] == ""
        # No side-files anywhere in the redirected dir.
        assert [
            p for p in tmp_video_dir.iterdir() if p.name.startswith("tt_img_")
        ] == []

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_image_path_in_shm_request_matches_written_file(
        self, MockVideoShm, tmp_video_dir
    ):
        """Cross-check: the ``image_path`` written into the ``VideoRequest``
        SHM payload must point to the SAME file that ``_write_image_side_file``
        created. Catches a future bug where the two helpers drift apart."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mp4_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "i2v-task-id-000000000000000000",
            VideoStatus.SUCCESS,
            mp4_path,
            "",
        )

        prompts = [_ImagePromptStub(image="b64", frame_pos=0)]
        request = _MockI2VRequest(image_prompts=prompts)

        captured: dict = {}

        def capture(req):
            captured["req"] = req
            captured["existed_at_write"] = os.path.exists(req.image_path)

        mock_input.write_request.side_effect = capture

        runner = SPRunner("dev0")
        runner.set_device()
        runner.run([request])

        written: VideoRequest = captured["req"]
        assert isinstance(written, VideoRequest)
        assert written.image_path
        assert captured["existed_at_write"], (
            "side-file must exist on disk at the moment write_request is called"
        )


# ── SPRunner.warmup() readiness round-trip ──


_mock_settings.sp_warmup_timeout_seconds = 5.0


@pytest.fixture
def _disable_warmup_ping(monkeypatch):
    """Default-off behaviour: SP_REQUIRE_WARMUP_PING absent or set to false."""
    monkeypatch.delenv("SP_REQUIRE_WARMUP_PING", raising=False)


@pytest.fixture
def _enable_warmup_ping(monkeypatch):
    monkeypatch.setenv("SP_REQUIRE_WARMUP_PING", "true")


class TestSPRunnerWarmup:
    """The warmup round-trip is what gates ``/health`` from 503 → 200 under
    the new eventually-consistent readiness contract. These tests pin both
    the legacy no-op fallback (so existing deployments keep working until
    the pipeline-side ping handler is rolled out) and each branch of the
    real round-trip."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_disabled_flag_is_noop(self, MockVideoShm, _disable_warmup_ping):
        """With SP_REQUIRE_WARMUP_PING unset, warmup must return True
        immediately without touching SHM. This preserves the legacy
        deployment story until both sides are upgraded."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())

        assert result is True
        mock_input.write_request.assert_not_called()
        mock_output.read_response.assert_not_called()

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_enabled_happy_path_returns_true(self, MockVideoShm, _enable_warmup_ping):
        """Pipeline answered our ping with SUCCESS — worker reports ready."""
        from ipc.video_shm import SP_WARMUP_TASK_ID

        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        mock_output.read_response.return_value = VideoResponse(
            task_id=SP_WARMUP_TASK_ID,
            status=VideoStatus.SUCCESS,
            file_path="",
            error_message="",
        )

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())

        assert result is True
        # Exactly one ping, with the sentinel task_id.
        mock_input.write_request.assert_called_once()
        sent_ping = mock_input.write_request.call_args[0][0]
        assert sent_ping.task_id == SP_WARMUP_TASK_ID

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_enabled_timeout_returns_false(self, MockVideoShm, _enable_warmup_ping):
        """``read_response`` returning None (deadline hit) must fail warmup
        so /health stays red. The scheduler/orchestrator gets to decide
        whether to retry by restarting the worker."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        mock_output.read_response.return_value = None

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())
        assert result is False

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_enabled_error_response_returns_false(
        self, MockVideoShm, _enable_warmup_ping
    ):
        """Pipeline answered ERROR (e.g. weight-load blew up). Fail warmup so
        the operator sees /health is still red and inspects pipeline logs."""
        from ipc.video_shm import SP_WARMUP_TASK_ID

        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        mock_output.read_response.return_value = VideoResponse(
            task_id=SP_WARMUP_TASK_ID,
            status=VideoStatus.ERROR,
            file_path="",
            error_message="weight load failed: out of memory",
        )

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())
        assert result is False

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_enabled_full_ring_skips_write_and_waits(
        self, MockVideoShm, _enable_warmup_ping
    ):
        """When the input ring is degenerately full (server has restarted
        many times against a missing pipeline), we MUST NOT write another
        ping — that would spin-block the worker. Instead, wait for an
        existing ping to be consumed and treat its response as our own."""
        from ipc.video_shm import SP_WARMUP_TASK_ID

        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 8  # ring full
        mock_input.INPUT_SLOTS = 8
        mock_output.read_response.return_value = VideoResponse(
            task_id=SP_WARMUP_TASK_ID,
            status=VideoStatus.SUCCESS,
            file_path="",
            error_message="",
        )

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())

        assert result is True
        mock_input.write_request.assert_not_called()
        mock_output.read_response.assert_called_once()

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_enabled_write_timeout_returns_false(
        self, MockVideoShm, _enable_warmup_ping
    ):
        """``write_request`` reporting False (slot never freed within the
        small write deadline) means the ring is in an unrecoverable state —
        fail fast and let the orchestrator restart us."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = False

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())
        assert result is False
        mock_output.read_response.assert_not_called()

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_enabled_response_with_wrong_task_id_returns_false(
        self, MockVideoShm, _enable_warmup_ping
    ):
        """If a real (non-warmup) response slips through, treat warmup as
        failed and surface the desync. The next set_device run will drain
        the output ring so subsequent retries start clean."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        mock_output.read_response.return_value = VideoResponse(
            task_id="some-real-task-id-0000000000000000",
            status=VideoStatus.SUCCESS,
            file_path="/dev/shm/tt_video_orphan.pkl",
            error_message="",
        )

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())
        assert result is False

    def test_warmup_without_set_device_returns_false(self, _enable_warmup_ping):
        """Defensive: a buggy caller path that triggers warmup before
        set_device must not crash with AttributeError. SHM not attached →
        return False so worker init fails cleanly."""
        runner = SPRunner("dev0")

        import asyncio

        result = asyncio.run(runner.warmup())
        assert result is False

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_heartbeat_fires_during_long_wait(
        self, MockVideoShm, _enable_warmup_ping, monkeypatch
    ):
        """While blocked on the pipeline's ack, the worker must emit a
        ``still waiting`` heartbeat every ``_WARMUP_HEARTBEAT_SECONDS``
        instead of going silent. Previously a 1 h cold start was a 1 h
        log blackout. We model that here by returning ``None`` twice (two
        chunks elapsed → two heartbeat lines) before returning success on
        the third call. ``TTLogger`` has ``propagate=False`` so ``caplog``
        is useless; spy on ``runner.logger.info`` directly."""
        from ipc.video_shm import SP_WARMUP_TASK_ID

        monkeypatch.setattr(_mock_settings, "sp_warmup_timeout_seconds", 10.0)
        monkeypatch.setattr(
            "tt_model_runners.sp_runner._WARMUP_HEARTBEAT_SECONDS", 0.05
        )

        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        ok_resp = VideoResponse(
            task_id=SP_WARMUP_TASK_ID,
            status=VideoStatus.SUCCESS,
            file_path="",
            error_message="",
        )
        mock_output.read_response.side_effect = [None, None, ok_resp]

        runner = SPRunner("dev0")
        runner.set_device()
        info_calls: list[str] = []
        monkeypatch.setattr(
            runner.logger, "info", lambda msg, **kw: info_calls.append(msg)
        )

        import asyncio

        result = asyncio.run(runner.warmup())

        assert result is True, "warmup must still succeed after heartbeats"
        # Three chunks total: 2 timed out (→ heartbeat), 1 returned success.
        assert mock_output.read_response.call_count == 3
        heartbeats = [
            m for m in info_calls if "still waiting for pipeline warmup ack" in m
        ]
        assert len(heartbeats) == 2, (
            f"expected exactly 2 heartbeat lines (one per timed-out chunk), "
            f"got {len(heartbeats)}: {heartbeats}"
        )

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_heartbeat_interval_clamped_to_one_second(
        self, MockVideoShm, _enable_warmup_ping, monkeypatch
    ):
        """A misconfigured 0 s heartbeat would spin-loop the worker. Guard
        clamps the interval to >= 1 s so the worst case is one wakeup per
        second, not an unbounded CPU burn."""
        from ipc.video_shm import SP_WARMUP_TASK_ID

        monkeypatch.setattr(_mock_settings, "sp_warmup_timeout_seconds", 2.0)
        monkeypatch.setattr("tt_model_runners.sp_runner._WARMUP_HEARTBEAT_SECONDS", 0.0)

        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        # One chunk of clamped 1 s → None → second chunk (1 s) → success.
        mock_output.read_response.side_effect = [
            None,
            VideoResponse(
                task_id=SP_WARMUP_TASK_ID,
                status=VideoStatus.SUCCESS,
                file_path="",
                error_message="",
            ),
        ]

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())
        assert result is True
        # If clamping failed the loop would burn the entire 2 s budget in
        # zero-second chunks and exhaust mock side_effect → StopIteration.
        assert mock_output.read_response.call_count == 2

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_stale_canary_ack_is_skipped_then_warmup_succeeds(
        self, MockVideoShm, _enable_warmup_ping, monkeypatch
    ):
        """A canary ack left in the output ring by a prior session must be
        discarded, not treated as a fatal desync. The one-shot drain in
        set_device() races with canary requests still queued in the input
        ring, which the peer answers after the drain. Before the fix this
        surfaced as ``unexpected response task_id='__canary__'`` and warmup
        failed, wedging the worker at is_ready=False. After the fix the loop
        skips the stale canary ack and waits for the real ``__sp_warmup__``
        ack."""
        from config.constants import CANARY_TASK_ID
        from ipc.video_shm import SP_WARMUP_TASK_ID

        monkeypatch.setattr(_mock_settings, "sp_warmup_timeout_seconds", 5.0)
        # Pin the heartbeat so the ack-wait loop's chunking is deterministic and
        # independent of whatever a sibling suite left on the module global.
        monkeypatch.setattr("tt_model_runners.sp_runner._WARMUP_HEARTBEAT_SECONDS", 1.0)

        mock_input, mock_output = _install_shm_factory(MockVideoShm)
        mock_input.queue_depth.return_value = 0
        mock_input.INPUT_SLOTS = 8
        mock_input.write_request.return_value = True
        mock_output.read_response.side_effect = [
            VideoResponse(
                task_id=CANARY_TASK_ID,
                status=VideoStatus.SUCCESS,
                file_path="",
                error_message="",
            ),
            VideoResponse(
                task_id=SP_WARMUP_TASK_ID,
                status=VideoStatus.SUCCESS,
                file_path="",
                error_message="",
            ),
        ]

        runner = SPRunner("dev0")
        runner.set_device()

        import asyncio

        result = asyncio.run(runner.warmup())

        assert result is True, "stale canary ack must not fail warmup"
        # First read = stale canary probe (skipped), second = the real warmup ack.
        assert mock_output.read_response.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
