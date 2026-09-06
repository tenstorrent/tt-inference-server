# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Worker failures must keep their type across the process boundary.

CpuWorkloadHandler marshals errors through a multiprocessing.Queue, which cannot
carry an exception object. Before the type name was sent alongside the message,
every worker failure surfaced as a bare Exception, so a client-side mistake like
over-length audio was reported as 500 instead of 400.
"""

import pytest
from utils.errors import (
    RECONSTRUCTIBLE_ERRORS,
    AudioTooLongError,
    reconstruct_worker_error,
)


def test_audio_too_long_error_is_reconstructed_by_name():
    err = reconstruct_worker_error("AudioTooLongError", "audio is 600.00s")
    assert isinstance(err, AudioTooLongError)
    assert str(err) == "audio is 600.00s"


def test_unknown_type_falls_back_to_generic_exception():
    err = reconstruct_worker_error("SomeUnregisteredError", "boom")
    assert type(err) is Exception
    assert str(err) == "boom"


def test_missing_type_name_falls_back():
    """Covers a worker still sending the old two-element payload."""
    err = reconstruct_worker_error(None, "boom")
    assert type(err) is Exception


def test_reconstructed_error_still_subclasses_value_error():
    assert isinstance(reconstruct_worker_error("AudioTooLongError", "x"), ValueError)


def test_registry_keys_match_class_names():
    for name, cls in RECONSTRUCTIBLE_ERRORS.items():
        assert name == cls.__name__


def test_error_payload_shapes_unpack():
    """The listener handles both the sentinel and the failure payload."""
    for payload, expect_type in (
        ((None, None), None),
        (("t1", "msg"), None),
        (("t1", "msg", "AudioTooLongError"), AudioTooLongError),
    ):
        task_id, error = payload[0], payload[1]
        error_type = payload[2] if len(payload) > 2 else None
        if task_id is None:
            continue
        err = reconstruct_worker_error(error_type, error)
        if expect_type is None:
            assert type(err) is Exception
        else:
            assert isinstance(err, expect_type)


def test_audio_manager_reexport_is_same_class():
    """Existing `from utils.audio_manager import AudioTooLongError` must not fork."""
    pytest.importorskip("numpy")
    from utils.audio_manager import AudioTooLongError as ReExported

    assert ReExported is AudioTooLongError


def _raise_too_long(_context, _payload):
    """Module-level so it is picklable for the worker Process."""
    from utils.errors import AudioTooLongError

    raise AudioTooLongError("Audio duration 600.00s exceeds the maximum")


def _raise_plain(_context, _payload):
    raise RuntimeError("something else broke")


@pytest.mark.asyncio
async def test_real_worker_roundtrip_preserves_type():
    """End-to-end through the actual multiprocessing queue.

    The helper-level tests above passed even when the real path was broken - the
    type was lost in the queue, not in reconstruction - so this exercises a live
    CpuWorkloadHandler rather than the helper alone.
    """
    from model_services.cpu_workload_handler import CpuWorkloadHandler

    handler = CpuWorkloadHandler("test-too-long", 1, _raise_too_long)
    try:
        with pytest.raises(AudioTooLongError) as excinfo:
            await handler.execute_task(b"ignored")
        assert "exceeds the maximum" in str(excinfo.value)
    finally:
        handler.stop_workers()


@pytest.mark.asyncio
async def test_real_worker_roundtrip_unregistered_stays_generic():
    from model_services.cpu_workload_handler import CpuWorkloadHandler

    handler = CpuWorkloadHandler("test-plain", 1, _raise_plain)
    try:
        with pytest.raises(Exception) as excinfo:
            await handler.execute_task(b"ignored")
        assert not isinstance(excinfo.value, AudioTooLongError)
        assert "something else broke" in str(excinfo.value)
    finally:
        handler.stop_workers()
