# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Rank 0's side of a multi-host mesh: the device worker's runner, wrapped."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any

from utils.logger import TTLogger

from multihost.comm import (
    CanaryMessage,
    RunMessage,
    ShutdownMessage,
    to_request_data,
)

# How long close_device waits for an in-progress canary before closing anyway.
_CLOSE_LOCK_TIMEOUT_S = 5.0


def end_job(reason: str) -> None:
    """Kill this worker process; the scheduler then exits rank 0 and tt-run stops every rank.

    The ranks can't be re-synchronised in place once one has diverged.
    """
    TTLogger().error(f"Rank 0: multi-host job cannot continue: {reason}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def discard_outputs(result: Any) -> None:
    """Delete files a throwaway run (deep canary) exported, if any."""
    paths = result if isinstance(result, (list, tuple)) else [result]
    for path in paths:
        if isinstance(path, str) and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


class MultiHostLockstepRunner:
    """Wraps rank 0's real runner so every rank runs the same calls together.

    Each ``run`` first sends the requests to the other ranks, then runs them
    locally; all ranks then meet in the runner's own collectives on the mesh.
    Rank 0 keeps the output. Anything not overridden here (``settings``,
    ``requires_image_conditioning``, ``export_in_runner``, ...) reads and
    writes through to the real runner, so the device worker can't tell the
    difference.

    One lock serialises everything that touches the mesh or the comm: the
    ranks must see requests and canary probes in the same order.

    A ``ValueError`` from the runner is a request every rank rejects
    identically before touching the mesh; it propagates as a request error.
    A failed broadcast or any other runner error leaves the ranks out of step
    (some may have received the message, or be waiting in a collective), so it
    ends the job.
    """

    def __init__(self, runner: Any, comm: Any = None):
        # ``comm``: anything with broadcast/barrier; the multihost.comm module
        # unless a test passes a fake.
        if comm is None:
            from multihost import comm
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "_comm", comm)
        object.__setattr__(self, "_mesh_lock", threading.Lock())
        object.__setattr__(self, "_followers_ready", False)
        object.__setattr__(self, "_logger", TTLogger())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._runner, name, value)

    def __repr__(self) -> str:
        return f"MultiHostLockstepRunner({self._runner!r})"

    async def warmup(self) -> bool:
        ok = await self._runner.warmup()
        if ok is False:
            return ok
        # Followers warm up on their own and then wait here, so /health only
        # turns ready once every rank is warm.
        self._logger.info("Rank 0: warm, waiting for the other ranks")
        await asyncio.to_thread(self._comm.barrier)
        object.__setattr__(self, "_followers_ready", True)
        self._logger.info("Rank 0: all ranks warm")
        return ok

    def run(self, requests: list[Any]):
        with self._mesh_lock:
            self._broadcast(
                RunMessage([to_request_data(request) for request in requests])
            )
            return self._run_locally(requests)

    def health_check(self, deep: bool = False) -> bool:
        """Canary probe that never makes a real request wait.

        If a request holds the mesh it is proof enough that every rank is
        alive (a stuck request is caught by the request timeout), so report
        healthy without touching the mesh. Otherwise a shallow probe is a
        barrier across all ranks and a deep probe replays the warmup forward
        pass on all ranks.
        """
        if not self._mesh_lock.acquire(blocking=False):
            return True
        try:
            self._broadcast(CanaryMessage(deep=deep))
            if deep:
                discard_outputs(
                    self._run_locally([self._runner._build_warmup_video_request()])
                )
            else:
                self._comm.barrier()
            return True
        finally:
            self._mesh_lock.release()

    def _broadcast(self, message: Any) -> None:
        try:
            self._comm.broadcast(message)
        except Exception as e:
            end_job(f"broadcast to the other ranks failed: {e}")

    def _run_locally(self, requests: list[Any]):
        try:
            return self._runner.run(requests)
        except ValueError:
            raise
        except Exception as e:
            end_job(f"run failed: {e}")

    def close_device(self):
        locked = self._mesh_lock.acquire(timeout=_CLOSE_LOCK_TIMEOUT_S)
        try:
            # Before the warmup barrier the followers aren't listening yet; they
            # are torn down by tt-run when this rank exits.
            if self._followers_ready:
                try:
                    self._comm.broadcast(ShutdownMessage())
                except Exception as e:
                    self._logger.warning(f"Rank 0: failed to send shutdown: {e}")
            return self._runner.close_device()
        finally:
            if locked:
                self._mesh_lock.release()


def wrap_if_multihost(runner: Any) -> Any:
    """Wrap ``runner`` when its open mesh spans more than one rank.

    Call only after ``runner.set_device()``: opening the mesh is what creates
    the tt-metal context (and starts MPI), so the check below only reads it.
    Runners that didn't open a mesh are returned unchanged.
    """
    if getattr(runner, "ttnn_device", None) is None:
        return runner
    import ttnn

    # Strict ``is True``: the binding returns a bool, and a mocked ttnn in
    # unit tests must not be mistaken for a distributed environment.
    if ttnn.using_distributed_env() is not True:
        return runner
    return MultiHostLockstepRunner(runner)
