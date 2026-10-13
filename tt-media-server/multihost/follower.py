# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Ranks 1..N-1 of a multi-host mesh: run whatever rank 0 runs, discard the output."""

from __future__ import annotations

import asyncio
import os
import signal
import traceback
from typing import Any, Callable

from utils.logger import TTLogger

from multihost.comm import (
    CanaryMessage,
    RunMessage,
    ShutdownMessage,
    from_request_data,
)
from multihost.lockstep_runner import discard_outputs


def _default_runner_factory() -> Any:
    from config.settings import settings
    from tt_model_runners.runner_fabric import get_device_runner

    # Same id rank 0's scheduler gives its single worker, so logs line up.
    worker_id = settings.device_ids.split("),(")[0].strip("() ")
    return get_device_runner(worker_id)


def _exit_on_signal(signum, _frame) -> None:
    # mpirun forwards SIGTERM/SIGINT to every rank. Rank 0 may be stopped
    # mid-request and never send ShutdownMessage, so don't wait for it.
    TTLogger().info(f"Follower rank: received signal {signum}, exiting")
    os._exit(0)


def install_exit_signal_handlers() -> None:
    """Exit on SIGTERM/SIGINT even while the follower thread is blocked in MPI.

    Must be called from the main thread. The handler runs there, so it fires
    regardless of what the follower thread is waiting on.
    """
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)


class MultiHostLockstepFollower:
    """Opens this rank's part of the mesh and mirrors rank 0's runner calls.

    Same runner, same mesh-opening path and same warmup as rank 0's device
    worker, followed by a loop over rank 0's messages. Errors that every rank
    hits identically (e.g. request validation) are logged and the loop goes
    on, as rank 0 reports them to the client. Anything else is fatal: the
    method returns non-zero and the process exits, which makes tt-run stop the
    whole job instead of leaving ranks waiting on each other.
    """

    def __init__(
        self,
        rank: int,
        runner_factory: Callable[[], Any] = _default_runner_factory,
        comm: Any = None,
    ):
        self.rank = rank
        self._runner_factory = runner_factory
        # Anything with receive/barrier; the multihost.comm module unless
        # a test passes a fake.
        self._comm = comm
        self._logger = TTLogger()

    def run(self) -> int:
        """Run until rank 0 says shut down (returns 0) or something fails (returns 1)."""
        try:
            runner = self._runner_factory()
            # Only rank 0 has frames worth encoding.
            if hasattr(runner, "export_in_runner"):
                runner.export_in_runner = False

            self._logger.info(f"Rank {self.rank}: opening mesh")
            runner.set_device()
            comm = self._comm
            if comm is None:
                from multihost import comm

            self._logger.info(f"Rank {self.rank}: warming up")
            if asyncio.run(runner.warmup()) is False:
                raise RuntimeError("warmup reported not-ready")
            self._logger.info(f"Rank {self.rank}: warm, waiting for all ranks")
            comm.barrier()
            self._logger.info(f"Rank {self.rank}: following rank 0")

            self._loop(runner, comm)
        except Exception as e:
            self._logger.error(
                f"Rank {self.rank}: fatal error: {e}\n{traceback.format_exc()}"
            )
            return 1

        self._logger.info(f"Rank {self.rank}: shutdown from rank 0")
        try:
            runner.close_device()
        except Exception as e:
            self._logger.warning(f"Rank {self.rank}: close_device failed: {e}")
        return 0

    def _loop(self, runner: Any, comm: Any) -> None:
        while True:
            message = comm.receive()
            if isinstance(message, ShutdownMessage):
                return
            if isinstance(message, RunMessage):
                requests = [from_request_data(data) for data in message.requests]
                self._run(runner, requests, label=_task_ids(requests))
            elif isinstance(message, CanaryMessage):
                if message.deep:
                    self._run(
                        runner,
                        [runner._build_warmup_video_request()],
                        label="deep canary",
                    )
                else:
                    comm.barrier()
            else:
                raise TypeError(f"unknown message from rank 0: {message!r}")

    def _run(self, runner: Any, requests: list[Any], label: str) -> None:
        try:
            discard_outputs(runner.run(requests))
        except Exception as e:
            self._logger.error(
                f"Rank {self.rank}: {label} failed: {e}\n{traceback.format_exc()}"
            )


def _task_ids(requests: list[Any]) -> str:
    return ", ".join(r._task_id for r in requests)
