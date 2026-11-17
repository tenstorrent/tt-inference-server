# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Ranks 1..N-1 of a multi-host mesh: run whatever rank 0 runs, discard the output."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
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


# Held while this rank runs a request; set once a stop signal arrived.
_RUNNING = threading.Lock()
_STOPPING = threading.Event()
# How long after rank 0's own stop budget a stopped follower waits for
# ShutdownMessage before it decides rank 0 is gone.
_STOP_MARGIN_S = 60.0


def _stop_grace_seconds() -> float:
    from config.settings import settings

    return settings.worker_shutdown_timeout_seconds + _STOP_MARGIN_S


def _exit_on_signal(signum, _frame) -> None:
    # mpirun forwards SIGTERM/SIGINT to every rank. Rank 0 then finishes its
    # render, drops its queue and sends ShutdownMessage, so this rank keeps
    # following and closes its part of the mesh after rank 0's last request.
    # A rank that leaves in the middle of a request leaves the mesh unusable
    # until it is reset, so if rank 0 is gone and the message never comes,
    # it leaves once idle, after rank 0's stop budget.
    if _STOPPING.is_set():
        return
    _STOPPING.set()
    TTLogger().info(f"Follower rank: received signal {signum}, stopping with rank 0")
    timer = threading.Timer(_stop_grace_seconds(), _exit_when_idle)
    timer.daemon = True
    timer.start()


def _exit_when_idle() -> None:
    with _RUNNING:
        TTLogger().warning(
            "Follower rank: no shutdown from rank 0 after a stop signal, exiting"
        )
        os._exit(0)


def install_exit_signal_handlers() -> None:
    """Stop on SIGTERM/SIGINT even while the follower thread is blocked in MPI.

    Must be called from the main thread. The handler runs there, so it fires
    regardless of what the follower thread is waiting on.
    """
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)


class MultiHostLockstepFollower:
    """Opens this rank's part of the mesh and mirrors rank 0's runner calls.

    Same runner, same mesh-opening path and same warmup as rank 0's device
    worker, followed by a loop over rank 0's messages. A ``ValueError`` from
    the runner is a request every rank rejects identically before touching the
    mesh, so it is logged and the loop goes on; rank 0 reports it to the
    client. Anything else is fatal: the method returns non-zero and the process
    exits, which makes tt-run stop the whole job instead of leaving rank 0
    waiting in a collective this rank has abandoned.
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
        # Every request rank 0 sends is run, stopping or not: rank 0 runs it
        # too, and waits for this rank in its collectives.
        with _RUNNING:
            try:
                discard_outputs(runner.run(requests))
            except ValueError as e:
                self._logger.error(f"Rank {self.rank}: {label} rejected: {e}")


def _task_ids(requests: list[Any]) -> str:
    return ", ".join(r._task_id for r in requests)
