# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Ranks 1..N-1 of a multi-host mesh: run whatever rank 0 runs, discard the output."""

from __future__ import annotations

import asyncio
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


def _log_stop_signal(signum, _frame) -> None:
    # A follower stops only on rank 0's ShutdownMessage, after rank 0's last
    # request: a rank that leaves part way through a render leaves the mesh
    # unusable until it is reset. Rank 0 alone gets the stop (serve_quad.sh
    # stop); a signal here comes from the launcher, which (prterun on SIGTERM,
    # tt-run on Ctrl-C) kills every rank itself, so there is nothing to wait for.
    TTLogger().info(
        f"Follower rank: received signal {signum}; stopping only on rank 0's shutdown"
    )


def install_exit_signal_handlers() -> None:
    """Log SIGTERM/SIGINT instead of letting uvicorn stop this rank mid-render.

    Must be called from the main thread.
    """
    signal.signal(signal.SIGTERM, _log_stop_signal)
    signal.signal(signal.SIGINT, _log_stop_signal)


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
        # Every request rank 0 sends is run: rank 0 runs it too, and waits for
        # this rank in its collectives.
        try:
            discard_outputs(runner.run(requests))
        except ValueError as e:
            self._logger.error(f"Rank {self.rank}: {label} rejected: {e}")


def _task_ids(requests: list[Any]) -> str:
    return ", ".join(r._task_id for r in requests)
