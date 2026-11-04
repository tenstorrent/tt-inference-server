# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Serve one mesh that spans several hosts (e.g. a 4x32 quad) from a single ``tt-run`` launch.

Every rank runs the normal ``uvicorn main:app`` command:

- Rank 0 runs the server unchanged. Its device worker opens the full mesh and
  wraps the runner in :class:`multihost.lockstep_runner.MultiHostLockstepRunner`,
  which sends each call to the other ranks before running it locally.
- Ranks 1..N-1 run :class:`multihost.follower.MultiHostLockstepFollower` from
  ``main.py``'s lifespan and never open the HTTP port.

This module is imported by the uvicorn process, so it must not import ttnn:
MPI is started only by the process that opens the mesh.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

# Set by mpirun for every rank it launches.
_MPI_RANK_ENV = "OMPI_COMM_WORLD_RANK"
_MPI_SIZE_ENV = "OMPI_COMM_WORLD_SIZE"
_LAUNCHER_POLL_S = 2.0


def launch_rank() -> int | None:
    """This process's MPI rank, read from the environment mpirun sets.

    ``ttnn.distributed_context_get_rank()`` can't be used here: it needs MPI to
    be started in this process, and on rank 0 only the forked device worker
    starts MPI (when it opens the mesh).
    """
    value = os.environ.get(_MPI_RANK_ENV)
    return int(value) if value is not None else None


def is_follower_rank() -> bool:
    rank = launch_rank()
    return rank is not None and rank != 0


def is_multihost_launch() -> bool:
    """True when mpirun launched more than one rank (a mesh spanning several hosts)."""
    return int(os.environ.get(_MPI_SIZE_ENV, "1")) > 1


async def stop_when_launcher_exits() -> None:
    """Rank 0: shut the server down when mpirun goes away.

    On Ctrl-C, tt-run SIGKILLs mpirun, which then can't signal its local rank.
    The remote ranks are torn down by their daemons, but rank 0 would keep
    serving with a dead mesh. Its parent is mpirun, so a changed parent PID
    means mpirun is gone; SIGTERM runs the normal uvicorn shutdown.
    """
    launcher_pid = os.getppid()
    while os.getppid() == launcher_pid:
        await asyncio.sleep(_LAUNCHER_POLL_S)
    logging.getLogger(__name__).error(
        "mpirun (pid %d) exited; shutting down rank 0", launcher_pid
    )
    os.kill(os.getpid(), signal.SIGTERM)
