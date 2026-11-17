# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""A FLUX server's uvicorn process must not import ttnn or tt-metal's `models`
package before it forks its device worker.

Loading ttnn starts Tracy's profiler thread. A device worker forked after
that inherits Tracy's queues without the thread that drains them, and grows
by every message it records (about 10 MB per job on the quad's rank 0).

FLUX only: the Wan and MiniMax-H3 request modules import tt-metal's policy
modules, which load ttnn, so those servers still do.

The server starts in a fresh interpreter, because conftest mocks ttnn in this
one. A finder there fails any import of the blocked packages, so the test
holds whether or not tt-metal is installed. It runs `main`'s lifespan up to
the worker's `Process.start()`, which exits the interpreter instead of
forking.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
FORKED = 17

_START_UP_TO_THE_FORK = f"""
import asyncio
import multiprocessing
import os
import sys

BLOCKED = ("ttnn", "tracy", "models")


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"the server process imported {{name}}")
        return None


sys.meta_path.insert(0, Block())
import main  # noqa: E402
import model_services.scheduler as scheduler  # noqa: E402


class Fork:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        loaded = [m for m in sys.modules if m.split(".")[0] in BLOCKED]
        print("loaded before the fork:", loaded, flush=True)
        # the service's Manager process holds this interpreter's output pipes
        for child in multiprocessing.active_children():
            child.kill()
        os._exit({FORKED} if not loaded else 1)


scheduler.Process = Fork


async def start():
    async with main.lifespan(main.app):
        await asyncio.sleep(120)


asyncio.run(start())
"""

# FLUX-only coverage: see the module docstring.
_FLUX_SERVERS = {
    "flux-3-image-klein on the quad": {
        "MODEL": "flux-3-image-klein",
        "DEVICE": "galaxy_quad",
        "MODEL_RUNNER": "tt-flux-3-image-klein",
    },
}


@pytest.mark.parametrize(
    "served", list(_FLUX_SERVERS.values()), ids=list(_FLUX_SERVERS)
)
def test_flux_server_forks_its_worker_without_tt_metal(served):
    env = {**os.environ, **served, "PYTHONPATH": str(PROJECT_DIR)}
    result = subprocess.run(
        [sys.executable, "-c", _START_UP_TO_THE_FORK],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == FORKED, (result.stdout + result.stderr)[-4000:]
