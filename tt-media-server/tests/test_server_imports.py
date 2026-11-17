# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The uvicorn process must not import ttnn or tt-metal's `models` package.

Loading ttnn starts Tracy's profiler thread. A device worker forked after
that inherits Tracy's queues without the thread that drains them, and grows
by every message it records (about 10 MB per job on the quad's rank 0).

`import main` runs in a fresh interpreter, because conftest mocks ttnn in
this one. A finder there fails any import of the blocked packages, so the
test holds whether or not tt-metal is installed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]

_IMPORT_MAIN = """
import sys

BLOCKED = ("ttnn", "tracy", "models")


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"the server process imported {name}")
        return None


sys.meta_path.insert(0, Block())
import main  # noqa: E402,F401
"""

_SERVED = {
    "flux-3-image-klein on the quad": {
        "MODEL": "flux-3-image-klein",
        "DEVICE": "galaxy_quad",
        "MODEL_RUNNER": "tt-flux-3-image-klein",
    },
}


@pytest.mark.parametrize("served", list(_SERVED.values()), ids=list(_SERVED))
def test_server_process_does_not_import_tt_metal(served):
    env = {**os.environ, **served, "PYTHONPATH": str(PROJECT_DIR)}
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_MAIN],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
