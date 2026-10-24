# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The drivers' run_command owns the tool's whole process tree: a timeout or a
normal exit leaves no worker process behind (aiperf runs its workers as
separate processes; tt-shield run 39659097908 kept one sending requests for
hours after the per-run timeout, and the job never exited)."""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from llm_module.drivers._subprocess import run_command

REPO_ROOT = Path(__file__).resolve().parents[2]

# A stand-in for aiperf: starts a worker that inherits stdout and outlives it,
# writes the worker's pid, then runs for argv[2] seconds.
TOOL = textwrap.dedent(
    """
    import subprocess, sys, time
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(sys.argv[1], "w") as f:
        f.write(str(worker.pid))
    time.sleep(float(sys.argv[2]))
    """
)


def _tool(tmp_path, runtime_s):
    script = tmp_path / "tool.py"
    script.write_text(TOOL)
    pid_file = tmp_path / "worker.pid"
    return [sys.executable, str(script), str(pid_file), str(runtime_s)], pid_file


def _gone(pid, within_s=5.0):
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_returns_124_and_kills_the_tools_workers(tmp_path):
    cmd, pid_file = _tool(tmp_path, runtime_s=60)
    start = time.monotonic()
    assert run_command(cmd, timeout_s=1) == 124
    assert time.monotonic() - start < 10
    assert _gone(int(pid_file.read_text()))


def test_a_worker_left_behind_by_a_normal_exit_is_killed(tmp_path):
    cmd, pid_file = _tool(tmp_path, runtime_s=0)
    assert run_command(cmd, timeout_s=30) == 0
    assert _gone(int(pid_file.read_text()))


def test_the_callers_output_pipe_closes_after_a_timeout(tmp_path):
    # The CI hang: the caller reads the driver's output until EOF, which never came
    # while a worker still held the pipe.
    cmd, _ = _tool(tmp_path, runtime_s=60)
    caller = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from llm_module.drivers._subprocess import run_command; "
            f"sys.exit(run_command({cmd!r}, timeout_s=1))",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    caller.communicate(timeout=20)
    assert caller.returncode == 124
