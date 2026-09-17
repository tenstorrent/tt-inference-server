# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import logging
import os
import sys
import time

import pytest

from workflow_module.proc import run_command


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        run_command(
            [sys.executable, "-c", "pass"], logging.getLogger(), timeout_seconds=value
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("file_logging", [False, True])
def test_deadline_preserves_partial_output_and_kills_children(tmp_path, file_logging):
    marker = tmp_path / "escaped-child"
    partial = tmp_path / "partial-result"
    child = f"import time,pathlib; time.sleep(2); pathlib.Path({str(marker)!r}).touch()"
    parent = (
        "import subprocess,sys,time,pathlib; "
        f"pathlib.Path({str(partial)!r}).write_text('partial'); "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "print('started',flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    rc = run_command(
        [sys.executable, "-c", parent],
        logging.getLogger(),
        timeout_seconds=0.5,
        log_file_path=str(tmp_path / "log") if file_logging else None,
    )
    assert rc == 124
    assert time.monotonic() - started < 2
    assert partial.read_text() == "partial"
    time.sleep(2)
    assert not marker.exists()


def test_leader_exit_does_not_leave_log_pipe_wait_unbounded():
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
    assert (
        run_command(
            [sys.executable, "-c", code], logging.getLogger(), timeout_seconds=0.3
        )
        == 124
    )


def test_success_and_checked_timeout():
    assert (
        run_command(
            [sys.executable, "-c", "pass"], logging.getLogger(), timeout_seconds=2
        )
        == 0
    )
    with pytest.raises(RuntimeError, match="124"):
        run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            logging.getLogger(),
            timeout_seconds=0.1,
            check=True,
        )
