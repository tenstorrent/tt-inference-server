# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from llm_module.agentic import owned_process


@pytest.fixture
def marker():
    base = "/tmp/harbor-owned-" + uuid.uuid4().hex
    yield base
    for suffix in (".pid", ".pid.new", ".cancel"):
        Path(base + suffix).unlink(missing_ok=True)


def wait_record(base):
    for _ in range(200):
        if Path(base + ".pid").exists():
            return json.loads(Path(base + ".pid").read_text())
        time.sleep(0.01)
    raise AssertionError("Owned launcher did not publish its identity")


def test_cancellation_stops_detached_child_and_not_unrelated_peer(marker, tmp_path):
    target, ready = tmp_path / "late-write", tmp_path / "ready"
    child = "import time; from pathlib import Path; time.sleep(1); Path({!r}).touch()".format(
        str(target)
    )
    program = (
        "import subprocess,sys,time; from pathlib import Path; "
        "subprocess.Popen([sys.executable,'-c', {!r}], start_new_session=True); "
        "Path({!r}).touch(); time.sleep(30)"
    ).format(child, str(ready))
    root = subprocess.Popen(
        [
            sys.executable,
            owned_process.__file__,
            "launch",
            marker,
            sys.executable,
            "-c",
            program,
        ]
    )
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        wait_record(marker)
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        report = owned_process.cleanup(marker)
        assert report == {
            "status": "stopped",
            "owned_processes": 2,
            "live_remaining": 0,
        }
        root.wait(timeout=2)
        time.sleep(1.1)
        assert not target.exists()
        assert peer.poll() is None
    finally:
        for process in (root, peer):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


def test_late_launch_is_rejected_after_cleanup(marker, tmp_path):
    target = tmp_path / "must-not-exist"
    report = owned_process.cleanup(marker)
    assert report["status"] == "launch_cancelled"
    result = subprocess.run(
        [
            sys.executable,
            owned_process.__file__,
            "launch",
            marker,
            sys.executable,
            "-c",
            "open({!r},'w').close()".format(str(target)),
        ]
    )
    assert result.returncode == 125
    assert not target.exists()


def test_reused_pid_identity_is_not_signaled(marker):
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        record = owned_process.identity(peer.pid)
        record["start"] += 1
        Path(marker + ".pid").write_text(json.dumps(record))
        assert owned_process.cleanup(marker)["status"] == "root_already_exited"
        assert peer.poll() is None
    finally:
        peer.kill()
        peer.wait(timeout=2)


def test_cli_rejects_unscoped_marker_path():
    result = subprocess.run(
        [sys.executable, owned_process.__file__, "cleanup", "/tmp"], capture_output=True
    )
    assert result.returncode != 0
    assert b"unscoped" in result.stderr
