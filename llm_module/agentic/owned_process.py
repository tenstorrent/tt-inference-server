# SPDX-License-Identifier: Apache-2.0
"""Linux-only cancellation helper for an explicitly owned task command.

This file is uploaded to the task container and uses only Python's standard
library. PID start times prevent signaling a reused PID. It is not a sandbox
against deliberate daemonization or an untrusted process rewriting its marker.
"""

import json
import os
from pathlib import Path
import re
import signal
import sys
import time


def identity(pid):
    try:
        fields = Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "parent": int(fields[1]), "start": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def same_process(record):
    actual = identity(record["pid"])
    return actual is not None and actual["start"] == record["start"] and actual["state"] != "Z"


def send(record, sig):
    if same_process(record):
        try:
            os.kill(record["pid"], sig)
        except ProcessLookupError:
            pass


def launch(base, command):
    cancel, marker = Path(base + ".cancel"), Path(base + ".pid")
    if cancel.exists():
        return 125
    if os.getpgrp() != os.getpid():
        os.setsid()
    record = identity(os.getpid())
    temporary = Path(base + ".pid.new")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(record, stream)
    os.replace(temporary, marker)
    # Cleanup writes this marker before looking for a PID, covering a command
    # whose Docker exec was accepted but has not started when cancellation hits.
    if cancel.exists():
        return 125
    os.execvp(command[0], command)


def cleanup(base):
    cancel, marker = Path(base + ".cancel"), Path(base + ".pid")
    cancel.touch(mode=0o600, exist_ok=True)
    deadline = time.monotonic() + 2
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not marker.exists():
        return {"status": "launch_cancelled", "owned_processes": 0, "live_remaining": 0}
    root = json.loads(marker.read_text())
    if not same_process(root):
        return {"status": "root_already_exited", "owned_processes": 0, "live_remaining": 0}
    owned = {root["pid"]: root}
    send(root, signal.SIGSTOP)
    # Freeze discovered parents before scanning again, so they cannot fork
    # between enumeration and termination. Detached-session children still have
    # their original parent and are included; unrelated container peers are not.
    for _ in range(32):
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            row = identity(int(entry.name))
            if row and row["parent"] in owned and row["pid"] not in owned:
                found.append(row)
        if not found:
            break
        for row in found:
            owned[row["pid"]] = row
            send(row, signal.SIGSTOP)
    else:
        raise RuntimeError("Owned process tree did not stabilize; cannot verify safely")
    for row in reversed(list(owned.values())):
        send(row, signal.SIGKILL)
    deadline = time.monotonic() + 2
    while any(same_process(row) for row in owned.values()) and time.monotonic() < deadline:
        time.sleep(0.02)
    remaining = sum(same_process(row) for row in owned.values())
    if remaining:
        raise RuntimeError("Owned processes remain live after cancellation")
    return {"status": "stopped", "owned_processes": len(owned), "live_remaining": 0}


def main():
    action, base = sys.argv[1:3]
    if not re.fullmatch(r"/tmp/harbor-owned-[0-9a-f]{32}", base):
        raise ValueError("Refusing an unscoped process marker path")
    if action == "launch":
        return launch(base, sys.argv[3:])
    if action != "cleanup":
        raise ValueError("Unsupported action")
    print(json.dumps(cleanup(base)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
