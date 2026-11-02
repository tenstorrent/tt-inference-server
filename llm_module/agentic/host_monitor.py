# SPDX-License-Identifier: Apache-2.0
"""Background monitor for Harbor agentic runs.

Every ``interval_s`` it appends one JSON line of host load to ``<job_dir>/host_resources.jsonl``
(CPU busy %, load average, available memory, free disk, top CPU processes), so a slow engine can be
told apart from a starved host. It also caps terminal recordings: an agent stuck in an output loop
once wrote a 185 GB ``agent/recording.cast``, which filled the runner disk and pushed most task
records out of the uploaded artifact. Recordings over ``max_bytes`` keep their last ``keep_bytes``
in ``recording.cast.tail`` and are truncated in place (the writer keeps appending).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + (v[4] if len(v) > 4 else 0)


def _top_processes(n=8):
    try:
        out = subprocess.run(["ps", "-eo", "pcpu,rss,comm", "--sort=-pcpu"], capture_output=True, text=True, timeout=10).stdout
        return [line.split(None, 2) for line in out.splitlines()[1:n + 1]]
    except Exception as e:  # noqa: BLE001 - diagnostics only
        return [["error", str(e)]]


def _container_cpu(n=10):
    """Per-container CPU and memory (docker stats), top n by CPU, so a busy task can be named."""
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                             capture_output=True, text=True, timeout=30).stdout
        rows = []
        for line in out.splitlines():
            name, cpu, mem = (line.split("\t") + ["", "", ""])[:3]
            try:
                rows.append([name, float(cpu.rstrip("%")), mem.split("/")[0].strip()])
            except ValueError:
                continue
        return sorted(rows, key=lambda r: -r[1])[:n]
    except Exception as e:  # noqa: BLE001 - diagnostics only
        return [["error", str(e)]]


def _host_image():
    out = subprocess.run(["docker", "ps", "--format", "{{.Image}}"], capture_output=True, text=True, timeout=30).stdout.split()
    if not out:
        raise RuntimeError("no running container image to reuse")
    return out[0]


def _cap_as_root(path: Path, keep_bytes: int) -> int:
    size = path.stat().st_size
    cmd = ["docker", "run", "--rm", "--user", "0", "--network", "none", "-v", f"{path.parent}:/a", "--entrypoint", "sh", _host_image(), "-c",
           f"tail -c {keep_bytes} /a/recording.cast > /a/recording.cast.tail && truncate -s 0 /a/recording.cast"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-300:])
    return size


def cap_recordings(job_dir: Path, max_bytes: int, keep_bytes: int) -> list:
    capped = []
    for path in job_dir.glob("*/agent/recording.cast"):
        try:
            size = path.stat().st_size
            if size <= max_bytes:
                continue
            with open(path, "rb") as src, open(path.with_name("recording.cast.tail"), "wb") as dst:
                src.seek(max(0, size - keep_bytes))
                dst.write(src.read(keep_bytes))
            os.truncate(path, 0)
            capped.append((str(path), size))
            logger.warning("HOST_MONITOR capped %s at %d bytes (kept last %d in recording.cast.tail)", path, size, keep_bytes)
        except PermissionError:
            # Written by the task container as root: cap it through docker as root, with an image
            # already on the host (the inference server's), so nothing is pulled.
            try:
                capped.append((str(path), _cap_as_root(path, keep_bytes)))
                logger.warning("HOST_MONITOR capped %s as root (kept last %d in recording.cast.tail)", path, keep_bytes)
            except Exception as e:  # noqa: BLE001
                logger.warning("HOST_MONITOR could not cap %s even as root: %s", path, e)
        except OSError as e:
            logger.warning("HOST_MONITOR could not cap %s: %s", path, e)
    return capped


def start(job_dir: Path, interval_s: float = 60.0, max_bytes: int = 1 << 30, keep_bytes: int = 256 << 20):
    """Start the monitor thread; returns a ``stop()`` callable that also does a final recording pass."""
    job_dir = Path(job_dir)
    stop_event = threading.Event()

    def loop():
        prev = _cpu_times()
        while not stop_event.wait(interval_s):
            try:
                job_dir.mkdir(parents=True, exist_ok=True)
                cur = _cpu_times(); total, idle = cur[0] - prev[0], cur[1] - prev[1]; prev = cur
                st = os.statvfs(str(job_dir))
                with open("/proc/meminfo") as f:
                    mem = {line.split(":")[0]: int(line.split()[1]) for line in f if line.startswith(("MemTotal", "MemAvailable"))}
                rec = dict(t=time.time(), cpu_busy_pct=round(100.0 * (1 - idle / total), 1) if total else None,
                           loadavg=os.getloadavg(), ncpu=os.cpu_count(), mem_avail_gb=round(mem.get("MemAvailable", 0) / 2**20, 1),
                           disk_free_gb=round(st.f_bavail * st.f_frsize / 2**30, 1), top=_top_processes(), containers=_container_cpu(),
                           capped=cap_recordings(job_dir, max_bytes, keep_bytes))
                with open(job_dir / "host_resources.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001 - never break the eval for diagnostics
                logger.warning("HOST_MONITOR sample failed: %s", e)

    thread = threading.Thread(target=loop, name="host-monitor", daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        thread.join(timeout=30)
        cap_recordings(job_dir, max_bytes, keep_bytes)

    return stop
