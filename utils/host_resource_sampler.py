"""Bounded read-only host observations for the explicit client24 experiment."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path


class HostResourceSampler:
    def __init__(self, output):
        self.output = Path(output)
        self.stop = threading.Event()
        self.thread = None
        self.paths = [str(Path.cwd()), "/", "/var/lib/docker", "/mnt/dockercache"]

    def sample(self):
        row = {"schema": 1, "wall_s": time.time(), "pid": os.getpid(), "files": {}, "filesystems": {}}
        for name in ("meminfo", "loadavg", "pressure/cpu", "pressure/memory", "pressure/io", "sys/fs/file-nr"):
            try:
                row["files"][name] = (Path("/proc") / name).read_text()[:4096]
            except OSError as exc:
                row["files"][name] = {"error": str(exc)}
        for name in dict.fromkeys(self.paths):
            try:
                stat = os.statvfs(name)
                row["filesystems"][name] = {"available_bytes": stat.f_bavail * stat.f_frsize,
                    "total_bytes": stat.f_blocks * stat.f_frsize, "free_inodes": stat.f_favail}
            except OSError as exc:
                row["filesystems"][name] = {"error": str(exc)}
        return row

    def observe(self):
        while not self.stop.is_set():
            try:
                row = self.sample()
                body = json.dumps(row, separators=(",", ":"))
                print("GEMMA_HOST_RESOURCE " + body, flush=True)
                self.output.parent.mkdir(parents=True, exist_ok=True)
                with self.output.open("a") as handle:
                    handle.write(body + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception as exc:
                # Observation failure must not change task/server behavior.
                try:
                    print("GEMMA_HOST_RESOURCE_ERROR " + repr(exc), flush=True)
                except Exception:
                    pass
            self.stop.wait(60)

    def __enter__(self):
        try:
            result = subprocess.run(["docker", "info", "--format", "{{.DockerRootDir}}"],
                                    capture_output=True, text=True, timeout=2, check=True)
            path = result.stdout.strip()
            if path.startswith("/") and "\n" not in path:
                self.paths.append(path)
        except Exception:
            pass
        try:
            self.thread = threading.Thread(target=self.observe, name="host-resource-observer", daemon=True)
            self.thread.start()
        except Exception:
            self.thread = None
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        return False
