"""Refuse a CI agentic run before any server/device work when host disk is low.

Terminal-Bench builds every task image locally; a runner whose Docker/root
filesystem fills mid-suite loses the whole job (seen twice on 120-qb2-p05t01).
"""
import os
import shutil
import socket
import subprocess
import sys

MIN_FREE_BYTES = int(os.environ.get("GEMMA_MIN_HOST_FREE_BYTES", str(400 * 10**9)))


def check(workflow):
    if workflow != "agentic" or os.environ.get("GITHUB_ACTIONS") != "true":
        return
    paths = {"/": "/"}
    try:
        root = subprocess.run(["docker", "info", "-f", "{{.DockerRootDir}}"], capture_output=True, text=True,
                              timeout=60).stdout.strip()
        if root:
            paths["docker"] = root
    except Exception:
        pass
    free = {name: shutil.disk_usage(path).free for name, path in paths.items() if os.path.exists(path)}
    low = {name: value for name, value in free.items() if value < MIN_FREE_BYTES}
    line = f"host={socket.gethostname()} min_free={MIN_FREE_BYTES} free={free}"
    if low:
        print(f"GEMMA_HOST_DISK_REJECT {line}", flush=True)
        sys.exit(77)
    print(f"GEMMA_HOST_DISK_ACCEPT {line}", flush=True)
