# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Start remote_runner.sh detached. The driver prepends "ARGS = [...]" (the
# runner's argv: pinned sha, work dir, vLLM pin, models). start_new_session=True
# is setsid(): no controlling terminal and its own process group, so the
# runner outlives this kernel call.
import os
import subprocess

W = "/content/gpuref"
for name in ("DONE", "FAILED", "phase"):
    if os.path.exists(os.path.join(W, name)):
        os.remove(os.path.join(W, name))
proc = subprocess.Popen(
    ["bash", os.path.join(W, "remote_runner.sh")] + ARGS,  # noqa: F821
    cwd=W,
    stdin=subprocess.DEVNULL,
    stdout=open(os.path.join(W, "runner.log"), "ab"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open(os.path.join(W, "runner.pid"), "w") as f:
    f.write(str(proc.pid))
print("GPUREF_LAUNCHED pid=%d" % proc.pid)
