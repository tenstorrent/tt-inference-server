# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Runner state for the driver's poll: DONE/FAILED marker, else RUNNING if the
# runner pid is alive (not a zombie), DIED if not, ABSENT if never launched.
import glob
import os
import shutil
import subprocess

W = "/content/gpuref"


def read(name):
    path = os.path.join(W, name)
    return open(path).read().strip() if os.path.exists(path) else ""


pid = read("runner.pid")
stat = f"/proc/{pid}/stat"
alive = (
    bool(pid)
    and os.path.exists(stat)
    and open(stat).read().rsplit(")", 1)[1].split()[0] != "Z"
)
if os.path.exists(os.path.join(W, "DONE")):
    state = "DONE"
elif os.path.exists(os.path.join(W, "FAILED")):
    state = "FAILED"
else:
    state = "RUNNING" if alive else ("DIED" if pid else "ABSENT")
print("GPUREF_STATE=" + state)
print("GPUREF_PHASE=" + read("phase"))
query = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader",
]
if shutil.which("nvidia-smi"):
    print(
        "GPUREF_GPU="
        + subprocess.run(query, capture_output=True, text=True).stdout.strip()
    )
for line in read("runner.log").splitlines()[-12:]:
    print("GPUREF_LOG| " + line)
logs = sorted(
    glob.glob(os.path.join(W, "results", "*", "run_py.log")), key=os.path.getmtime
)
if logs:  # latest progress line of the model in evals (tqdm writes \r)
    tail = (
        open(logs[-1], "rb")
        .read()[-4096:]
        .decode("utf-8", "replace")
        .replace("\r", "\n")
        .splitlines()
    )
    if tail:
        print(
            "GPUREF_EVAL="
            + os.path.basename(os.path.dirname(logs[-1]))
            + ": "
            + tail[-1][-200:]
        )
