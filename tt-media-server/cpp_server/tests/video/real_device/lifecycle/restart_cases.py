#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Start, stop and restart behaviour of the video service on a real device.

Report: docs/tests/2026-11-14-video-real-device-extended-checks.md

Unlike the other real-device tests this one starts and stops the server itself,
several times, so no other server may use the chips. It needs a T3K with the
Wan2.2 weights; the environment is taken from the arguments (see the README of
tests/video/real_device for what each variable does). Example, from cpp_server:

    python3 tests/video/real_device/lifecycle/restart_cases.py \
        --tt-metal <tt-metal> \
        --dit-cache <tt_dit weight cache> --api-key <token>

Phases (each is a separate server start):
  1. purge on start: planted files in TT_VIDEO_OUTPUT_DIR (only top-level files named like
     a result, <uuid>.mp4 and <uuid>_silent.mp4, go; other mp4 files stay), a job, DELETE removes its mp4, seed 7 video kept for phase 3
  2. SIGTERM while idle: clean exit; the next start works WITHOUT `tt-smi -r`;
     TT_VIDEO_PURGE_ON_START=0 keeps the planted files; seed 7 again (same video as in
     phase 1 after a restart?)
  3. SIGTERM in the middle of a job: exit time, no worker left, no ipc folder; whether the
     next start works without `tt-smi -r` (it is recorded, then the chips are reset)
  4. an output folder the server cannot write to: how the failure shows
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

GEN = "/v1/videos/generations"
# Names the server gives its results (<uuid>.mp4, <uuid>_silent.mp4): only these are purged.
OLD1 = "11111111-1111-4111-8111-111111111111.mp4"
OLD2 = "22222222-2222-4222-8222-222222222222.mp4"
OLD3 = "33333333-3333-4333-8333-333333333333_silent.mp4"
KEEP = "44444444-4444-4444-8444-444444444444.mp4"
LOCK = ".tt_video_server.lock"
OTHERS = ("link.mp4", "holiday.mp4")
results = []
T0 = time.time()


def p(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def note(ok, name, detail=""):
    results.append(ok)
    p("PASS" if ok else "FAIL", name, detail)


class Server:
    def __init__(self, args, outdir, log, extra_env=None):
        self.args = args
        self.outdir = outdir
        self.log = log
        self.extra_env = extra_env or {}
        self.proc = None
        self.base = f"http://127.0.0.1:{args.port}"

    def start(self):
        a = self.args
        env = dict(
            os.environ,
            MODEL_SERVICE="video",
            MODEL="Wan2.2-T2V-A14B-Diffusers",
            MODEL_RUNNER="tt-wan2.2",
            DEVICE="t3k",
            DEVICE_IDS="(0,1,2,3)",
            TT_DIT_CACHE_DIR=a.dit_cache,
            TT_METAL_HOME=a.tt_metal,
            HF_HUB_OFFLINE="1",
            OPENAI_API_KEY=a.api_key,
            TT_VIDEO_OUTPUT_DIR=str(self.outdir),
        )
        # The runner only needs the python/cpp_dit_adapters package; tt-media-server is not needed.
        # TT_VIDEO_TRACED is taken from the caller's environment (0 on a T3K).
        env.pop("TT_PYTHON_PATH", None)
        env.update(self.extra_env)
        cmd = f"source '{a.tt_metal}/python_env/bin/activate' && exec ./build/tt_media_server_cpp -p {a.port}"
        self.logf = open(self.log, "wb")
        self.t_start = time.time()
        self.proc = subprocess.Popen(
            ["bash", "-c", cmd],
            env=env,
            stdout=self.logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def wait_ready(self, timeout=900):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                return False
            st, j = self.call("GET", "/tt-liveness", token=None)
            if st == 200 and j.get("model_ready") is True:
                self.ready_after = time.time() - self.t_start
                return True
            time.sleep(2)
        return False

    def call(self, method, path, body=None, token="default", raw=False, timeout=60):
        r = urllib.request.Request(
            self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
        )
        if body is not None:
            r.add_header("content-type", "application/json")
        tok = self.args.api_key if token == "default" else token
        if tok:
            r.add_header("authorization", f"Bearer {tok}")
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                data = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            data, status = e.read(), e.code
        except OSError:
            return None, {}
        if raw:
            return status, data
        try:
            return status, (json.loads(data) if data.strip() else {})
        except ValueError:
            return status, {}

    def job(self, jid):
        st, j = self.call("GET", f"{GEN}/{jid}")
        return j if st == 200 else {"status": f"http{st}"}

    def wait_job(self, jid, wanted, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if self.job(jid).get("status") in wanted:
                return True
            time.sleep(2)
        return False

    def submit(self, prompt="a red fox", **kw):
        st, j = self.call("POST", GEN, {"prompt": prompt, "num_inference_steps": 4, **kw})
        return st, j

    def log_text(self):
        return open(self.log, errors="replace").read()

    def terminate(self, sig=signal.SIGTERM, timeout=120):
        if self.proc is None or self.proc.poll() is not None:
            return self.proc.returncode if self.proc else None
        self.proc.send_signal(sig)
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill_all(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def descendants(pid):
    found = []
    for ch in children(pid):
        found.append(ch)
        found.extend(descendants(ch))
    return found


def server_pid(srv):
    """The tt_media_server process (bash -c ... exec replaces bash, so it is the child itself)."""
    return srv.proc.pid


def media_shm():
    """Shm queues of the service (the names of the defaults); they must not outlive the server."""
    return sorted(
        n
        for n in os.listdir("/dev/shm")
        if n.startswith(("tt_media_", "tt_warmup_", "tt_worker_metrics"))
    )


def shm_names(prefixes=("tt_", "TT_MEDIA", "media")):
    return sorted(
        n
        for n in os.listdir("/dev/shm")
        if not n.startswith("TT_UMD") and not n.startswith("sem.TT_UMD")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-metal", required=True)
    ap.add_argument(
        "--media-server",
        default="",
        help="unused, kept so older command lines still work (the worker no longer "
        "loads tt-media-server)",
    )
    ap.add_argument("--dit-cache", required=True)
    ap.add_argument("--api-key", default="your-secret-key")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workdir", default="", help="logs and downloads (default: temp dir)")
    args = ap.parse_args()

    work = args.workdir or tempfile.mkdtemp(prefix="tt-video-lifecycle-real.")
    os.makedirs(work, exist_ok=True)
    out = os.path.join(work, "videos")
    os.makedirs(out, exist_ok=True)
    p("workdir:", work)
    if (
        subprocess.run(
            ["curl", "-sf", f"http://127.0.0.1:{args.port}/tt-liveness"], capture_output=True
        ).returncode
        == 0
    ):
        print("a server already answers on this port; stop it first (this test owns the chips)")
        return 2

    # ---------------------------------------------------------------- phase 1: purge, delete, seed
    p("=== phase 1: purge on start, DELETE, seed 7")
    planted = {
        OLD1: b"orphan 1",
        OLD2: b"orphan 2",
        OLD3: b"silent",
        "holiday.mp4": b"the user's own clip, not named like a result",
        "audio.wav": b"wav",
        "notes.txt": b"txt",
    }
    for name, data in planted.items():
        open(os.path.join(out, name), "wb").write(data)
    os.makedirs(os.path.join(out, "sub"), exist_ok=True)
    open(os.path.join(out, "sub", "inner.mp4"), "wb").write(b"inner")
    os.symlink(os.path.join(out, "notes.txt"), os.path.join(out, "link.mp4"))
    srv = Server(args, out, os.path.join(work, "phase1.log")).start()
    if not srv.wait_ready():
        note(False, "phase 1: server ready", srv.log_text()[-400:])
        srv.kill_all()
        return 1
    note(True, "phase 1: server ready", f"after {srv.ready_after:.0f}s (weights from the cache)")
    left = sorted(os.listdir(out))
    p("  output folder after the start:", left)
    note(
        left == sorted(["audio.wav", "notes.txt", "sub", "link.mp4", "holiday.mp4", LOCK]),
        "purge removed the server's own top-level results only (kept: .wav, .txt, subfolder, "
        "symlink, holiday.mp4, and the lock file the server created)",
        str(left),
    )
    note(
        "removed 3 orphan mp4" in srv.log_text(),
        "the log reports the purge",
        "removed 3 orphan mp4",
    )
    note("tt_dit weight cache" in srv.log_text(), "the log names the tt_dit weight cache folder")
    st, j = srv.submit(seed=7)
    note(st == 202, "job with seed 7 accepted")
    note(
        srv.wait_job(j["id"], ("completed", "failed"), 600)
        and srv.job(j["id"]).get("status") == "completed",
        "it completes",
    )
    st, v1 = srv.call("GET", f"{GEN}/{j['id']}/download", raw=True)
    open(os.path.join(work, "seed7_run1.mp4"), "wb").write(v1)
    files = [f for f in os.listdir(out) if f.endswith(".mp4") and f not in OTHERS]
    note(len(files) == 1, "the finished video is one mp4 in the output folder", str(files))
    st, _ = srv.call("DELETE", f"{GEN}/{j['id']}")
    after = [f for f in os.listdir(out) if f.endswith(".mp4") and f not in OTHERS]
    note(st == 200 and after == [], "DELETE removes the job and its mp4", f"http{st} left={after}")
    st, j2 = srv.call("GET", f"{GEN}/{j['id']}")
    note(st == 404, "the deleted job is gone (404)", f"http{st}")

    # ---------------------------------------------------------------- phase 2: idle SIGTERM, then PURGE=0
    p("=== phase 2: SIGTERM while idle, then start without tt-smi -r and TT_VIDEO_PURGE_ON_START=0")
    pid = server_pid(srv)
    desc = descendants(pid)
    ipc_mine = f"tt-video-ipc-{pid}"
    # One DEVICE_IDS group runs the runner on a thread of the server (no worker process,
    # no ipc folder); only several groups give worker processes.
    in_process = "Creating in-process video service" in srv.log_text()
    p("  execution mode:", "in-process (no worker, no ipc folder)" if in_process else "worker process")
    if in_process:
        note(ipc_mine not in os.listdir("/tmp"), "in-process mode: no ipc folder is created", ipc_mine)
    else:
        note(ipc_mine in os.listdir("/tmp"), "the ipc folder exists while the server runs", ipc_mine)
    shm_run = shm_names()
    t0 = time.time()
    code = srv.terminate(signal.SIGTERM, 120)
    note(
        code == 0,
        "the idle server exits with code 0 after SIGTERM",
        f"code={code} after {time.time() - t0:.1f}s",
    )
    time.sleep(2)
    alive = [
        d
        for d in desc
        if os.path.exists(f"/proc/{d}") and open(f"/proc/{d}/stat").read().split()[2] != "Z"
    ]
    note(not alive, "no server or worker process is left", f"alive={alive}")
    note(ipc_mine not in os.listdir("/tmp"), "the ipc folder is removed")
    note("graceful" in srv.log_text(), "the log says the shutdown was graceful")
    note(not media_shm(), "the shm queues of the service are removed", str(media_shm()))
    keep = os.path.join(out, KEEP)
    open(keep, "wb").write(b"keep")
    srv = Server(
        args, out, os.path.join(work, "phase2.log"), {"TT_VIDEO_PURGE_ON_START": "0"}
    ).start()
    ok = srv.wait_ready()
    note(
        ok,
        "the server starts after an idle SIGTERM without resetting the chips",
        f"after {getattr(srv, 'ready_after', 0):.0f}s" if ok else srv.log_text()[-500:],
    )
    if not ok:
        srv.kill_all()
        return finish()
    note(os.path.exists(keep), "TT_VIDEO_PURGE_ON_START=0 keeps a result-named mp4")
    st, j = srv.submit(seed=7)
    srv.wait_job(j["id"], ("completed", "failed"), 600)
    st, v2 = srv.call("GET", f"{GEN}/{j['id']}/download", raw=True)
    open(os.path.join(work, "seed7_run2.mp4"), "wb").write(v2)
    p(f"  seed 7: run 1 {len(v1)} B sha {sha(v1)}, run 2 {len(v2)} B sha {sha(v2)}")
    p("  identical bytes across a restart:", v1 == v2)
    if v1 != v2 and shutil.which("ffmpeg"):
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "info",
                "-i",
                os.path.join(work, "seed7_run1.mp4"),
                "-i",
                os.path.join(work, "seed7_run2.mp4"),
                "-lavfi",
                "psnr",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        p("  psnr:", [ln for ln in (r.stderr + r.stdout).splitlines() if "PSNR" in ln][-1:])
    note(srv.job(j["id"]).get("status") == "completed", "seed 7 completes again")

    # ---------------------------------------------------------------- phase 3: SIGTERM mid job
    p("=== phase 3: SIGTERM in the middle of a job")
    pid = server_pid(srv)
    desc = descendants(pid)
    ipc_mine = f"tt-video-ipc-{pid}"
    st, run_job = srv.submit("job killed by SIGTERM", seed=11)
    srv.call("POST", GEN, {"prompt": "queued behind", "num_inference_steps": 4})
    srv.wait_job(run_job["id"], ("in_progress",), 60)
    time.sleep(10)
    t0 = time.time()
    code = srv.terminate(signal.SIGTERM, 180)
    took = time.time() - t0
    note(
        code is not None,
        "the server exits after SIGTERM with a job running",
        f"code={code} after {took:.1f}s",
    )
    p(
        "  the job was in step 1-2 of 4; the server did not wait for it"
        if took < 10
        else "  the server waited for the job"
    )
    time.sleep(2)
    alive = [
        d
        for d in desc
        if os.path.exists(f"/proc/{d}") and open(f"/proc/{d}/stat").read().split()[2] != "Z"
    ]
    note(not alive, "no server or worker process is left", f"alive={alive}")
    note(ipc_mine not in os.listdir("/tmp"), "the ipc folder is removed")
    note(not media_shm(), "the shm queues of the service are removed", str(media_shm()))
    p("  mp4 files in the output folder:", [f for f in os.listdir(out) if f.endswith(".mp4")])
    srv = Server(
        args,
        out,
        os.path.join(work, "phase3_restart.log"),
        {"TT_VIDEO_PURGE_ON_START": "0", "TT_VIDEO_WORKER_MAX_RESTARTS": "0"},
    ).start()
    ok = srv.wait_ready(300)
    p(
        "  OBSERVED: the start right after a SIGTERM in the middle of a job",
        "works" if ok else "FAILS (the chips need tt-smi -r)",
    )
    note(
        True,
        "start after a mid-job SIGTERM without a chip reset is recorded",
        "works" if ok else "fails: " + ("ETH" if "ETH" in srv.log_text() else "other"),
    )
    if ok:
        srv.terminate()
    else:
        srv.kill_all()
        p("  running tt-smi -r")
        subprocess.run(["tt-smi", "-r"], capture_output=True, timeout=300)
        time.sleep(5)

    # ---------------------------------------------------------------- phase 4: unwritable output folder
    p("=== phase 4: an output folder the server cannot write to")
    ro = os.path.join(work, "readonly_videos")
    os.makedirs(ro, exist_ok=True)
    os.chmod(ro, 0o555)
    srv = Server(args, ro, os.path.join(work, "phase4.log")).start()
    ok = srv.wait_ready()
    note(
        ok,
        "the server starts with an unwritable output folder",
        f"after {getattr(srv, 'ready_after', 0):.0f}s" if ok else srv.log_text()[-500:],
    )
    if ok:
        # The mp4 is written only after the whole generation, so a request
        # must be refused up front (503) instead of failing after the run.
        st, body = srv.submit()
        note(st == 503, "a request is refused with 503, no generation is started", f"http{st}")
        note(
            "not writable" in json.dumps(body),
            "the answer names the output directory problem",
            json.dumps(body)[:160],
        )
        note(
            srv.call("GET", "/v1/videos/jobs")[1] == [],
            "no job record was created",
        )
        note(
            "Video output directory is not usable" in srv.log_text(),
            "the startup log names the problem",
        )
        note(
            srv.call("GET", "/tt-liveness", token=None)[1].get("status") == "alive",
            "the server is alive",
        )
        # Fixing the folder is enough: no restart.
        os.chmod(ro, 0o755)
        st, j = srv.submit("after the fix")
        note(st == 202, "after the folder is fixed a request is accepted again", f"http{st}")
        if st == 202:
            srv.wait_job(j["id"], ("completed", "failed"), 600)
            body = srv.job(j["id"])
            note(
                body.get("status") == "completed",
                "that job completes",
                str(body.get("error")),
            )
    srv.terminate()
    os.chmod(ro, 0o755)
    return finish()


def finish():
    failed = results.count(False)
    print(f"\n{len(results) - failed} PASS, {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
