#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The multihost code path of the video service on ONE T3K, under `tt-run`.

`run_tt_run_t3k.sh` starts the server as several MPI ranks that each see only their own
PCIe cards; tt-metal's mesh graph descriptor joins them into one 2x4 mesh. Rank 0 serves
HTTP and runs the model in lockstep with the follower ranks. This test starts and stops
that job itself, several times, so no other server may use the chips. It needs a T3K with
the Wan2.2 weights. The first start of a layout builds the per-rank weight cache from
PyTorch (tens of minutes); later starts load it (about two minutes).

Run, from cpp_server:

    python3 tests/video/real_device/multihost/multihost_cases.py \
        --tt-metal <tt-metal> --dit-cache <tt_dit weight cache> --api-key <token> \
        [--ranks 2] [--reference <mp4 of the same request from a single host>]

Phases (each from a fresh start unless said otherwise):
  1. start; every rank warms up; rank 0 runs the model in the server process; the same
     request twice gives the same bytes; the video is a real 832x480 mp4; with --reference,
     it is close to the single-host video (the bytes differ, see the report)
  2. two jobs queued behind each other both complete (the ranks stay in step)
  3. SIGTERM to rank 0 while idle: the follower is told to shut down and exits 0, every
     rank closes its mesh, nothing is left; the next start works without `tt-smi -r`
  4. SIGTERM to a follower while idle: mpirun ends the whole job, nothing is left
  5. kill -9 of a follower while idle: the same
  6. kill -9 of mpirun while idle: the ranks do not outlive it
  7. SIGTERM to rank 0 in the middle of a job: the job ends (no rank hangs); the next start
     needs `tt-smi -r`, which is run after checking that nothing holds the chips
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

GEN = "/v1/videos/generations"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(HERE, "run_tt_run_t3k.sh")
REQUEST = {"prompt": "a red ball rolling on grass", "seed": 7, "num_inference_steps": 4}
results = []
T0 = time.time()


def p(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def note(ok, name, detail=""):
    results.append(ok)
    p("PASS" if ok else "FAIL", name, detail)


def pids_of(name):
    out = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def children_of(pid, name):
    out = subprocess.run(
        ["pgrep", "-P", str(pid), "-x", name], capture_output=True, text=True
    ).stdout
    return [int(x) for x in out.split()]


def pid_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def port_listeners(port):
    hexport = f":{port:04X}"
    n = 0
    with open("/proc/net/tcp") as f:
        next(f)
        for line in f:
            fields = line.split()
            if fields[3] == "0A" and fields[1].endswith(hexport):
                n += 1
    return n


def chips_held():
    out = subprocess.run(
        "lsof /dev/tenstorrent/* 2>/dev/null",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout
    return [ln for ln in out.splitlines()[1:] if ln.strip()]


class Job:
    """One `tt-run` launch of the server (the launcher script execs tt-run)."""

    def __init__(self, args, workdir, tag):
        self.args = args
        self.dir = os.path.join(workdir, tag)
        os.makedirs(self.dir, exist_ok=True)
        self.log = os.path.join(self.dir, "job.log")
        self.base = f"http://127.0.0.1:{args.port}"
        self.proc = None
        self.mpirun = None
        self.rank_pids_at_ready = []

    def mpirun_pid(self):
        """mpirun is a child of the tt-run process the launcher script execs."""
        if self.mpirun is None and self.proc is not None:
            found = children_of(self.proc.pid, "mpirun")
            self.mpirun = found[0] if found else None
        return self.mpirun

    def rank_pids(self):
        """The rank processes of THIS launch: children of its mpirun (a server of another
        session, such as a mock-runner benchmark, may run on the machine too)."""
        m = self.mpirun_pid()
        return children_of(m, "tt_media_server") if m else []

    def start(self):
        a = self.args
        cmd = [
            RUN_SCRIPT,
            "--tt-metal",
            a.tt_metal,
            "--dit-cache",
            a.dit_cache,
            "--api-key",
            a.api_key,
            "--ranks",
            str(a.ranks),
            "--port",
            str(a.port),
            "--workdir",
            self.dir,
        ]
        if a.ranks == 4:
            cmd += ["--host-topology", a.host_topology, "--card-order", a.card_order]
        self.logf = open(self.log, "wb")
        self.t_start = time.time()
        self.proc = subprocess.Popen(
            cmd, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True
        )
        return self

    def log_text(self):
        t = open(self.log, errors="replace").read()
        return re.sub(r"\x1b\[[0-9;]*m", "", t)

    def call(self, method, path, body=None, raw=False, timeout=60):
        r = urllib.request.Request(
            self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
        )
        if body is not None:
            r.add_header("content-type", "application/json")
        r.add_header("authorization", f"Bearer {self.args.api_key}")
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                data, status = resp.read(), resp.status
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

    def wait_ready(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                return False
            st, j = self.call("GET", "/tt-liveness", timeout=5)
            if st == 200 and j.get("model_ready") is True:
                self.ready_after = time.time() - self.t_start
                self.rank_pids_at_ready = self.rank_pids()
                return True
            time.sleep(3)
        return False

    def rank0_pid(self):
        st, j = self.call("GET", "/tt-liveness", timeout=5)
        try:
            return int(j["worker_info"]["0"]["pid"])
        except (KeyError, TypeError, ValueError):
            return None

    def followers(self):
        r0 = self.rank0_pid()
        return [x for x in self.rank_pids() if x != r0]

    def generate(self, body=None, timeout=900):
        st, j = self.call("POST", GEN, body or REQUEST)
        if st != 202:
            return None, None, f"POST {st}"
        jid = j["id"]
        end = time.time() + timeout
        while time.time() < end:
            st, g = self.call("GET", f"{GEN}/{jid}")
            if g.get("status") in ("completed", "failed", "cancelled"):
                break
            time.sleep(2)
        else:
            return jid, None, "timeout"
        if g["status"] != "completed":
            return jid, None, f"{g['status']}: {g.get('error')}"
        st, video = self.call("GET", f"{GEN}/{jid}/download", raw=True, timeout=120)
        return jid, video, "ok" if st == 200 else f"download {st}"

    def wait_gone(self, timeout):
        """Seconds until no rank, no mpirun and no listener on the port is left, or None."""
        t0 = time.time()
        mine = list(self.rank_pids_at_ready) + ([self.mpirun] if self.mpirun else [])
        while time.time() - t0 < timeout:
            if not any(pid_alive(x) for x in mine) and not port_listeners(
                self.args.port
            ):
                return time.time() - t0
            time.sleep(0.5)
        return None

    def kill_everything(self):
        for pid in self.rank_pids() + (
            [self.mpirun_pid()] if self.mpirun_pid() else []
        ):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def md5(b):
    return hashlib.md5(b).hexdigest()


def probe_video(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,codec_name",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
    ).stdout
    try:
        return json.loads(out)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return {}


def psnr(a, b):
    out = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            a,
            "-i",
            b,
            "-lavfi",
            "[0:v][1:v]psnr",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stderr
    m = re.search(r"average:([0-9.]+)", out)
    return float(m.group(1)) if m else None


def start_and_wait(args, work, tag, timeout):
    job = Job(args, work, tag).start()
    ok = job.wait_ready(timeout)
    note(
        ok,
        f"{tag}: the job is ready",
        f"after {getattr(job, 'ready_after', 0):.0f}s" if ok else job.log_text()[-600:],
    )
    if not ok:
        job.kill_everything()
    return job if ok else None


def reset_chips():
    held = chips_held()
    if held:
        p("chips are held, no reset:", held[:2])
        return False
    cmd = "source '%s/python_env/bin/activate' && tt-smi -r" % ARGS.tt_metal
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=300)
    p("tt-smi -r exit", r.returncode)
    return r.returncode == 0


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-metal", required=True)
    ap.add_argument("--dit-cache", required=True)
    ap.add_argument("--api-key", default="your-secret-key")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ranks", type=int, choices=(2, 4), default=2)
    ap.add_argument("--host-topology", default="2, 2", help="4 ranks only")
    ap.add_argument("--card-order", default="0 1 2 3", help="4 ranks only")
    ap.add_argument("--workdir", default="")
    ap.add_argument(
        "--reference", default="", help="mp4 of REQUEST made on a single host"
    )
    ap.add_argument(
        "--ready-timeout",
        type=int,
        default=5400,
        help="first start builds the weight cache",
    )
    ap.add_argument(
        "--skip-teardown-phases", action="store_true", help="only phases 1 and 2"
    )
    args = ap.parse_args()
    ARGS = args
    work = args.workdir or tempfile.mkdtemp(prefix="tt-video-multihost.")
    os.makedirs(work, exist_ok=True)
    p("workdir:", work, "| ranks:", args.ranks)

    if port_listeners(args.port) or chips_held():
        p("the port is taken or a process holds the chips; refusing to start")
        return 2

    # ------------------------------------------------------------- phase 1
    p("=== phase 1: start, a generation, repeatability")
    job = start_and_wait(args, work, "p1", args.ready_timeout)
    if job is None:
        return finish()
    log = job.log_text()
    note(
        len(job.rank_pids()) == args.ranks,
        f"{args.ranks} rank processes are alive",
        str(job.rank_pids()),
    )
    note(
        "Creating in-process video service" in log,
        "rank 0 runs the model in the server process",
    )
    note(
        log.count("following rank 0") == args.ranks - 1,
        "every follower rank follows rank 0",
    )
    note("all ranks warm" in log, "rank 0 saw all ranks warm")
    note(
        job.rank0_pid() is not None and job.rank0_pid() not in job.followers(),
        "liveness names rank 0 only",
    )
    t0 = time.time()
    jid1, v1, why1 = job.generate()
    note(
        v1 is not None,
        "a generation completes",
        f"{why1} after {time.time() - t0:.0f}s",
    )
    jid2, v2, why2 = job.generate()
    note(v2 is not None, "the same request again completes", why2)
    if v1 and v2:
        note(
            md5(v1) == md5(v2),
            "the same request gives the same bytes",
            f"{md5(v1)} {md5(v2)}",
        )
        path = os.path.join(work, "p1_video.mp4")
        open(path, "wb").write(v1)
        info = probe_video(path)
        note(
            info.get("width") == 832
            and info.get("height") == 480
            and info.get("codec_name") == "h264",
            "the video is an h264 mp4 of 832x480",
            str(info),
        )
        if args.reference:
            value = psnr(path, args.reference)
            note(
                value is not None and value > 28.0,
                "close to the single-host video (PSNR > 28 dB)",
                f"PSNR {value} dB, md5 {md5(v1)} vs {md5(open(args.reference, 'rb').read())}",
            )
    # ------------------------------------------------------------- phase 2
    p("=== phase 2: two jobs behind each other")
    a = job.call("POST", GEN, {**REQUEST, "seed": 1})[1]
    b = job.call("POST", GEN, {**REQUEST, "seed": 2})[1]
    ok = True
    for j in (a, b):
        end = time.time() + 900
        while time.time() < end and job.call("GET", f"{GEN}/{j['id']}")[1].get(
            "status"
        ) not in ("completed", "failed"):
            time.sleep(2)
        ok = ok and job.call("GET", f"{GEN}/{j['id']}")[1].get("status") == "completed"
    note(ok, "both queued jobs complete and the ranks stay in step")
    st, _ = job.call("GET", "/tt-liveness")
    note(st == 200 and len(job.rank_pids()) == args.ranks, "all ranks alive afterwards")
    if args.skip_teardown_phases:
        job.proc.send_signal(signal.SIGTERM)
        job.wait_gone(60)
        return finish()

    # ------------------------------------------------------------- phase 3
    p("=== phase 3: SIGTERM to rank 0 while idle")
    r0 = job.rank0_pid()
    os.kill(r0, signal.SIGTERM)
    gone = job.wait_gone(60)
    note(
        gone is not None and gone < 15,
        "every rank and mpirun are gone",
        f"after {gone}s",
    )
    log = job.log_text()
    note(
        log.count("shutdown from rank 0") == args.ranks - 1,
        "every follower was told to shut down by rank 0",
    )
    note(
        log.count("Follower rank") >= args.ranks - 1 and "exited with 0" in log,
        "followers exit with 0",
    )
    note("Server shutdown complete (graceful)" in log, "rank 0 shut down gracefully")
    note(
        log.count("Closed mesh device") == args.ranks,
        "every rank closed its mesh",
        str(log.count("Closed mesh device")),
    )
    # ------------------------------------------------------------- phase 4
    p(
        "=== phase 4: SIGTERM to a follower while idle (the start also proves no reset was needed)"
    )
    job = start_and_wait(args, work, "p4", 1800)
    if job is None:
        note(reset_chips(), "chips reset after a failed start")
        return finish()
    f = job.followers()[0]
    os.kill(f, signal.SIGTERM)
    gone = job.wait_gone(60)
    note(gone is not None and gone < 20, "the whole job ends", f"after {gone}s")
    # ------------------------------------------------------------- phase 5
    p("=== phase 5: kill -9 of a follower while idle")
    job = start_and_wait(args, work, "p5", 1800)
    if job is None:
        note(reset_chips(), "chips reset after a failed start")
        return finish()
    os.kill(job.followers()[0], signal.SIGKILL)
    gone = job.wait_gone(60)
    note(gone is not None and gone < 20, "the whole job ends", f"after {gone}s")
    # ------------------------------------------------------------- phase 6
    p("=== phase 6: kill -9 of mpirun while idle")
    job = start_and_wait(args, work, "p6", 1800)
    if job is None:
        note(reset_chips(), "chips reset after a failed start")
        return finish()
    os.kill(job.mpirun_pid(), signal.SIGKILL)
    gone = job.wait_gone(60)
    note(
        gone is not None and gone < 20,
        "the ranks do not outlive mpirun",
        f"after {gone}s",
    )
    # ------------------------------------------------------------- phase 7
    p("=== phase 7: SIGTERM to rank 0 in the middle of a job")
    job = start_and_wait(args, work, "p7", 1800)
    if job is None:
        note(reset_chips(), "chips reset after a failed start")
        return finish()
    st, j = job.call("POST", GEN, REQUEST)
    time.sleep(30)
    os.kill(job.rank0_pid(), signal.SIGTERM)
    gone = job.wait_gone(90)
    note(
        gone is not None and gone < 40, "the job ends, no rank hangs", f"after {gone}s"
    )
    note(
        "did not finish within" in job.log_text(),
        "rank 0 reports that the run did not finish in time",
    )
    job = Job(args, work, "p7b").start()
    ok = job.wait_ready(300)
    p(
        "  OBSERVED: the start right after a mid-job SIGTERM",
        "works" if ok else "FAILS (the chips need tt-smi -r)",
    )
    note(
        True,
        "start after a mid-job SIGTERM without a chip reset is recorded",
        "works" if ok else "fails",
    )
    if ok:
        job.proc.send_signal(signal.SIGTERM)
        job.wait_gone(60)
    else:
        job.kill_everything()
        time.sleep(5)
        note(reset_chips(), "chips reset (nothing held them)")
    return finish()


def finish():
    passed, failed = results.count(True), results.count(False)
    p(f"{passed} PASS, {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
