# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Live hardware tests: MiniMax-H3 FL2VA and Ref2VA generation + DELETE on a running deployment.

These tests talk to a real tt-media-server over HTTP and occupy the mesh for
about an hour. They are **opt-in**: the module is skipped unless ``H3_LIVE_URL``
is set, so the default unit-test run is unaffected. The unit tests for the same
code paths live in ``test_job_manager.py`` and ``test_video_api.py``.

    H3_LIVE_URL=http://localhost:8000 pytest tests/test_minimax_h3_live.py -s -v

What one combination does (``TestFl2va`` keyframe layouts, ``TestRef2va``
reference mixes):

1. optionally start fresh workers for the task (see *deployment control*)
2. request 1 pays the kernel compile; ``H3_LIVE_REPEATS`` more requests are
   the warm measurements -- every request records API wall time and the mp4
   is downloaded and probed
3. DELETE contract on every job: 409 while live, 200 when terminal, then GET /
   download / DELETE answer 404, the job is gone from ``/v1/videos/jobs``; when
   the tests run on the server host the result file is gone from disk and the
   download left no remuxed copy behind in the temp dir

A MiniMax-H3 deployment serves one task. Without deployment control the tests
detect the served task (an empty body gets a "This deployment ..." 422 from the
endpoints it refuses and a field-validation 422 from the one it serves) and
skip the other class. With deployment control they switch tasks themselves.

Environment::

    H3_LIVE_URL             base URL of the server (required)
    H3_LIVE_API_KEY         bearer token (default: your-secret-key)
    H3_LIVE_REPEATS         warm generations after the compile request (default 3)
    H3_LIVE_POLL_TIMEOUT_S  per-job completion budget (default 2700)
    H3_LIVE_TASK            fl2va | ref2va -- what the server serves; probed when unset
    H3_LIVE_ASSETS_DIR      dir with img_512.png, vid_5s.mp4, aud_4s.mp3 and (optionally)
                            key_first.jpg, key_last.jpg; synthesised with ffmpeg when unset
    H3_LIVE_VIDEO_DIR       the server's TT_VIDEO_OUTPUT_DIR, when the tests run on the
                            server host -> enables the on-disk checks
    H3_LIVE_TMP_DIR         the server's temp dir for /download remux copies (default /tmp)
    H3_LIVE_REPORT          write the per-request timing table here as JSON

Deployment control (all optional, ``{task}`` is substituted)::

    H3_LIVE_START_CMD       start fresh workers + API for a task, e.g.
                            "bash /home/zni/h3-deploy/h3ctl.sh start {task}"
    H3_LIVE_WAIT_CMD        block until the deployment is ready, e.g.
                            "bash /home/zni/h3-deploy/h3ctl.sh wait-ready 1800"
    H3_LIVE_RESET_CMD       reset the chips on every host, e.g.
                            "bash /home/zni/h3-deploy/h3ctl.sh reset"
    H3_LIVE_RESET_SETTLE_S  seconds to let the inter-host links retrain after a reset (45)
    H3_LIVE_START_RETRIES   reset + start cycles after a failed start (2)
    H3_LIVE_RESET_FIRST     1 (default): reset the chips before this session's first start too.
                            Twice on the quad a run killed mid-request left the fabric in a state
                            where the next process hung 300 s into loading the vision tower
                            ("device timeout in fetch queue wait" at blocks.16.mlp.linear_fc2);
                            a reset first costs ~110 s once. 0 skips it.

When a start command is configured every combination begins with fresh workers
(a failed job does not release device memory, so anything measured after a
failure in the same process is noise -- see tt-inference-server#5044). The
chips are reset, not merely restarted, whenever the hardware may be wedged:

* a fresh start does not come up ready -> reset, let the links retrain, start
  again (``H3_LIVE_START_RETRIES`` cycles), fail the run if that does not help;
* a job failed or timed out (device memory is poisoned / a rank may hang) ->
  the next start is preceded by a reset;
* a job is still not terminal after the poll budget -> reset right away;
* at session end, if the last combination left the deployment poisoned, reset
  and restart it so the cluster is not left wedged for the next user.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.live

LIVE_URL = os.environ.get("H3_LIVE_URL", "").rstrip("/")
if not LIVE_URL:
    pytest.skip(
        "live MiniMax-H3 tests need H3_LIVE_URL (e.g. http://localhost:8000)",
        allow_module_level=True,
    )

API_KEY = os.environ.get("H3_LIVE_API_KEY", "your-secret-key")
REPEATS = int(os.environ.get("H3_LIVE_REPEATS", "3"))
POLL_TIMEOUT_S = float(os.environ.get("H3_LIVE_POLL_TIMEOUT_S", "2700"))
VIDEO_DIR = os.environ.get("H3_LIVE_VIDEO_DIR") or None
TMP_DIR = os.environ.get("H3_LIVE_TMP_DIR", "/tmp")
START_CMD = os.environ.get("H3_LIVE_START_CMD") or None
WAIT_CMD = os.environ.get("H3_LIVE_WAIT_CMD") or None
RESET_CMD = os.environ.get("H3_LIVE_RESET_CMD") or None
RESET_SETTLE_S = float(os.environ.get("H3_LIVE_RESET_SETTLE_S", "45"))
START_RETRIES = int(os.environ.get("H3_LIVE_START_RETRIES", "2"))
RESET_FIRST = os.environ.get("H3_LIVE_RESET_FIRST", "1") == "1"

ENDPOINT = {
    "t2va": "/v1/videos/generations",
    "fl2va": "/v1/videos/generations/i2v",
    "ref2va": "/v1/videos/generations/ref2va",
}
PROMPT = "A calm seaside village at golden hour, gentle waves"
# 1x1 transparent PNG: enough to get past body validation, never reaches a device
PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

# --------------------------------------------------------------------------- HTTP


def http(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    timeout: float = 120,
    auth: bool = True,
    save_to: Path | None = None,
) -> tuple[int | None, Any]:
    """-> (status, parsed json | text | {'bytes': n}); status None on a transport error."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {API_KEY}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{LIVE_URL}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            if save_to is not None:
                save_to.write_bytes(raw)
                return r.status, {"bytes": len(raw)}
            txt = raw.decode(errors="replace")
            try:
                return r.status, json.loads(txt or "{}")
            except ValueError:
                return r.status, txt
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(txt)
        except ValueError:
            return e.code, txt
    except Exception as e:  # noqa: BLE001 - transport errors are data here
        return None, f"{type(e).__name__}: {e}"


def probe_served_task() -> str:
    """The one video endpoint whose empty-body 422 is field validation, not a deployment refusal."""
    served = []
    for task, path in ENDPOINT.items():
        code, resp = http("POST", path, {}, timeout=60)
        detail = resp.get("detail") if isinstance(resp, dict) else None
        if code == 422 and isinstance(detail, list):
            served.append(task)
    if len(served) != 1:
        pytest.fail(f"could not determine the served task from the 422 probes: {served}")
    return served[0]


# --------------------------------------------------------------------------- shell


def sh(cmd: str, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {cmd}"


def ffprobe(path: Path) -> dict:
    """{'video': bool, 'audio': bool, 'width', 'height', 'frames', 'duration'} or {} without ffprobe."""
    if not shutil.which("ffprobe"):
        return {}
    rc, out = sh(
        f"ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames "
        f"-show_entries format=duration -of json {shlex.quote(str(path))}",
        timeout=60,
    )
    if rc != 0:
        return {"error": out[-200:]}
    info = json.loads(out)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {
        "video": v is not None,
        "audio": a is not None,
        "width": v and v.get("width"),
        "height": v and v.get("height"),
        "frames": v and int(v.get("nb_frames") or 0),
        "duration": float(info.get("format", {}).get("duration") or 0),
    }


# --------------------------------------------------------------------------- deployment control


class Deployment:
    """Fresh workers per combination and chip resets, when the operator wired the commands in."""

    def __init__(self) -> None:
        self.controllable = bool(START_CMD)
        self.task: str | None = None
        # A job failed / hung since the last fresh start -> reset before the next one. Starts True
        # when RESET_FIRST: whatever ran before this session may have been killed mid-request.
        self.poisoned = RESET_FIRST
        self.log: list[str] = []

    def _say(self, msg: str) -> None:
        line = f"[deployment {time.strftime('%H:%M:%S')}] {msg}"
        self.log.append(line)
        print(line, flush=True)

    def reset_chips(self) -> None:
        if not RESET_CMD:
            self._say("no H3_LIVE_RESET_CMD configured; cannot reset the chips")
            return
        self._say(f"resetting chips: {RESET_CMD}")
        rc, out = sh(RESET_CMD, timeout=900)
        self._say(f"reset rc={rc} {out.strip().splitlines()[-1][:160] if out.strip() else ''}")
        # The reset returns when the chips are back; the inter-host Ethernet links
        # retrain a little later. Starting before they are up fails topology
        # discovery ("Exit node connection not found between host ... and host ...").
        self._say(f"letting the links settle for {RESET_SETTLE_S:.0f}s")
        time.sleep(RESET_SETTLE_S)

    def _start_once(self, task: str) -> tuple[bool, str]:
        assert START_CMD
        rc, out = sh(START_CMD.format(task=task), timeout=900)
        if rc != 0:
            return False, f"start rc={rc}: {out[-300:]}"
        if WAIT_CMD:
            rc, out = sh(WAIT_CMD.format(task=task), timeout=3600)
            if rc != 0:
                return False, f"wait rc={rc}: {out[-300:]}"
        # with or without a wait command, believe the server itself
        deadline = time.time() + (60 if WAIT_CMD else 1800)
        while True:
            code, live = http("GET", "/tt-liveness", timeout=30)
            ready = isinstance(live, dict) and live.get("model_ready") is True
            if ready or time.time() > deadline:
                return ready, f"liveness={code} model_ready={ready}"
            time.sleep(15)

    def fresh(self, task: str) -> None:
        """Bring up fresh workers for ``task``; reset the chips first if the last run poisoned them."""
        if not self.controllable:
            return
        if self.poisoned:
            why = "a job failed or hung" if self.task else "first start of this session (H3_LIVE_RESET_FIRST)"
            self._say(f"{why}; resetting before starting {task}")
            self.reset_chips()
        self._say(f"starting fresh {task} workers")
        ok, detail = self._start_once(task)
        for attempt in range(1, START_RETRIES + 1):
            if ok:
                break
            self._say(f"start failed ({detail}); resetting chips and retrying ({attempt}/{START_RETRIES})")
            self.reset_chips()
            ok, detail = self._start_once(task)
        if not ok:
            pytest.fail(f"deployment for {task} did not come up after {START_RETRIES} chip reset(s): {detail}")
        self.task = task
        self.poisoned = False
        self._say(f"{task} ready ({detail})")

    def mark_poisoned(self, why: str, reset_now: bool = False) -> None:
        self.poisoned = True
        self._say(f"deployment poisoned: {why}")
        if reset_now and self.controllable:
            self.reset_chips()

    def leave_clean(self) -> None:
        """Session end: never leave the cluster wedged for the next user."""
        if self.controllable and self.poisoned and self.task:
            self._say("last combination left the deployment poisoned; reset + restart before leaving")
            self.reset_chips()
            ok, detail = self._start_once(self.task)
            self._say(f"restart {'ok' if ok else 'FAILED'} ({detail})")


# --------------------------------------------------------------------------- assets


@dataclass
class Assets:
    img: str  # base64
    vid: str
    aud: str
    key_first: str
    key_last: str


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _synth(out: Path) -> None:
    """Tiny synthetic references + two distinct 1344x768 keyframes, all via ffmpeg."""
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg is needed to synthesise assets (or set H3_LIVE_ASSETS_DIR)")
    cmds = [
        f"{ff} -y -v error -f lavfi -i color=c=steelblue:s=512x512 -frames:v 1 {out/'img_512.png'}",
        f"{ff} -y -v error -f lavfi -i testsrc2=size=320x240:rate=24:duration=5 -c:v libx264 -pix_fmt yuv420p {out/'vid_5s.mp4'}",
        f"{ff} -y -v error -f lavfi -i sine=frequency=440:duration=4 -c:a libmp3lame -q:a 6 {out/'aud_4s.mp3'}",
        f"{ff} -y -v error -f lavfi -i testsrc2=size=1344x768:rate=1 -frames:v 1 -q:v 2 {out/'key_first.jpg'}",
        f"{ff} -y -v error -f lavfi -i color=c=darkorange:s=1344x768 -frames:v 1 -q:v 2 {out/'key_last.jpg'}",
    ]
    for c in cmds:
        rc, o = sh(c, timeout=120)
        if rc != 0:
            pytest.skip(f"ffmpeg could not synthesise an asset: {o[-200:]}")


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> Assets:
    """Reference assets from H3_LIVE_ASSETS_DIR, anything missing there synthesised with ffmpeg."""
    synth = tmp_path_factory.mktemp("h3-assets")
    src = Path(os.environ["H3_LIVE_ASSETS_DIR"]) if os.environ.get("H3_LIVE_ASSETS_DIR") else None
    names = ("img_512.png", "vid_5s.mp4", "aud_4s.mp3", "key_first.jpg", "key_last.jpg")
    if src is None or not all((src / n).exists() for n in names):
        _synth(synth)

    def pick(name: str) -> Path:
        return src / name if src is not None and (src / name).exists() else synth / name

    return Assets(*(_b64(pick(n)) for n in names))


# --------------------------------------------------------------------------- combinations


@dataclass(frozen=True)
class Combo:
    name: str
    task: str
    keyframes: tuple[int, ...] = ()  # fl2va: frame_pos values
    images: int = 0                  # ref2va
    videos: int = 0
    audios: int = 0
    note: str = ""

    def body(self, a: Assets) -> dict:
        if self.task == "fl2va":
            pick = {0: a.key_first, -1: a.key_last}
            return {"prompt": PROMPT, "seed": 7,
                    "image_prompts": [{"image": pick[p], "frame_pos": p} for p in self.keyframes]}
        refs: dict = {}
        if self.images:
            refs["images"] = [{"b64": a.img} for _ in range(self.images)]
        if self.videos:
            refs["videos"] = [{"b64": a.vid} for _ in range(self.videos)]
        if self.audios:
            refs["audios"] = [{"b64": a.aud} for _ in range(self.audios)]
        return {"prompt": PROMPT, "references": refs}

    @property
    def refs(self) -> int:
        return self.images + self.videos + self.audios


FL2VA_COMBOS = [
    Combo("first+last", "fl2va", keyframes=(0, -1)),
    Combo("first_only", "fl2va", keyframes=(0,)),
    Combo("last_only", "fl2va", keyframes=(-1,)),
]

# Order matters when there is no deployment control: everything after an OOM in
# the same worker process is noise, so the known-limit mixes come last.
REF2VA_COMBOS = [
    Combo("vid1", "ref2va", videos=1),
    Combo("img1", "ref2va", images=1),
    Combo("aud3_img1", "ref2va", images=1, audios=3, note="audio modality cap"),
    Combo("mix_2i_3v", "ref2va", images=2, videos=3, note="5 visual refs"),
    Combo("img6", "ref2va", images=6),
    # These four completed once per worker process and OOMed on the second request until metal
    # zni/h3-ref2va-serving-fixes (piecewise VAE readback, transient conditioner gathers,
    # policy-sized arena caps); measured 4/4 on OM Quad1 2026-09-04.
    Combo("mix_3i_3v", "ref2va", images=3, videos=3, note="67,584 condition rows, needs the 92,160 cap"),
    Combo("img7", "ref2va", images=7),
    Combo("img9", "ref2va", images=9, note="policy maximum images; 36,946 presentation tokens"),
]
SECOND_REQUEST_OOM = pytest.mark.xfail(
    reason="tt-inference-server#5044: at the 176,128 rung the DiT's per-rung resident state leaves "
    "~350 MB too little for the conditioner's second pass (full-sequence [1, L, 5120] "
    "embedding gather), so the mix completes once per worker process and OOMs on the next request",
    strict=False,
)
REF2VA_LIMIT_COMBOS = [
    pytest.param(Combo("img8", "ref2va", images=8), marks=SECOND_REQUEST_OOM, id="img8"),
    # 6 images + 3 clips = 79,872 condition video rows: the heaviest mixed case the policy admits
    # short of 9 + 3; admitted by the 92,160 cap, first request completes (413 s), second OOMs.
    pytest.param(Combo("mix_6i_3v", "ref2va", images=6, videos=3), marks=SECOND_REQUEST_OOM, id="mix_6i_3v"),
]
REF2VA_OVER_LIMIT = Combo("img9", "ref2va", images=9, note="OOMs on the first request")


# --------------------------------------------------------------------------- report


@dataclass
class Report:
    rows: list[dict] = field(default_factory=list)

    def add(self, **row) -> None:
        self.rows.append(row)

    def dump(self) -> None:
        if not self.rows:
            return
        print("\n" + "=" * 96)
        print(f" {'combo':14} {'task':6} {'request':8} {'status':10} {'wall s':>7} {'mp4 B':>9}  probe")
        print("-" * 96)
        for r in self.rows:
            probe = r.get("probe") or {}
            p = f"{probe.get('width')}x{probe.get('height')} {probe.get('frames')}f {probe.get('duration', 0):.2f}s" if probe.get("video") else ""
            print(f" {r['combo']:14} {r['task']:6} {r['request']:8} {r['status']:10} {r['wall_s']:7.1f} {str(r.get('mp4_bytes') or '-'):>9}  {p}")
        print("=" * 96, flush=True)
        if os.environ.get("H3_LIVE_REPORT"):
            Path(os.environ["H3_LIVE_REPORT"]).write_text(json.dumps(self.rows, indent=1))


@pytest.fixture(scope="module")
def deployment():
    dep = Deployment()
    yield dep
    dep.leave_clean()


@pytest.fixture(scope="module")
def report():
    rep = Report()
    yield rep
    rep.dump()


@pytest.fixture(scope="module")
def served_task(deployment) -> str | None:
    """The task the server serves right now; None means the tests switch it themselves."""
    if deployment.controllable:
        return None
    task = os.environ.get("H3_LIVE_TASK") or probe_served_task()
    print(f"\n[live] server at {LIVE_URL} serves {task}", flush=True)
    return task


def _need_task(task: str, served_task: str | None, deployment: Deployment) -> None:
    """Make ``task`` the served task, or skip when the tests cannot switch."""
    if deployment.controllable:
        if deployment.task != task or deployment.poisoned:
            deployment.fresh(task)
        return
    if served_task != task:
        pytest.skip(f"server serves {served_task}, this test needs {task} (set H3_LIVE_START_CMD to switch)")


def _fresh_or_skip(task: str, served_task: str | None, deployment: Deployment) -> None:
    """Every combination starts on fresh workers when the tests can restart them."""
    if deployment.controllable:
        deployment.fresh(task)
    else:
        _need_task(task, served_task, deployment)


# --------------------------------------------------------------------------- disk snapshot


class Disk:
    """On-disk evidence, only when the tests run on the server host (H3_LIVE_VIDEO_DIR set)."""

    enabled = VIDEO_DIR is not None

    @classmethod
    def videos(cls) -> set[str]:
        return set(glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))) if cls.enabled else set()

    @classmethod
    def remux_copies(cls) -> int:
        return len(glob.glob(os.path.join(TMP_DIR, "tmp*.mp4"))) if cls.enabled else 0


# --------------------------------------------------------------------------- the checks


def _wait_terminal(job_id: str, t0: float, deployment: Deployment) -> tuple[str, dict, float]:
    last = None
    while True:
        _, job = http("GET", f"/v1/videos/generations/{job_id}", timeout=60)
        status = job.get("status") if isinstance(job, dict) else None
        el = time.time() - t0
        if status != last:
            print(f"    {el:6.0f}s  {job_id[:8]} status={status}", flush=True)
            last = status
        if status in ("completed", "failed", "cancelled"):
            return status, job, round(el, 1)
        if el >= POLL_TIMEOUT_S:
            # a rank is probably hung: this is the one place a reset cannot wait
            deployment.mark_poisoned(f"job {job_id} still {status} after {el:.0f}s", reset_now=True)
            return "timeout", job, round(el, 1)
        time.sleep(5)


def _delete_contract(job_id: str, expect_file_gone: str | None = None) -> None:
    remux_before = Disk.remux_copies()
    code, resp = http("DELETE", f"/v1/videos/generations/{job_id}", timeout=60)
    assert code == 200 and isinstance(resp, dict), f"DELETE terminal job -> {code} {resp}"
    assert resp.get("deleted") is True and resp.get("id") == job_id and resp.get("object") == "video", resp
    code, _ = http("GET", f"/v1/videos/generations/{job_id}", timeout=30)
    assert code == 404, f"GET after delete -> {code}"
    code, _ = http("GET", f"/v1/videos/generations/{job_id}/download", timeout=30)
    assert code == 404, f"download after delete -> {code}"
    code, _ = http("DELETE", f"/v1/videos/generations/{job_id}", timeout=30)
    assert code == 404, f"second DELETE -> {code}"
    code, jobs = http("GET", "/v1/videos/jobs", timeout=30)
    assert code == 200 and isinstance(jobs, list)
    assert all(j.get("id") != job_id for j in jobs), "deleted job still listed in /v1/videos/jobs"
    if expect_file_gone:
        assert not os.path.exists(expect_file_gone), f"result file still on disk: {expect_file_gone}"
    if Disk.enabled:
        assert Disk.remux_copies() == remux_before, "DELETE created a temp remux copy"


def _generate_repeatedly(combo: Combo, a: Assets, deployment: Deployment, report: Report, tmp_path: Path) -> None:
    """The body of a combination test: compile + REPEATS generations, then DELETE all of them."""
    body = combo.body(a)
    videos_before = Disk.videos()
    remux_before = Disk.remux_copies()
    jobs: list[dict] = []
    for i in range(1 + REPEATS):
        label = "compile" if i == 0 else f"gen{i}"
        t0 = time.time()
        code, resp = http("POST", ENDPOINT[combo.task], body, timeout=300)
        assert code in (200, 202) and isinstance(resp, dict) and resp.get("id"), f"{label} submit -> {code} {str(resp)[:300]}"
        job_id = resp["id"]
        print(f"  {combo.name} {label}: submitted job {job_id}", flush=True)

        if i == 0:
            # the one moment we know the job is live
            time.sleep(2)
            code, resp = http("DELETE", f"/v1/videos/generations/{job_id}", timeout=30)
            assert code == 409, f"DELETE while live -> {code} {resp}"
            code, j = http("GET", f"/v1/videos/generations/{job_id}", timeout=30)
            assert code == 200 and j.get("status") in ("queued", "in_progress"), f"job vanished after refused delete: {code} {j}"

        status, job, wall = _wait_terminal(job_id, t0, deployment)
        row: dict = {"combo": combo.name, "task": combo.task, "request": label, "status": status, "wall_s": wall, "job_id": job_id}
        if status == "completed":
            dl = tmp_path / f"{combo.name}.{label}.mp4"
            code, r = http("GET", f"/v1/videos/generations/{job_id}/download", timeout=600, save_to=dl)
            row["download_code"] = code
            row["mp4_bytes"] = r.get("bytes") if isinstance(r, dict) else None
            assert code == 200 and (row["mp4_bytes"] or 0) > 10_000, f"{label} download -> {code} {row['mp4_bytes']} B"
            assert dl.read_bytes()[4:8] == b"ftyp", "download is not an MP4 container"
            row["probe"] = ffprobe(dl)
            if row["probe"]:
                assert row["probe"].get("video") and row["probe"].get("audio"), f"H3 must emit video + audio: {row['probe']}"
            if Disk.enabled:
                time.sleep(1)
                row["remux_copies"] = Disk.remux_copies()
        else:
            row["error"] = str(job.get("error"))[:400] if isinstance(job, dict) else None
            if status != "timeout":
                deployment.mark_poisoned(f"{combo.name} {label} ended {status}: {row['error']}")
        report.add(**row)
        jobs.append(row)
        if status != "completed":
            break  # anything after a failure in this process is noise

    # --- disk: exactly the new files we expect, then none of them after DELETE ---
    new_files: set[str] = set()
    if Disk.enabled:
        new_files = Disk.videos() - videos_before
        completed = sum(1 for j in jobs if j["status"] == "completed")
        assert len(new_files) == completed, f"expected {completed} new mp4 in {VIDEO_DIR}, found {len(new_files)}"
        leaked = [j["remux_copies"] - remux_before for j in jobs if j["status"] == "completed"]
        assert not any(leaked), f"/download left remuxed copies behind in {TMP_DIR}: +{leaked}"

    # --- DELETE contract on every job, completed or failed --------------------------
    for j in jobs:
        if j["status"] == "timeout":
            continue  # not terminal: cannot be deleted, and the run already flagged it
        _delete_contract(j["job_id"])
    if Disk.enabled:
        assert Disk.videos() == videos_before, f"files left behind after DELETE: {Disk.videos() - videos_before}"

    failed = [j for j in jobs if j["status"] != "completed"]
    assert not failed, f"{combo.name}: {failed[0]['request']} ended {failed[0]['status']}: {failed[0].get('error')}"
    assert len(jobs) == 1 + REPEATS


def _cancel_then_delete(task: str, body: dict, deployment: Deployment) -> None:
    videos_before = Disk.videos()
    t0 = time.time()
    code, resp = http("POST", ENDPOINT[task], body, timeout=300)
    assert code in (200, 202) and isinstance(resp, dict), f"submit -> {code} {resp}"
    job_id = resp["id"]
    for _ in range(24):  # let the workers pick it up
        _, j = http("GET", f"/v1/videos/generations/{job_id}", timeout=30)
        if isinstance(j, dict) and j.get("status") == "in_progress":
            break
        time.sleep(5)
    code, resp = http("POST", f"/v1/videos/generations/{job_id}/cancel", timeout=60)
    assert code == 200, f"cancel -> {code} {resp}"
    status, _, wall = _wait_terminal(job_id, t0, deployment)
    print(f"  cancelled job reached {status} after {wall}s", flush=True)
    assert status == "cancelled", f"cancelled job ended as {status}"
    code, _ = http("GET", f"/v1/videos/generations/{job_id}/download", timeout=60)
    assert code == 404, f"download of a cancelled job -> {code}"
    _delete_contract(job_id)
    if Disk.enabled:
        assert Disk.videos() == videos_before, "a cancelled job left an mp4 behind after DELETE"


def _routing(task: str) -> None:
    others = {
        "t2va": ("/v1/videos/generations", {"prompt": "x"}),
        "fl2va": ("/v1/videos/generations/i2v", {"prompt": "x", "image_prompts": [{"image": PX, "frame_pos": 0}]}),
        "ref2va": ("/v1/videos/generations/ref2va", {"prompt": "x", "references": {"images": [{"b64": PX}]}}),
    }
    for other, (path, body) in others.items():
        if other == task:
            continue
        code, resp = http("POST", path, body, timeout=60)
        detail = resp.get("detail") if isinstance(resp, dict) else resp
        assert code == 422 and isinstance(detail, str) and "deployment" in detail.lower(), (
            f"{other} body on a {task} deployment -> {code} {detail}")
    code, _ = http("DELETE", "/v1/videos/generations/does-not-exist", timeout=30)
    assert code == 404
    code, _ = http("DELETE", "/v1/videos/generations/x", timeout=30, auth=False)
    assert code == 401


# --------------------------------------------------------------------------- FL2VA


class TestFl2va:
    """Keyframe layouts on a MiniMax-H3 FL2VA deployment (MODEL_RUNNER=tt-minimax-h3-fl2va)."""

    def test_routing_and_delete_negatives(self, served_task, deployment):
        _need_task("fl2va", served_task, deployment)
        _routing("fl2va")

    @pytest.mark.parametrize("combo", FL2VA_COMBOS, ids=lambda c: c.name)
    def test_repeated_generation_and_delete(self, combo, assets, served_task, deployment, report, tmp_path):
        _fresh_or_skip("fl2va", served_task, deployment)
        _generate_repeatedly(combo, assets, deployment, report, tmp_path)

    def test_cancel_then_delete(self, assets, served_task, deployment):
        _need_task("fl2va", served_task, deployment)
        _cancel_then_delete("fl2va", FL2VA_COMBOS[0].body(assets), deployment)


# --------------------------------------------------------------------------- Ref2VA


class TestRef2va:
    """Reference mixes on a MiniMax-H3 Ref2VA deployment (MODEL_RUNNER=tt-minimax-h3-ref2va)."""

    def test_routing_and_delete_negatives(self, served_task, deployment):
        _need_task("ref2va", served_task, deployment)
        _routing("ref2va")

    @pytest.mark.parametrize("combo", REF2VA_COMBOS, ids=lambda c: c.name)
    def test_repeated_generation_and_delete(self, combo, assets, served_task, deployment, report, tmp_path):
        _fresh_or_skip("ref2va", served_task, deployment)
        _generate_repeatedly(combo, assets, deployment, report, tmp_path)

    def test_cancel_then_delete(self, assets, served_task, deployment):
        _need_task("ref2va", served_task, deployment)
        _cancel_then_delete("ref2va", REF2VA_COMBOS[0].body(assets), deployment)

    # -- known limits, last: they poison the worker process ------------------------------

    @pytest.mark.parametrize("combo", REF2VA_LIMIT_COMBOS)
    def test_known_limit_mix_repeats(self, combo, assets, served_task, deployment, report, tmp_path):
        _fresh_or_skip("ref2va", served_task, deployment)
        _generate_repeatedly(combo, assets, deployment, report, tmp_path)

    def test_failed_job_is_deletable(self, assets, served_task, deployment, report):
        """Over the reference limit the job fails on device; a failed job is terminal and deletable."""
        _fresh_or_skip("ref2va", served_task, deployment)
        combo = REF2VA_OVER_LIMIT
        videos_before = Disk.videos()
        t0 = time.time()
        code, resp = http("POST", ENDPOINT["ref2va"], combo.body(assets), timeout=300)
        assert code in (200, 202) and isinstance(resp, dict), f"submit -> {code} {resp}"
        job_id = resp["id"]
        status, job, wall = _wait_terminal(job_id, t0, deployment)
        report.add(combo=combo.name, task="ref2va", request="compile", status=status, wall_s=wall, job_id=job_id,
                   error=str(job.get("error"))[:300] if isinstance(job, dict) else None)
        assert status in ("completed", "failed"), f"over-limit job ended {status}"
        if status == "failed":
            deployment.mark_poisoned(f"{combo.name} failed as expected")
            code, _ = http("GET", f"/v1/videos/generations/{job_id}/download", timeout=60)
            assert code == 404, f"download of a failed job -> {code}"
        _delete_contract(job_id)
        if Disk.enabled:
            assert Disk.videos() == videos_before
