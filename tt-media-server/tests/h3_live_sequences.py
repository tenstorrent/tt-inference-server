# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Ordered request sequences against ONE worker process, judged output by output.

The shape/rung/audio bugs found on OM Quad1 (2026-09-05) only show up when several requests of
different shapes share a process: a rung's trace is captured on its second request and replayed on
later ones, and what was allocated in between decides whether the replay is sound.  ``run_sequence``
therefore keeps the deployment as it is (``fresh=False``) or starts fresh workers once
(``fresh=True``) and then sends the specs strictly in order, downloading and judging every result
with :mod:`h3_media_checks` and recording rung / capture / compile facts from the worker log when it
is readable (``H3_LIVE_WORKER_LOG``, default ``$H3_DEPLOY_DIR/workers.log``).

Canvas expectations reproduce ``models.tt_dit.pipelines.minimax_h3.packing.resolve_canvas_size``
(short edge 768, area capped at 768*1344, both axes rounded to a multiple of 32) so a test can assert
the served canvas without importing tt-metal.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h3_live_common as live
import h3_media_checks as media

SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
DURATIONS_S = tuple(range(4, 16))
# t2va/fl2va trace-bucket ladder at the (4, 32) preset; a request pads to the smallest rung >= its
# packed length.  Only used to LABEL rows (the pipeline logs the rung it chose).
BUCKET_LADDER = (22528, 31744, 44032, 61440, 86016, 119808)   # top rung 118784 -> 119808 in metal 8fb0c4c0483
# ref2va has its own ladder since metal e9dfcbdae42 (JIT-compile bucket + ref2va trace region).
REF2VA_BUCKET_LADDER = (61440, 86016, 118784, 176128, 245760, 322560)
WORKER_LOG = os.environ.get("H3_LIVE_WORKER_LOG") or (
    os.path.join(os.environ["H3_DEPLOY_DIR"], "workers.log") if os.environ.get("H3_DEPLOY_DIR") else None
)


def parse_aspect(aspect: str) -> tuple[int, int]:
    w, h = aspect.split(":")
    return int(w), int(h)


def expected_canvas(aspect: str) -> tuple[int, int]:
    """(width, height) the server resolves for ``aspect`` -- same arithmetic as packing.py."""
    aw, ah = parse_aspect(aspect)
    ratio = aw / ah
    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    return int(round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE), int(round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def rows_per_frame(aspect: str) -> int:
    w, h = expected_canvas(aspect)
    return (w // 32) * (h // 32)


def packed_length_estimate(aspect: str, seconds: float, *, task: str = "t2va", keyframes: int = 0, text_tokens: int = 40) -> int:
    """Media rows + prompt tokens as the pipeline packs them (fl2va adds one canvas of condition rows
    and one canvas of vision tokens per keyframe).  Good to +-50 tokens; used only for labelling."""
    frames = media.expected_frames(seconds)
    latent_frames = ((frames - 5) // 17) * 5 + 2
    audio_rows = 2 * int(round(frames / 24 * 40))
    rpf = rows_per_frame(aspect)
    n = text_tokens + latent_frames * rpf + audio_rows
    if task == "fl2va":
        n += keyframes * (rpf + rpf + 4)
    return n


def expected_rung(aspect: str, seconds: float, task: str = "t2va", **kw) -> int | None:
    n = packed_length_estimate(aspect, seconds, task=task, **kw)
    ladder = REF2VA_BUCKET_LADDER if task == "ref2va" else BUCKET_LADDER
    return next((r for r in ladder if n <= r), None)


@dataclass(frozen=True)
class Spec:
    """One request: ``task`` t2va|fl2va|ref2va, ``aspect`` like "16:9", integer ``seconds`` 4..15,
    fl2va ``keyframes`` as frame_pos values ((0,) first only, (0, -1) both), ref2va ``ref_images``
    (count of the image reference), ``seed``, ``prompt``."""
    task: str = "t2va"
    aspect: str = "16:9"
    seconds: int = 5
    keyframes: tuple[int, ...] = ()
    ref_images: int = 0
    ref_videos: int = 0
    seed: int = 7
    prompt: str = live.PROMPT
    label: str = ""

    @property
    def tag(self) -> str:
        kf = {(): "", (0,): "+first", (-1,): "+last", (0, -1): "+first+last"}.get(self.keyframes, f"+kf{self.keyframes}")
        refs = (f"+{self.ref_images}img" if self.ref_images else "") + (f"+{self.ref_videos}vid" if self.ref_videos else "")
        return self.label or f"{self.task} {self.aspect} {self.seconds}s{kf}{refs}"

    def body(self, assets: live.Assets) -> dict:
        b = {"prompt": self.prompt, "seed": self.seed, "aspect_ratio": self.aspect, "duration_seconds": self.seconds}
        if self.task == "fl2va":
            pick = {0: assets.key_first, -1: assets.key_last}
            b["image_prompts"] = [{"image": pick[p], "frame_pos": p} for p in self.keyframes]
        elif self.task == "ref2va":
            refs: dict = {}
            if self.ref_images or not self.ref_videos:
                refs["images"] = [{"b64": assets.img} for _ in range(max(1, self.ref_images))]
            if self.ref_videos:
                refs["videos"] = [{"b64": assets.vid} for _ in range(self.ref_videos)]
            b["references"] = refs
        return b


@dataclass
class Result:
    spec: Spec
    job_id: str | None = None
    status: str = "not-submitted"
    submit_code: int | None = None
    wall_s: float = 0.0
    error: str | None = None
    mp4: Path | None = None
    mp4_bytes: int | None = None
    sha256: str | None = None
    verdict: media.Verdict | None = None
    rung: int | None = None            # from the worker log: "packed sequence N -> bucket R"
    captured: bool | None = None       # worker log: "capturing trace..." inside this request's window
    compiled_kernels: int | None = None  # worker log: BuildKernels lines inside the window
    steps: int | None = None           # worker log: "<task> WxH, F frames (..), .. N steps, anchors=.." / "reference_resize_mode=.."
    pre_captured: bool | None = None   # the rung was captured at construction (metal >= 56cdeeb9095 warms the ladder in __init__)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and (self.verdict is None or self.verdict.ok)

    def row(self) -> dict:
        v = self.verdict
        return {
            "combo": self.spec.tag, "task": self.spec.task, "request": "seq", "status": self.status, "wall_s": self.wall_s,
            "job_id": self.job_id, "mp4_bytes": self.mp4_bytes, "sha256": self.sha256, "error": self.error,
            "rung": self.rung, "captured": self.captured, "compiled_kernels": self.compiled_kernels, "steps": self.steps,
            "probe": {"video": bool(v and v.probe and v.probe.video_streams), "audio": bool(v and v.probe and v.probe.audio_streams),
                      "width": v.probe.width if v and v.probe else None, "height": v.probe.height if v and v.probe else None,
                      "frames": v.probe.nb_frames if v and v.probe else None, "duration": v.probe.duration_s if v and v.probe else 0} if v else None,
            "audio_mean_db": v.audio.mean_db if v and v.audio else None, "audio_max_db": v.audio.max_db if v and v.audio else None,
            "content_ok": v.ok if v else None, "content_reasons": v.reasons if v else None,
        }

    def describe(self) -> str:
        s = f"{self.spec.tag}: {self.status} in {self.wall_s:.0f}s"
        if self.rung:
            s += f" rung={self.rung}{' capture' if self.captured else ''}{f' jit={self.compiled_kernels}' if self.compiled_kernels else ''}"
        if self.verdict:
            s += f" | {self.verdict.summary()}"
        if self.error:
            s += f" | ERR {self.error[:160]}"
        return s


# --------------------------------------------------------------------------- host-side evidence

SIDEFILE_DIR = os.environ.get("H3_LIVE_SIDEFILE_DIR", "/dev/shm")


def count_side_files() -> int | None:
    """Number of ``tt_img_*`` side-files the SP runner has on tmpfs (None when not on the server host)."""
    if not (live.Disk.enabled or os.environ.get("H3_LIVE_ON_SERVER_HOST") == "1"):
        return None
    try:
        return sum(1 for n in os.listdir(SIDEFILE_DIR) if n.startswith("tt_img_"))
    except OSError:
        return None


# Lines a healthy run prints anyway; everything else that says critical/fatal/timeout/traceback is news.
BENIGN_LOG_PATTERNS = (
    "DRAM Auto slice could not find valid slice configuration",   # matmul config search fallback, caught
    "Attempting fallback to width-slicing",
    "distributed across mesh device unevenly",
    "Padding out_channels",
    "depthwise conv1d needs C-chunking",
)
# Per-request refusals are the API's business (tests send them on purpose); they are not process alarms.
REQUEST_REFUSAL_PATTERNS = (
    "validation error for",              # pydantic ValidationError re-raised by the worker
    "ValidationError",
    "request exceeds the arena caps",
    "duration_seconds must be",
    "aspect_ratio must be",
    "must be one of",
    "ERROR for task",                    # the per-job error line itself
    "[run] failed after",
)
ALARM_LOG_PATTERN = re.compile(r"critical|TT_FATAL|TT_THROW|Traceback|TIMEOUT|Permission denied|hang detected|Out of Memory|Not enough space", re.I)


def worker_log_alarms(start_offset: int = 0) -> list[str]:
    """Alarming worker-log lines (all ranks) from byte ``start_offset`` on -- process-level signals only:
    known-benign matmul fallbacks are dropped, and a Traceback whose exception (within the next 15 lines)
    is a per-request refusal (pydantic validation, arena caps, duration/aspect policy) is dropped too."""
    if not WORKER_LOG or not os.path.exists(WORKER_LOG):
        return []
    try:
        with open(WORKER_LOG, "rb") as f:
            f.seek(start_offset)
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        if not ALARM_LOG_PATTERN.search(ln) or any(b in ln for b in BENIGN_LOG_PATTERNS):
            continue
        if any(b in ln for b in REQUEST_REFUSAL_PATTERNS):
            continue
        if "Traceback" in ln:
            tail = "\n".join(lines[i:i + 16])
            if any(b in tail for b in REQUEST_REFUSAL_PATTERNS):
                continue
        out.append(ln.strip()[:220])
    return out


def worker_log_size() -> int:
    try:
        return os.path.getsize(WORKER_LOG) if WORKER_LOG and os.path.exists(WORKER_LOG) else 0
    except OSError:
        return 0


# --------------------------------------------------------------------------- worker-log facts

_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def _worker_window(job_id: str) -> list[str]:
    """Rank-0 worker log lines between 'Starting inference for task <id>' and its 'Inference done'/'ERROR'."""
    if not WORKER_LOG or not os.path.exists(WORKER_LOG):
        return []
    try:
        text = Path(WORKER_LOG).read_text(errors="replace")
    except OSError:
        return []
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [ln for ln in text.split("\n") if ln.startswith("[1,0]")]
    start = next((i for i, ln in enumerate(lines) if f"Starting inference for task {job_id}" in ln), None)
    if start is None:
        return []
    end = next((i for i in range(start, len(lines)) if f"for task {job_id}" in lines[i] and ("Inference done" in lines[i] or "ERROR" in lines[i])), len(lines) - 1)
    return lines[start:end + 1]


def _construction_captured() -> bool:
    """True when the CURRENT worker process captured traces before serving anything (rank-0 'capturing
    trace...' / 'Capturing bucket traces' lines ahead of the first 'Starting inference for task'): metal
    >= 56cdeeb9095 binds and captures the whole bucket ladder in the pipeline constructor, so every served
    request is a traced replay and the bind/capture/replay distinction below no longer exists."""
    if not WORKER_LOG or not os.path.exists(WORKER_LOG):
        return False
    try:
        text = Path(WORKER_LOG).read_text(errors="replace")
    except OSError:
        return False
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    for ln in text.split("\n"):
        if not ln.startswith("[1,0]"):
            continue
        if "Starting inference for task" in ln:
            return False
        if "capturing trace" in ln or "Capturing bucket traces" in ln:
            return True
    return False


def enrich_from_worker_log(res: Result) -> None:
    if not res.job_id:
        return
    win = _worker_window(res.job_id)
    if not win:
        return
    m = next((re.search(r"packed sequence \d+ -> (?:bucket )?(\d+)", ln) for ln in win if "packed sequence" in ln), None)
    if m:
        res.rung = int(m.group(1))
    res.captured = any("capturing trace" in ln for ln in win)
    if res.rung is not None and not res.captured:
        res.pre_captured = _construction_captured()
    res.compiled_kernels = sum(1 for ln in win if "BuildKernels | compiled" in ln)
    m = next((re.search(r"(\d+) steps, (?:anchors|references|reference_resize_mode)=", ln) for ln in win if " steps, " in ln), None)
    if m:
        res.steps = int(m.group(1))


# --------------------------------------------------------------------------- the runner


def submit(spec: Spec, assets: live.Assets) -> Result:
    res = Result(spec=spec)
    code, resp = live.http("POST", live.ENDPOINT[spec.task], spec.body(assets), timeout=300)
    res.submit_code = code
    if code in (200, 202) and isinstance(resp, dict) and resp.get("id"):
        res.job_id = resp["id"]
        res.status = "submitted"
    else:
        res.status = f"submit_{code}"
        res.error = str(resp)[:400]
    return res


def run_one(spec: Spec, assets: live.Assets, deployment: live.Deployment, out_dir: Path, *, judge: bool = True,
            delete: bool = True) -> Result:
    t0 = time.time()
    res = submit(spec, assets)
    if not res.job_id:
        res.wall_s = round(time.time() - t0, 1)
        return res
    print(f"  {spec.tag}: job {res.job_id}", flush=True)
    status, job, wall = live._wait_terminal(res.job_id, t0, deployment)
    res.status, res.wall_s = status, wall
    if status == "completed":
        out_dir.mkdir(parents=True, exist_ok=True)
        res.mp4 = out_dir / f"{re.sub(r'[^A-Za-z0-9.+-]+', '_', spec.tag)}.{res.job_id[:8]}.mp4"
        code, r = live.http("GET", f"/v1/videos/generations/{res.job_id}/download", timeout=600, save_to=res.mp4)
        if code == 200 and res.mp4.exists():
            data = res.mp4.read_bytes()
            res.mp4_bytes, res.sha256 = len(data), hashlib.sha256(data).hexdigest()
            if judge:
                res.verdict = media.judge(res.mp4, expect_seconds=spec.seconds, expect_canvas=expected_canvas(spec.aspect))
        else:
            res.status, res.error = "download_failed", f"download -> {code} {r}"
    else:
        res.error = str(job.get("error"))[:400] if isinstance(job, dict) else None
        if status != "timeout":
            deployment.mark_poisoned(f"{spec.tag} ended {status}: {res.error}")
    enrich_from_worker_log(res)
    print(f"  {res.describe()}", flush=True)
    if delete and res.job_id and status in ("completed", "failed", "cancelled"):
        live.http("DELETE", f"/v1/videos/generations/{res.job_id}", timeout=60)
    return res


DEVICE_HANG_PATTERN = re.compile(r"fetch queue wait|potential hang detected|TIMEOUT: device timeout|TT_THROW", re.I)


def device_hung(res: Result) -> bool:
    """A failed result whose error is a device-level timeout/throw: the whole mesh is stuck, not this request."""
    return res.status == "failed" and bool(res.error) and bool(DEVICE_HANG_PATTERN.search(res.error))


def run_sequence(specs: Iterable[Spec], assets: live.Assets, deployment: live.Deployment, report: live.Report | None,
                 out_dir: Path, *, fresh: str | None = None, stop_on_failure: bool = False, judge: bool = True) -> list[Result]:
    """Run ``specs`` in order on one worker process.  ``fresh="t2va"`` starts fresh workers first
    (when the deployment is controllable).  Every result is appended to ``report``."""
    if fresh and deployment.controllable:
        deployment.fresh(fresh)
    results: list[Result] = []
    for spec in specs:
        res = run_one(spec, assets, deployment, out_dir, judge=judge)
        results.append(res)
        if report is not None:
            report.add(**res.row())
        if device_hung(res):
            # The mesh is stuck (every later request in this process would burn the 300 s op timeout
            # and fail the same way -- 2026-09-06 fl2va 1:1 then 3:4 then 9:16): stop here, the
            # deployment is already marked poisoned and the next fresh start resets the chips.
            print(f"  [sequence] device hang on {spec.tag}: skipping the rest of this sequence", flush=True)
            break
        if stop_on_failure and not res.ok:
            break
    return results


def _audio_only(r: Result) -> bool:
    return r.status == "completed" and r.verdict is not None and not r.verdict.ok and all(
        "audio looks like noise" in x for x in r.verdict.reasons)


def replay_flags(results: list[Result]) -> list[bool]:
    """For each result (in order): True when it was a traced REPLAY -- its rung had already been served
    earlier in this process (or was captured at construction, ``pre_captured``) and this request did not
    capture.  Binds and captures are False."""
    seen: set[int] = set()
    flags = []
    for r in results:
        is_replay = r.rung is not None and r.captured is False and (r.rung in seen or r.pre_captured is True)
        flags.append(is_replay)
        if r.rung is not None:
            seen.add(r.rung)
    return flags


def failures(results: list[Result], *, ignore_replay_audio: bool = False) -> list[str]:
    """Descriptions of the results that are not ok.  ``ignore_replay_audio=True`` drops results whose
    ONLY problem is the audio-noise verdict AND that were traced replays (the known rung-replay bug), so
    a long matrix stays strict on status / duration / canvas / video -- and on audio for binds and
    captures, where noise would be a different bug."""
    out = []
    for r, is_replay in zip(results, replay_flags(results)):
        if r.ok:
            continue
        if ignore_replay_audio and is_replay and _audio_only(r):
            continue
        out.append(r.describe())
    return out


def replay_audio_failures(results: list[Result]) -> list[str]:
    return [r.describe() for r, is_replay in zip(results, replay_flags(results)) if is_replay and _audio_only(r)]
