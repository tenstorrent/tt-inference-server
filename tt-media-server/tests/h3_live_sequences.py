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
BUCKET_LADDER = (22528, 31744, 44032, 61440, 86016, 118784)
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


def expected_rung(aspect: str, seconds: float, **kw) -> int | None:
    n = packed_length_estimate(aspect, seconds, **kw)
    return next((r for r in BUCKET_LADDER if n <= r), None)


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
    seed: int = 7
    prompt: str = live.PROMPT
    label: str = ""

    @property
    def tag(self) -> str:
        kf = {(): "", (0,): "+first", (-1,): "+last", (0, -1): "+first+last"}.get(self.keyframes, f"+kf{self.keyframes}")
        refs = f"+{self.ref_images}img" if self.ref_images else ""
        return self.label or f"{self.task} {self.aspect} {self.seconds}s{kf}{refs}"

    def body(self, assets: live.Assets) -> dict:
        b = {"prompt": self.prompt, "seed": self.seed, "aspect_ratio": self.aspect, "duration_seconds": self.seconds}
        if self.task == "fl2va":
            pick = {0: assets.key_first, -1: assets.key_last}
            b["image_prompts"] = [{"image": pick[p], "frame_pos": p} for p in self.keyframes]
        elif self.task == "ref2va":
            b["references"] = {"images": [{"b64": assets.img} for _ in range(max(1, self.ref_images))]}
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

    @property
    def ok(self) -> bool:
        return self.status == "completed" and (self.verdict is None or self.verdict.ok)

    def row(self) -> dict:
        v = self.verdict
        return {
            "combo": self.spec.tag, "task": self.spec.task, "request": "seq", "status": self.status, "wall_s": self.wall_s,
            "job_id": self.job_id, "mp4_bytes": self.mp4_bytes, "sha256": self.sha256, "error": self.error,
            "rung": self.rung, "captured": self.captured, "compiled_kernels": self.compiled_kernels,
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
    res.compiled_kernels = sum(1 for ln in win if "BuildKernels | compiled" in ln)


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
        if stop_on_failure and not res.ok:
            break
    return results


def _audio_only(r: Result) -> bool:
    return r.status == "completed" and r.verdict is not None and not r.verdict.ok and all(
        "audio looks like noise" in x for x in r.verdict.reasons)


def replay_flags(results: list[Result]) -> list[bool]:
    """For each result (in order): True when it was a traced REPLAY -- its rung had already been served
    earlier in this process and this request did not capture.  Binds and captures are False."""
    seen: set[int] = set()
    flags = []
    for r in results:
        is_replay = r.rung is not None and r.rung in seen and r.captured is False
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
