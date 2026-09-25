# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Everything about the deployment under test that is not one generation: discovery
and gate probes, the health pre-gate, the per-host stop/strike state, cancelling the
jobs this process left behind, waiting for an idle endpoint, and one case end to end
(warmup + N measured runs, judged)."""

from __future__ import annotations

import os
import re
import statistics
import time
from dataclasses import dataclass, field

from . import adapters as A
from . import judge as J
from . import models as M
from . import runner as R

MAX_STRIKES = 3
IDLE_WAIT_S = int(os.environ.get("H3_IDLE_WAIT_S", "60"))
STALE_SUCCESS_S = 1800  # a deployment whose newest generation failed and that has not succeeded for this long
TERMINAL = {"completed", "failed", "cancelled"}
NIL_JOB = "00000000-0000-4000-8000-000000000000"


def resolve_assets_dir(explicit: str | None = None) -> str:
    """The preferred pack: an explicit path, else the first staged directory that exists
    (``H3_ASSETS``, the shared CI volume, ``$PERSISTENT_VOLUME_ROOT/h3-assets``,
    ``/localdev/persistent-volume/h3-assets``), else the in-repo pack. Files are looked up
    per name through every one of them in that order (``models.asset_path``) and pinned
    against the repo manifest, so a partially staged directory neither hides nor re-pins
    anything."""
    for candidate in (explicit, *M.staged_asset_dirs()):
        if candidate and os.path.isdir(candidate):
            return candidate
    return M.REPO_ASSETS


def short_name(url: str) -> str:
    host = re.sub(r"^https?://", "", url).split("/")[0]
    name = host.split(":")[0]
    if not re.fullmatch(r"[\d.]+", name) and ":" not in name:
        name = name.split(".")[0]
    if ":" in host:
        name += "-" + host.rsplit(":", 1)[1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "host"


@dataclass
class Endpoint:
    task: str
    url: str
    name: str = ""
    reachable: bool = False
    error: str = ""
    health: int = 0
    routes: set = field(default_factory=set)
    liveness: dict = field(default_factory=dict)
    liveness_status: int = 0
    not_ready: str = ""
    auth_required: bool | None = None
    key_ok: bool | None = None
    key_detail: str = ""
    jobs_ok: bool = False
    unauth_jobs_code: int = 0  # GET /v1/videos/jobs without a key
    gates: dict = field(default_factory=dict)
    gate_detail: dict = field(default_factory=dict)
    stopped: str = ""
    strikes: int = 0

    def __str__(self) -> str:
        return f"{self.task}-{self.name}"

    def stop(self, why: str) -> None:
        self.stopped = why

    def strike(self, n: int = 1) -> int:
        self.strikes += n
        return self.strikes

    def clear_strikes(self) -> None:
        self.strikes = 0


def _detail(body):
    return body.get("detail") if isinstance(body, dict) else None


def discover(adapter: A.TenstorrentH3, task: str) -> Endpoint:
    """Cheap GETs plus gate probes that can never start work: an empty, unauthenticated
    POST names the gate (401 = open behind auth, a pydantic list = open with nothing in
    front, a str detail = refused, 405/503 str = not ready)."""
    url = adapter.base_url(task)
    ep = Endpoint(task=task, url=url, name=short_name(url))
    code, body = adapter.get(task, "/health", key=False)
    ep.health = code
    if code == 0:
        ep.error = M.extract_error(body)[0]
        return ep
    ep.reachable = True
    code, spec = adapter.get(task, "/openapi.json", key=False, timeout=60)
    if code == 200 and isinstance(spec, dict):
        ep.routes = set(spec.get("paths", {}))
    code, live = adapter.get(task, "/tt-liveness", key=False)
    ep.liveness_status = code
    if isinstance(live, dict):
        ep.liveness = live
    probes = {}
    for t in A.TASKS:
        code, body = adapter.post_json(task, A.ROUTE[t], b"{}", key=False)
        probes[t] = (code, _detail(body))
    saw_auth = any(code in (401, 403) for code, _ in probes.values())
    saw_schema = any(isinstance(detail, list) for _, detail in probes.values())
    warming = all(
        code in (405, 503) and isinstance(detail, str)
        for code, detail in probes.values()
    )
    if saw_auth:
        ep.auth_required = True
    elif saw_schema:
        ep.auth_required = False
    elif warming:
        ep.auth_required = None
        ep.not_ready = "; ".join(sorted({f"{c} {d}" for c, d in probes.values()}))[:200]
    ep.unauth_jobs_code = adapter.get(task, A.JOBS, key=False)[0]
    if ep.auth_required is None and not warming:
        ep.auth_required = ep.unauth_jobs_code in (401, 403)
    for t, (code, detail) in probes.items():
        if code in (401, 403) or isinstance(detail, list):
            state = "open"
        elif code in (405, 503) and isinstance(detail, str):
            state = "not-ready"
        elif code == 422 and isinstance(detail, str):
            state = "refused"
        elif ep.auth_required is False and isinstance(detail, str):
            state = "refused"
        else:
            state = "unknown"
        ep.gates[t] = state
        ep.gate_detail[t] = f"{code} {str(detail)[:160]}"
    code, body = adapter.get(task, A.JOBS)
    ep.jobs_ok = code == 200 and isinstance(body, list)
    if ep.not_ready:
        ep.key_ok, ep.key_detail = (
            None,
            f"not verifiable, the deployment is not ready ({ep.not_ready[:80]})",
        )
    elif not ep.auth_required:
        ep.key_ok, ep.key_detail = True, "no auth in front of this deployment"
    elif ep.jobs_ok:
        ep.key_ok, ep.key_detail = True, "ok"
    elif code in (401, 403):
        ep.key_ok, ep.key_detail = False, f"{code} {M.extract_error(body)[0][:120]}"
    else:
        code2, body2 = adapter.get(task, f"/v1/videos/generations/{NIL_JOB}")
        if code2 == 404:
            ep.key_ok, ep.key_detail = True, f"ok (jobs listing -> {code})"
        elif code2 in (401, 403):
            ep.key_ok, ep.key_detail = (
                False,
                f"{code2} {M.extract_error(body2)[0][:120]}",
            )
        else:
            ep.key_ok, ep.key_detail = (
                True,
                f"unverified (jobs -> {code}, job lookup -> {code2})",
            )
    return ep


def device_block(ep: Endpoint) -> str:
    """Why this endpoint cannot take a job from us now, or '' when it can."""
    if not ep.reachable:
        return f"{ep.url} unreachable: {ep.error}"
    if ep.not_ready:
        return f"{ep.name} is not ready: {ep.not_ready}"
    if ep.auth_required and not ep.key_ok:
        return f"{ep.name} rejects the configured key: {ep.key_detail}"
    if ep.stopped:
        return f"{ep}: this host's device tests are stopped -- {ep.stopped}"
    return ""


def probe_checks(ep: Endpoint) -> list:
    """The read-only contract checks (the benchmark's probe level). [] = clean."""
    problems = []
    if ep.health != 200:
        problems.append(f"GET /health -> {ep.health}")
    if ep.liveness_status != 200:
        problems.append(f"GET /tt-liveness -> {ep.liveness_status}")
    elif "model_ready" in ep.liveness and ep.liveness.get("model_ready") is not True:
        problems.append(f"/tt-liveness model_ready={ep.liveness.get('model_ready')!r}")
    if ep.routes:
        for route in (
            A.ROUTE[ep.task],
            "/v1/videos/generations/{job_id}",
            "/v1/videos/generations/{job_id}/download",
        ):
            if route not in ep.routes:
                problems.append(f"openapi.json does not list {route}")
    if ep.gates.get(ep.task) == "refused":
        problems.append(
            f"the deployment refuses {ep.task}: {ep.gate_detail.get(ep.task)}"
        )
    if ep.auth_required and ep.key_ok is False:
        problems.append(f"API key rejected: {ep.key_detail}")
    if ep.auth_required and ep.unauth_jobs_code in (200,):
        problems.append(
            "GET /v1/videos/jobs without a key answered 200: the listing is not behind auth"
        )
    return problems


def _metric(text: str, name: str, labels: str = "") -> float | None:
    """The newest value of a Prometheus series: every ``name`` line whose label set holds the
    whole ``label="value"`` pair (any line when no label is asked), max() of them. The
    last-generation timestamps carry one series per request_type, and the first line in
    the exposition is not the newest."""
    values = []
    for m in re.finditer(
        rf"^{re.escape(name)}(?:\{{([^}}]*)\}})?\s+([-+0-9.eE]+)", text, re.M
    ):
        pairs = [p.strip() for p in (m.group(1) or "").split(",") if p.strip()]
        if labels and labels not in pairs:
            continue
        try:
            values.append(float(m.group(2)))
        except ValueError:
            continue
    return max(values) if values else None


def pregate(
    adapter: A.TenstorrentH3, ep: Endpoint, idle_wait_s: int = IDLE_WAIT_S
) -> list:
    """Is the deployment fit to be measured? Read-only: liveness says ready with an
    empty queue, no foreign job in flight (bounded wait), the canary is not dead, and
    the newest generation did not fail without a success in the last 30 minutes."""
    problems = []
    code, fresh = adapter.get(ep.task, "/tt-liveness", key=False)
    live = fresh if code == 200 and isinstance(fresh, dict) else (ep.liveness or {})
    if live.get("model_ready") is False:
        problems.append("model_ready is false")
    if isinstance(live.get("queue_size"), (int, float)) and live["queue_size"] > 0:
        problems.append(
            f"queue_size is {live['queue_size']} (expected an idle endpoint)"
        )
    if ep.jobs_ok:
        t0 = time.time()
        while True:
            code, jobs = adapter.get(ep.task, A.JOBS)
            busy = (
                [
                    j
                    for j in jobs
                    if isinstance(j, dict) and j.get("status") not in TERMINAL
                ]
                if code == 200 and isinstance(jobs, list)
                else []
            )
            if not busy:
                break
            if time.time() - t0 > idle_wait_s:
                problems.append(
                    f"{len(busy)} foreign job(s) still in flight after {idle_wait_s}s"
                )
                break
            M.log(f"  [idle] {ep}: {len(busy)} foreign job(s) in flight -- waiting")
            time.sleep(10)
    code, metrics = adapter.get(ep.task, "/metrics", key=False, timeout=20)
    if code == 200 and isinstance(metrics, dict) and metrics.get("error") == "non-json":
        text = metrics.get("body", "")
        # Prometheus text: only the head is returned by http(); fetch the whole thing.
        code, raw = A.http("GET", f"{ep.url}/metrics", timeout=20, raw=True)
        if code == 200 and isinstance(raw, (bytes, bytearray)):
            text = raw.decode("utf-8", "replace")
        if _metric(text, "tt_canary_state", 'state="dead"') == 1:
            problems.append("canary state is dead")
        success = _metric(
            text, "tt_media_server_video_last_generation_timestamp", 'status="success"'
        )
        failure = _metric(
            text, "tt_media_server_video_last_generation_timestamp", 'status="failure"'
        )
        if (
            failure
            and success is not None
            and failure > success
            and time.time() - success > STALE_SUCCESS_S
        ):
            problems.append(
                "the newest generation failed and none succeeded in the last 30 minutes"
            )
    return problems


class JobTracker:
    """Every job this process submitted, so teardown can cancel what it left live."""

    def __init__(self, adapter: A.TenstorrentH3):
        self.adapter = adapter
        self.jobs: dict = {}
        adapter.on_submit = self.record

    def record(self, task: str, job_id: str) -> None:
        self.jobs[job_id] = {"task": task, "terminal": False}

    def mark(self, rec: dict) -> None:
        job = rec.get("job_id")
        if job:
            meta = self.jobs.setdefault(
                job, {"task": rec.get("task", "t2va"), "terminal": False}
            )
            meta["terminal"] = (
                rec.get("outcome") == "ok" or rec.get("last_status") in TERMINAL
            )

    def reconcile(self) -> None:
        for jid, meta in list(self.jobs.items()):
            if meta["terminal"]:
                continue
            code, body = self.adapter.get(
                meta["task"], f"/v1/videos/generations/{jid}", timeout=20
            )
            status = body.get("status") if isinstance(body, dict) else None
            if code == 200 and status in TERMINAL:
                meta["terminal"] = True
                continue
            if code == 0:
                continue
            meta["terminal"] = self.adapter.cancel(meta["task"], jid) != 0

    def cancel_leftovers(self) -> list:
        for attempt in range(3):
            pending = [j for j, m in self.jobs.items() if not m["terminal"]]
            if not pending:
                return []
            M.log(
                f"  [cleanup] {len(pending)} job(s) this run may have left live -- checking"
            )
            try:
                self.reconcile()
            except Exception as exc:  # noqa: BLE001 - teardown must not mask the run's result
                M.log(f"  [cleanup] failed: {type(exc).__name__}: {exc}")
            if attempt < 2 and any(not m["terminal"] for m in self.jobs.values()):
                time.sleep(5)
        return [j for j, m in self.jobs.items() if not m["terminal"]]


def after_run(
    adapter: A.TenstorrentH3,
    tracker: JobTracker,
    ep: Endpoint,
    rec: dict,
    keep_going: bool = False,
) -> str:
    """Bookkeeping after one run_once(): a reason to stop this host, or ''.

    A job still generating when the budget ran out wedges the HOST (unless keep_going);
    a job that never left queued or whose record vanished is a strike; device-trouble
    text is a strike per attempt; three strikes stop the host; a synchronous
    deployment stops it at once. A validation or capability refusal is none of these.
    """
    tracker.mark(rec)
    tracker.reconcile()
    outcome, job, last = rec.get("outcome"), rec.get("job_id"), rec.get("last_status")
    if outcome == "ok":
        ep.clear_strikes()
        return ""
    msg = str(rec.get("error_message") or "")
    shape = f"{rec['case']}-{rec['tag']}"
    if rec.get("failure_class") == "client_capability":
        if "synchronous deployment" in msg:
            ep.stop(msg)
            M.log(f"  [stop] {ep}: {msg}")
            return f"contract mismatch: {msg}"
        return ""
    if outcome in ("submit_failed", "unreachable") and (
        rec.get("submit_http") == 0 or rec.get("failure_class") == "transport"
    ):
        healthy = adapter.get(ep.task, "/health", key=False, timeout=15)[0] == 200
        if not healthy:
            ep.stop(f"host unreachable: {shape}: {R.why(rec)}; GET /health fails too")
            M.log(f"  [stop] {ep}: {ep.stopped}")
            return ep.stopped
        if "timed out" in msg.lower():
            ep.stop(
                f"{shape}: the POST blocked for its whole budget without a 202 while /health answers"
            )
            M.log(f"  [stop] {ep}: {ep.stopped}")
            return ep.stopped
        n = ep.strike()
        M.log(
            f"  [strike {n}/{MAX_STRIKES}] {ep}: {shape}: {R.why(rec)} (the host answers /health)"
        )
        if n >= MAX_STRIKES:
            ep.stop(f"{n} transport failures in a row, last: {R.why(rec)!r}")
            return ep.stopped
        return ""
    if rec.get("failure_class") == "lost":
        n = ep.strike()
        M.log(
            f"  [strike {n}/{MAX_STRIKES}] {ep}: {shape}: {R.why(rec)} -- not a mesh hang"
        )
        if n >= MAX_STRIKES:
            ep.stop(f"{n} failures in a row, last: {R.why(rec)!r}")
            return ep.stopped
        return ""
    if outcome == "timeout":
        budget = rec.get("effective_timeout_s") or rec.get("timeout_s")
        if (
            rec.get("transport_errors")
            and adapter.get(ep.task, "/health", key=False, timeout=15)[0] != 200
        ):
            ep.stop(
                f"host unreachable: {shape} job {job} last seen {last!r}; the polls failed and GET /health fails too"
            )
            M.log(f"  [stop] {ep}: {ep.stopped}")
            return ep.stopped
        if last is None or last in M.QUEUED_STATUSES:
            what = (
                "the job record vanished"
                if last is None
                else f"still {last!r} -- the endpoint never started it"
            )
            n = ep.strike()
            M.log(
                f"  [strike {n}/{MAX_STRIKES}] {ep}: {shape} job {job}: {what} after {budget}s -- not a mesh hang"
            )
            if n >= MAX_STRIKES:
                ep.stop(f"{n} failures in a row, last: {what}")
                return ep.stopped
            return ""
        why = (
            f"{shape} job {job} still {last!r} after {budget}s (2x extension included)"
        )
        if keep_going:
            M.log(
                f"  [wedge] {ep}: {why} -- continue_after_timeout is set, carrying on"
            )
            return ""
        ep.stop(f"wedged: {why}")
        M.log(f"  [wedge] {ep}: {why} -- skipping this host's remaining device tests")
        return f"endpoint wedged: {why}"
    hits = [a for a in (rec.get("attempts") or [rec]) if M.is_device_trouble(a)]
    if hits or rec.get("failure_class") == "stalled":
        n = ep.strike(len(hits) or 1)
        M.log(f"  [strike {n}/{MAX_STRIKES}] {ep}: {shape}: {R.why(rec)}")
        if n >= MAX_STRIKES:
            ep.stop(f"{n} device-trouble failures in a row, last: {R.why(rec)!r}")
            return ep.stopped
    return ""


def smoke_case(task: str, cases_cfg: dict) -> dict:
    """The shortest real generation for one task: 5 s, minimum prompt, minimum inputs."""
    if task == "t2va":
        case = dict(cases_cfg["smoke"])
        case.setdefault("id", "SMOKE")
        return case
    base = {"task": task, "duration_s": 5, "steps": 8, "seed": 42, "aspect_ratio": "16:9",
            "prompt": "prompt_min.txt", "images": ["img_min_256px_scientist.jpg"]}  # fmt: skip
    if task == "fl2va":
        return dict(base, id="SMOKE-FL2VA")
    return dict(
        base,
        id="SMOKE-REF2VA",
        videos=["vid_min_2s_city_skyline.mp4"],
        audios=["aud_min_2s_score.wav"],
    )


def bench_case(
    adapter,
    tracker,
    ep,
    combo,
    cfg,
    case,
    runs,
    timeout_table,
    force=False,
    keep_going=False,
) -> dict:
    """Warmup + N measured runs with a stop after a wedge or three strikes. Resume: only
    the runs still missing from results.jsonl are made; tags are never reused."""
    counts = R.existing_ok_counts(combo)
    tags = R.existing_tags(combo)
    have = 0 if force else counts.get(case["id"], 0)
    if have >= runs:
        M.log(
            f"{combo} {case['id']}: resume -- already has {have}/{runs} successful measured runs"
        )
        return {
            "resumed": True,
            "have": have,
            "rc": 0,
            "warm": None,
            "done": [],
            "stop": "",
        }
    used = tags.get(case["id"], set())
    wtag = next(
        t for t in ["warmup"] + [f"warmup{i}" for i in range(2, 1000)] if t not in used
    )
    warm = R.run_once(
        adapter, combo, cfg, case, wtag, measured=False, timeout_table=timeout_table
    )
    stop = after_run(adapter, tracker, ep, warm, keep_going)
    done = []
    if not stop:
        for tag in R.next_free_tags(used, runs - have):
            rec = R.run_once(
                adapter,
                combo,
                cfg,
                case,
                tag,
                measured=True,
                timeout_table=timeout_table,
            )
            done.append(rec)
            stop = after_run(adapter, tracker, ep, rec, keep_going)
            if stop:
                break
    rc = R.case_verdict(combo, case, runs, have, warm, done)
    return {
        "resumed": False,
        "rc": rc,
        "warm": warm,
        "done": done,
        "have": have,
        "stop": stop,
    }


def case_result(combo: str, case: dict, runs: int, outcome: dict) -> dict:
    """The verdict for one case, with every counting clip judged. The rows that count are
    the ones THIS call produced; only a resumed or partially resumed case (CLI use, never
    the spec test) reads the newest ok rows back from results.jsonl."""
    problems, notes = [], []
    done = outcome.get("done") or []
    warm = outcome.get("warm")
    if outcome.get("resumed") or outcome.get("have"):
        rows = R.newest_ok_rows(combo, case["id"], runs)
    else:
        rows = [r for r in done if r.get("outcome") == "ok"]
    for rec in rows:
        p, n = J.run_problems(rec, case)
        problems += [f"{rec.get('tag')}: {x}" for x in p]
        notes += [f"{rec.get('tag')}: {x}" for x in n]
    bad = [r for r in ([warm] if warm else []) + done if r.get("outcome") != "ok"]
    times = sorted(R.primary_metric(r) for r in rows if R.primary_metric(r) is not None)
    xfail = (
        bool(bad)
        and all(r.get("failure_class") == "client_capability" for r in bad)
        and not any(r.get("outcome") == "ok" for r in done)
        and not any("synchronous deployment" in R.why(r) for r in bad)
    )
    status = (
        "xfail"
        if xfail
        else "pass"
        if outcome.get("rc") == 0 and not problems and not outcome.get("stop")
        else "fail"
    )
    return {
        "case": case["id"], "task": case["task"], "duration_s": case["duration_s"], "runs_requested": runs,
        "runs_ok": outcome.get("have", 0) + sum(1 for r in done if r.get("outcome") == "ok"),
        "resumed": outcome.get("resumed", False), "status": status, "stop": outcome.get("stop") or "",
        "problems": problems, "notes": notes,
        "median_s": statistics.median(times) if times else None, "min_s": times[0] if times else None,
        "max_s": times[-1] if times else None,
        "failures": [{"tag": r.get("tag"), "outcome": r.get("outcome"), "failure_class": r.get("failure_class"),
                      "error": R.why(r)} for r in bad],
    }  # fmt: skip
