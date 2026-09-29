# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""One generation end to end (submit -> poll -> fetch), retries, resume bookkeeping,
and the per-case verdict. One results.jsonl line per ``run_once`` call."""

from __future__ import annotations

import collections
import csv
import json
import os
import time

from . import models as M


def _attempt_once(
    adapter, combo: str, cfg: dict, case: dict, tag: str, measured: bool, timeout_s: int
) -> dict:
    rid = f"{combo}-{case['id']}-{tag}"
    os.makedirs(os.path.join(M.logs_dir(), combo, case["id"]), exist_ok=True)
    os.makedirs(os.path.join(M.clips_dir(), combo), exist_ok=True)
    rec = {
        "combo": combo, "case": case["id"], "tag": tag, "measured": measured,
        "engine": cfg.get("engine", "minimax-h3 (tt-media-server)"), "hw": cfg.get("hw", "Blackhole Galaxy"),
        "gpus": cfg.get("gpus", 1), "node": cfg.get("node", ""),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "timeout_s": timeout_s, "media": "b64",
        "steps_effective": getattr(adapter, "FIXED_STEPS", None) or M.load_cases()["_fixed"].get("steps"),
    }  # fmt: skip
    t0 = time.time()
    (code, body), task = adapter.post(case)
    rec["submit_http"] = code
    job = body.get("id") if isinstance(body, dict) else None
    if not job:
        msg, raw = M.extract_error(body)
        rec.update(outcome="submit_failed", error=msg, error_message=msg, error_raw=raw,
                   e2e_s=round(time.time() - t0, 2), poll_history=[])  # fmt: skip
        return rec

    rec["job_id"] = job
    t_start = None
    last = None
    transport_errs = 0
    poll_history: list = []
    last_sample_ts = t0
    last_seen = (None, None)
    last_change_ts = t0
    changes_seen = 0
    effective_timeout = timeout_s
    extended = False
    lost_polls = 0
    sbody = None

    def _sample(elapsed, status, progress):
        nonlocal last_sample_ts
        now_ = t0 + elapsed
        if (
            now_ - last_sample_ts >= M.POLL_HISTORY_SAMPLE_S
            and len(poll_history) < M.POLL_HISTORY_MAX
        ):
            poll_history.append([round(elapsed, 1), status, progress])
            last_sample_ts = now_

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed > effective_timeout:
            # Progress-aware patience: a job whose status kept moving gets one 2x extension.
            progressing = (changes_seen > 1) and (now - last_change_ts) < M.STALL_S
            if progressing and not extended and effective_timeout < 2 * timeout_s:
                effective_timeout = 2 * timeout_s
                extended = True
                M.log(
                    f"{rid}: still progressing at the {timeout_s}s estimate -- extending to {effective_timeout}s (cap 2x)"
                )
                continue
            rec.update(outcome="timeout", last_status=last, poll_history=poll_history,
                       timeout_extended=extended, effective_timeout_s=int(effective_timeout))  # fmt: skip
            return rec

        code, st, sbody = adapter.status(task, job)
        now = time.time()
        elapsed = now - t0
        if code == 0:
            transport_errs += 1
            rec["transport_errors"] = transport_errs
            _sample(elapsed, "transport_error", None)
            if transport_errs > M.TRANSPORT_POLLS:
                msg = f"endpoint unreachable for {transport_errs} consecutive polls (last known status {last!r})"
                rec.update(outcome="unreachable", failure_class="transport", error=msg, error_message=msg,
                           last_status=last, poll_history=poll_history)  # fmt: skip
                return rec
            time.sleep(M.POLL_S)
            continue
        transport_errs = 0
        if st is None and code in (401, 403, 404):
            lost_polls += 1
            _sample(elapsed, f"lost:{code}", None)
            if lost_polls >= M.LOST_POLLS:
                msg = (f"job record lost: GET status answered HTTP {code} with no status "
                       f"{lost_polls} polls in a row (last known status {last!r})")  # fmt: skip
                rec.update(outcome="failed", failure_class="lost", last_status=last, error=msg, error_message=msg,
                           poll_history=poll_history, queue_s=round((t_start or time.time()) - t0, 2))  # fmt: skip
                return rec
            time.sleep(M.POLL_S)
            continue
        lost_polls = 0
        last = st
        progress = (sbody or {}).get("progress") if isinstance(sbody, dict) else None
        _sample(elapsed, st, progress)
        seen = (st, progress)
        if st is not None and seen != last_seen:
            last_change_ts = now
            changes_seen += 1
            last_seen = seen
        if st and st not in M.QUEUED_STATUSES and t_start is None:
            t_start = time.time()
        if st in ("completed", "succeeded"):
            break
        if st in ("failed", "cancelled"):
            msg, raw = M.extract_error(sbody)
            rec.update(outcome="failed", last_status=st, error=msg, error_message=msg, error_raw=raw,
                       queue_s=round((t_start or time.time()) - t0, 2), poll_history=poll_history)  # fmt: skip
            return rec
        never_started = st in M.QUEUED_STATUSES and t_start is None
        if never_started and (now - last_change_ts) >= M.STALL_S:
            msg = f"job still {st!r} after {M.STALL_S:.0f}s -- the endpoint never started it"
            rec.update(outcome="failed", failure_class="stalled", last_status=st, error=msg, error_message=msg,
                       poll_history=poll_history, queue_s=round(time.time() - t0, 2))  # fmt: skip
            return rec
        time.sleep(M.POLL_S)

    t_done = time.time()
    rec["api_exposes_progress"] = t_start is not None
    if t_start is None:
        t_start = t0
    dest = os.path.join(M.clips_dir(), combo, f"{case['id']}-{tag}.mp4")
    ok, content_code = adapter.content(task, job, dest)
    rec["content_http"] = content_code
    t_content = time.time()
    rec.update(
        outcome="ok" if ok else "content_failed",
        queue_s=round(t_start - t0, 2),
        gen_s=round(t_done - t_start, 2),
        e2e_s=round(t_content - t0, 2),
        engine_inference_s=(sbody or {}).get("inference_time_s")
        if isinstance(sbody, dict)
        else None,
        poll_history=poll_history,
    )
    if ok:
        rec.update(out_file=dest, out_sha256=M.sha256(dest)[:16], out_bytes=os.path.getsize(dest),
                   mvhd_duration_s=M.mvhd_duration(dest), has_audio=M.has_audio(dest))  # fmt: skip
    return rec


def run_once(
    adapter,
    combo: str,
    cfg: dict,
    case: dict,
    tag: str,
    measured: bool,
    timeout_table: str,
) -> dict:
    """One clip, retried only when the failure is transient (see ``models.is_transient``)."""
    timeout_s = M.case_timeout_s(case, timeout_table)
    rid = f"{combo}-{case['id']}-{tag}"
    attempts_log = []
    attempt = 0
    while True:
        attempt += 1
        rec = _attempt_once(adapter, combo, cfg, case, tag, measured, timeout_s)
        rec["failure_class"] = rec.get("failure_class") or M.classify_failure(rec)
        attempts_log.append({"attempt": attempt, "outcome": rec["outcome"], "failure_class": rec["failure_class"],
                             "error_message": rec.get("error_message")})  # fmt: skip
        if rec["outcome"] == "ok" or not M.is_transient(rec) or attempt > M.MAX_RETRIES:
            break
        wait = M.RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
        M.log(
            f"{rid}: attempt {attempt} transient {rec['outcome']} ({rec.get('error_message')}) -- retrying in {wait:.0f}s"
        )
        time.sleep(wait)
    rec["attempt"] = attempt
    rec["attempts"] = attempts_log
    return finish(rec, rid, case, combo)


def primary_metric(rec: dict):
    """Engine-reported inference time when exposed, else gen_s (queue excluded), else e2e_s."""
    value = rec.get("engine_inference_s")
    if value is not None:
        return float(value)
    gen = rec.get("gen_s")
    return float(gen) if gen is not None else rec.get("e2e_s")


def _append_csv(rec: dict, case: dict) -> None:
    eff = rec.get("steps_effective")
    row = {"primary_metric_s": primary_metric(rec), "steps_requested": case.get("steps"),
           "steps_effective": "" if eff is None else eff}  # fmt: skip
    path = M.results_csv_path()
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=M.CSV_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, rec.get(k, "")) for k in M.CSV_COLUMNS})


def finish(rec: dict, rid: str, case: dict, combo: str) -> dict:
    with open(os.path.join(M.logs_dir(), combo, case["id"], f"{rid}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    with open(M.results_path(), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    _append_csv(rec, case)
    extra = ""
    if rec["outcome"] != "ok":
        extra = f" class={rec.get('failure_class')} attempt={rec.get('attempt')} err={rec.get('error_message')!r}"
    M.log(
        f"{rid}: {rec['outcome']} gen={rec.get('gen_s')}s queue={rec.get('queue_s')}s dur={rec.get('mvhd_duration_s')}{extra}"
    )
    return rec


# ---------------------------------------------------------------- resume
def read_results(path: str | None = None) -> list:
    path = path or M.results_path()
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def existing_ok_counts(combo: str) -> dict:
    counts: dict = {}
    for r in read_results():
        if r.get("combo") == combo and r.get("measured") and r.get("outcome") == "ok":
            counts[r["case"]] = counts.get(r["case"], 0) + 1
    return counts


def existing_tags(combo: str) -> dict:
    tags: dict = {}
    for r in read_results():
        if r.get("combo") == combo:
            tags.setdefault(r.get("case"), set()).add(r.get("tag"))
    return tags


def next_free_tags(used_tags, n: int) -> list:
    out = []
    i = 1
    while len(out) < n:
        tag = f"r{i}"
        if tag not in used_tags:
            out.append(tag)
        i += 1
    return out


def newest_ok_rows(combo: str, case_id: str, n: int) -> list:
    rows = [r for r in read_results()
            if r.get("combo") == combo and r.get("case") == case_id and r.get("measured") and r.get("outcome") == "ok"]  # fmt: skip
    return rows[-n:] if n else []


def case_verdict(
    combo: str, case: dict, runs: int, have: int, warm: dict | None, done: list
) -> int:
    """rc 0 = every measured run asked for; 1 = some; 2 = none. One summary line per case."""
    ok = [r for r in done if r.get("outcome") == "ok"]
    bad = [r for r in done if r.get("outcome") != "ok"]
    total_ok = have + len(ok)
    detail = ""
    if bad:
        classes = collections.Counter(
            r.get("failure_class") or r.get("outcome") for r in bad
        )
        top, n = classes.most_common(1)[0]
        msg = next((r.get("error_message") for r in bad
                    if (r.get("failure_class") or r.get("outcome")) == top and r.get("error_message")), "")  # fmt: skip
        detail = f" | {len(bad)} failed, mostly {top} x{n}" + (
            f": {msg}" if msg else ""
        )
    if warm and warm.get("outcome") != "ok":
        detail += f" | warmup also {warm.get('outcome')}"
    verdict = "OK" if total_ok >= runs else ("PARTIAL" if total_ok else "NO DATA")
    M.log(
        f"{combo} {case['id']}: {verdict} -- {total_ok}/{runs} measured runs succeeded{detail}"
    )
    return 0 if total_ok >= runs else (1 if total_ok else 2)


def why(rec: dict) -> str:
    msg = rec.get("error_message")
    if msg:
        return str(msg)
    if rec.get("outcome") == "timeout":
        budget = rec.get("effective_timeout_s") or rec.get("timeout_s")
        if rec.get("last_status") is None:
            return f"job {rec.get('job_id')}: record vanished (polls answered no status) within {budget}s"
        return (
            f"job {rec.get('job_id')} still {rec.get('last_status')!r} after {budget}s"
        )
    return str(rec.get("outcome"))
