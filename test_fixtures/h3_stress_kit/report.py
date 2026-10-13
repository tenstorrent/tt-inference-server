#!/usr/bin/env python3
"""Render results/<run>/ into report.md (+ summary.json). Runs automatically at the end of
h3_stress.py; re-run by hand to re-render:

    python3 report.py results/20261010_120000
"""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def stats(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    if not vals:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {"n": len(vals), "p50": pct(vals, 0.5), "p95": pct(vals, 0.95), "max": max(vals), "mean": statistics.fmean(vals)}


def pass_rate(recs):
    judged = [r for r in recs if r.get("pass") is not None]
    if not judged:
        return None, 0, 0
    ok = sum(1 for r in judged if r["pass"])
    return ok / len(judged), ok, len(judged)



def hammer_split(hs, rows):
    """Split hammer poll codes into 200, 404 on a job the harness itself had already deleted, and anything else."""
    deleted = {r.get("job_id") for r in rows if r.get("kind") == "request" and r.get("delete_http") == 200}
    ok = gone = other = 0
    for h in hs:
        for c, v in h["codes"].items():
            c = str(c)
            if c == "200":
                ok += v
            elif c == "404-after-delete" or (c == "404" and h.get("job_id") in deleted):
                gone += v
            else:
                other += v
    return ok, gone, other

def render(out_dir):
    run = json.load(open(os.path.join(out_dir, "run.json")))
    rows = load_jsonl(os.path.join(out_dir, "results.jsonl"))
    mon = load_jsonl(os.path.join(out_dir, "monitor.jsonl"))
    reqs = [r for r in rows if r.get("kind") == "request"]
    servers = list(run["servers_start"].keys())
    L = []
    summary = {"run": run["run"], "servers": {}}

    L.append(f"# MiniMax-H3 stress test — run {run['run']}")
    L.append("")
    L.append(f"Started {run['started_at']}, finished {run.get('finished_at', '?')} "
             f"({run.get('elapsed_min', '?')} min wall clock, budget {run['args']['minutes']:g} min). "
             f"Sustain workers {run['args']['workers']} (fl2va,ref2va), burst sizes {run['args']['burst']}, "
             f"poll interval {run['args']['poll_interval']} s, job timeout {run['args']['job_timeout']:.0f} s.")
    L.append("")
    L.append("| Server | URL | Model | Worker start (epoch) | restart_count start → end | error_count start → end |")
    L.append("|---|---|---|---|---|---|")
    for s in servers:
        a = run["servers_start"][s]
        b = (run.get("servers_end") or {}).get(s, {})
        la, lb = a.get("liveness", {}), b.get("liveness", {})
        L.append(f"| {s} | {a['url']} | {a.get('model')} | {la.get('worker_start_time')} → {lb.get('worker_start_time')} | "
                 f"{la.get('restart_count')} → {lb.get('restart_count')} | {la.get('error_count')} → {lb.get('error_count')} |")
    L.append("")

    # ---------------------------------------------------------------- headline
    L.append("## Headline")
    L.append("")
    L.append("| Server | Requests | Expected-ok sent | Passed | Pass rate | Failed jobs | Timeouts | 5xx | Transport errors | Jobs/min (sustain) | MB downloaded |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in servers:
        rs = [r for r in reqs if r["server"] == s]
        ok_sent = [r for r in rs if r["expect"] == "ok"]
        rate, ok, judged = pass_rate(ok_sent)
        sus = [r for r in rs if r["phase"] == "sustain" and r["expect"] == "ok"]
        sus_min = None
        if len(sus) >= 2:
            ts = sorted(r["submitted_at"] for r in sus)
            from datetime import datetime
            span = (datetime.fromisoformat(ts[-1]) - datetime.fromisoformat(ts[0])).total_seconds() / 60
            sus_min = len(sus) / span if span > 0 else None
        failed_jobs = sum(1 for r in ok_sent if r.get("status_final") == "failed")
        timeouts = sum(1 for r in ok_sent if r.get("status_final") in ("timeout", "lost"))
        fivexx = sum(1 for r in rs if (r.get("http_code") or 0) >= 500)
        transport = sum(1 for r in rs if r.get("http_code") == 0)
        mb = sum(r.get("bytes") or 0 for r in rs) / 1e6
        L.append(f"| {s} | {len(rs)} | {len(ok_sent)} | {ok} | {f(rate * 100 if rate is not None else None)}% | {failed_jobs} | {timeouts} | {fivexx} | {transport} | {f(sus_min, 2)} | {mb:.0f} |")
        summary["servers"][s] = {"requests": len(rs), "ok_sent": len(ok_sent), "passed": ok, "pass_rate": rate, "failed_jobs": failed_jobs,
                                 "timeouts": timeouts, "http_5xx": fivexx, "transport_errors": transport, "sustain_jobs_per_min": sus_min, "mb": mb}
    L.append("")

    # ----------------------------------------------------------------- phases
    L.append("## Latency by phase (expected-ok requests that completed)")
    L.append("")
    L.append("queue = accepted → first non-queued status; gen = that → completed; e2e = POST sent → MP4 verified (includes upload and download). Seconds.")
    L.append("")
    L.append("| Server | Phase | n | pass | queue p50 / p95 | gen p50 / p95 / max | e2e p50 / p95 | max in-flight at submit |")
    L.append("|---|---|---|---|---|---|---|---|")
    phases = ["baseline", "sustain", "burst", "recovery", "edge"]
    summary["phases"] = {}
    for s in servers:
        for p in phases:
            rs = [r for r in reqs if r["server"] == s and r["phase"] == p and r["expect"] == "ok"]
            if not rs:
                continue
            done = [r for r in rs if r.get("status_final") == "completed"]
            rate, ok, judged = pass_rate(rs)
            q, g, e = stats(done, "queue_s"), stats(done, "gen_s"), stats(done, "e2e_s")
            infl = max((r.get("inflight_at_submit") or 0) for r in rs)
            L.append(f"| {s} | {p} | {len(rs)} | {ok}/{judged} | {f(q['p50'])} / {f(q['p95'])} | {f(g['p50'])} / {f(g['p95'])} / {f(g['max'])} | {f(e['p50'])} / {f(e['p95'])} | {infl} |")
            summary["phases"][f"{s}/{p}"] = {"n": len(rs), "pass": ok, "judged": judged, "queue": q, "gen": g, "e2e": e, "max_inflight": infl}
    L.append("")

    # --------------------------------------------------- baseline vs recovery
    L.append("## Baseline vs recovery (same cases, serial, before and after the load)")
    L.append("")
    L.append("| Server | Case | gen baseline | gen recovery | Δ | e2e baseline | e2e recovery |")
    L.append("|---|---|---|---|---|---|---|")
    degraded = []
    for s in servers:
        base = {r["case_id"]: r for r in reqs if r["server"] == s and r["phase"] == "baseline" and r.get("status_final") == "completed"}
        rec = {r["case_id"]: r for r in reqs if r["server"] == s and r["phase"] == "recovery" and r.get("status_final") == "completed"}
        for cid in base:
            if cid in rec:
                gb, gr = base[cid]["gen_s"], rec[cid]["gen_s"]
                d = (gr - gb) / gb * 100 if gb else None
                if d is not None and d > 25 and gr - gb > 2:
                    degraded.append((s, cid, gb, gr))
                L.append(f"| {s} | {cid} | {f(gb)} | {f(gr)} | {f(d, 0)}% | {f(base[cid]['e2e_s'])} | {f(rec[cid]['e2e_s'])} |")
    L.append("")

    # --------------------------------------------------------------- by case
    L.append("## Per case (all phases)")
    L.append("")
    L.append("| Server | Case | expect | n | pass | fail | gen p50 | gen max | e2e p95 | mp4 dur (median) | size MB (median) | note |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    summary["cases"] = {}
    by_case = defaultdict(list)
    for r in reqs:
        by_case[(r["server"], r["case_id"])].append(r)
    for (s, cid), rs in sorted(by_case.items()):
        rate, ok, judged = pass_rate(rs)
        g = stats([r for r in rs if r.get("status_final") == "completed"], "gen_s")
        e = stats([r for r in rs if r.get("status_final") == "completed"], "e2e_s")
        durs = [r["mp4_duration_s"] for r in rs if r.get("mp4_duration_s")]
        sizes = [r["bytes"] / 1e6 for r in rs if r.get("bytes")]
        note = rs[0].get("note") or ""
        fails = judged - ok
        L.append(f"| {s} | {cid} | {rs[0]['expect']} | {len(rs)} | {ok} | {fails} | {f(g['p50'])} | {f(g['max'])} | {f(e['p95'])} | "
                 f"{f(statistics.median(durs), 2) if durs else '–'} | {f(statistics.median(sizes)) if sizes else '–'} | {note[:70]} |")
        summary["cases"][f"{s}/{cid}"] = {"n": len(rs), "pass": ok, "fail": fails, "gen_p50": g["p50"], "gen_max": g["max"], "e2e_p95": e["p95"]}
    L.append("")

    # ---------------------------------------------------------- validation
    L.append("## Admission control: negative and probe cases")
    L.append("")
    L.append("reject = a 4xx was expected; probe = behaviour unknown beforehand, recorded as seen. Outcome is the first occurrence; n counts every send (sustain re-injects them under load).")
    L.append("")
    L.append("| Server | Case | expect | n | pass | HTTP | outcome | detail |")
    L.append("|---|---|---|---|---|---|---|---|")
    val = defaultdict(list)
    for r in reqs:
        if r["expect"] in ("reject", "probe"):
            val[(r["server"], r["case_id"])].append(r)
    unexpected = []
    for (s, cid), rs in sorted(val.items()):
        r0 = rs[0]
        rate, ok, judged = pass_rate(rs)
        if r0["expect"] == "reject":
            outcome = "rejected" if r0["pass"] else "ACCEPTED"
            if not r0["pass"]:
                unexpected.append((s, cid, r0["http_code"]))
        else:
            outcome = (f"job {r0.get('status_final')}" if r0.get("job_id") else f"rejected HTTP {r0['http_code']}")
            if r0.get("mp4_duration_s"):
                outcome += f", mp4 {r0['mp4_duration_s']:.2f}s"
        codes = Counter(r["http_code"] for r in rs)
        detail = (r0.get("detail") or r0.get("error") or "")
        detail = detail.replace("|", "/").replace("\n", " ")[:120]
        L.append(f"| {s} | {cid} | {r0['expect']} | {len(rs)} | {ok}/{judged} | {dict(codes)} | {outcome} | {detail} |")
    L.append("")

    # ------------------------------------------------------------- lifecycle
    lcs = [r for r in rows if r.get("kind") == "lifecycle"]
    if lcs:
        L.append("## Lifecycle checks (cancel, concurrent download, delete, unknown id)")
        L.append("")
        L.append("| Server | check | result | detail |")
        L.append("|---|---|---|---|")
        for lc in lcs:
            for k, v in lc["checks"].items():
                d = {kk: vv for kk, vv in v.items() if kk != "pass"}
                L.append(f"| {lc['server']} | {k} | {'ok' if v.get('pass') else 'FAIL'} | {json.dumps(d)[:140].replace('|', '/')} |")
        L.append("")

    # ---------------------------------------------------- api under load
    ham = [r for r in rows if r.get("kind") == "hammer"]
    lst = [r for r in rows if r.get("kind") == "jobs-list"]
    if ham or lst:
        L.append("## Control-plane under load")
        L.append("")
        if ham:
            L.append("Status-poll hammer: 3 threads × 30 GETs at 10 Hz on a live job, repeated during sustain. "
                     "A 404 on a job the harness had already deleted (it deletes each job after downloading its MP4) is the correct answer and is counted separately.")
            L.append("")
            L.append("| Server | rounds | polls | 200 | 404 after our delete | other non-200 | p50 | p95 | max |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for s in servers:
                hs = [h for h in ham if h["server"] == s]
                if hs:
                    n = sum(h["n"] for h in hs)
                    ok, gone, other = hammer_split(hs, rows)
                    L.append(f"| {s} | {len(hs)} | {n} | {ok} | {gone} | {other} | {f(statistics.median([h['p50_s'] for h in hs]), 3)} | {f(max(h['p95_s'] for h in hs), 3)} | {f(max(h['max_s'] for h in hs), 3)} |")
            L.append("")
        if lst:
            L.append("GET /v1/videos/jobs (the full job list, base64 payloads included) during sustain:")
            L.append("")
            L.append("| Server | probes | non-200 | jobs (first → last) | MB (first → last) | latency p50 / max |")
            L.append("|---|---|---|---|---|---|")
            for s in servers:
                ls = [l for l in lst if l["server"] == s]
                if ls:
                    bad = sum(1 for l in ls if l["http_code"] != 200)
                    L.append(f"| {s} | {len(ls)} | {bad} | {ls[0].get('n_jobs')} → {ls[-1].get('n_jobs')} | {f((ls[0].get('bytes') or 0)/1e6)} → {f((ls[-1].get('bytes') or 0)/1e6)} | {f(statistics.median([l['latency_s'] for l in ls]))} / {f(max(l['latency_s'] for l in ls))} |")
            L.append("")

    # ------------------------------------------------------------- monitor
    L.append("## Server health during the run (/tt-liveness every 10 s)")
    L.append("")
    L.append("| Server | samples | non-200 | latency p95 | queue_size max | model_ready false | canary not healthy | error_count start → end | restart_count start → end |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    summary["monitor"] = {}
    for s in servers:
        ms = [m for m in mon if m["server"] == s]
        if not ms:
            continue
        okm = [m for m in ms if m["http_code"] == 200]
        qmax = max((m.get("queue_size") or 0) for m in okm) if okm else None
        L.append(f"| {s} | {len(ms)} | {len(ms) - len(okm)} | {f(pct([m['latency_s'] for m in ms], 0.95), 2)} | {qmax} | "
                 f"{sum(1 for m in okm if m.get('model_ready') is False)} | {sum(1 for m in okm if m.get('canary_state') not in (None, 'healthy'))} | "
                 f"{okm[0].get('error_count') if okm else '–'} → {okm[-1].get('error_count') if okm else '–'} | "
                 f"{okm[0].get('restart_count') if okm else '–'} → {okm[-1].get('restart_count') if okm else '–'} |")
        summary["monitor"][s] = {"samples": len(ms), "non200": len(ms) - len(okm), "queue_max": qmax,
                                 "error_count": [okm[0].get("error_count") if okm else None, okm[-1].get("error_count") if okm else None],
                                 "restart_count": [okm[0].get("restart_count") if okm else None, okm[-1].get("restart_count") if okm else None]}
    L.append("")

    # ----------------------------------------------------------- throughput
    from datetime import datetime
    L.append("## Throughput per minute (expected-ok requests, by submit time)")
    L.append("")
    if reqs:
        t0 = min(datetime.fromisoformat(r["submitted_at"]) for r in reqs)
        per_min = defaultdict(lambda: Counter())
        phase_of_min = {}
        for r in reqs:
            if r["expect"] != "ok":
                continue
            m = int((datetime.fromisoformat(r["submitted_at"]) - t0).total_seconds() // 60)
            per_min[m][r["server"]] += 1
            per_min[m][r["server"] + "_fail"] += 0 if r.get("pass") else 1
            phase_of_min.setdefault(m, r["phase"])
        L.append("| minute | phase | " + " | ".join(f"{s} sent / failed" for s in servers) + " |")
        L.append("|---|---|" + "---|" * len(servers))
        for m in range(0, max(per_min) + 1 if per_min else 0):
            c = per_min.get(m, Counter())
            L.append(f"| {m} | {phase_of_min.get(m, '')} | " + " | ".join(f"{c[s]} / {c[s + '_fail']}" for s in servers) + " |")
        L.append("")

    # ---------------------------------------------------------------- errors
    errs = Counter()
    for r in reqs:
        if r.get("error") and r["expect"] != "reject":
            errs[(r["server"], r["case_id"], r["error"][:110])] += 1
    L.append("## Errors on expected-ok and probe requests")
    L.append("")
    if errs:
        L.append("| count | server | case | error |")
        L.append("|---|---|---|---|")
        for (s, cid, e), n in errs.most_common(40):
            L.append(f"| {n} | {s} | {cid} | {e.replace('|', '/')} |")
    else:
        L.append("None.")
    L.append("")

    # --------------------------------------------------------- determinism
    det = defaultdict(list)
    for r in reqs:
        if r["case_id"].endswith("seed-det") and r.get("sha256"):
            det[(r["server"], r["case_id"])].append((r["sha256"], r.get("bytes")))
    if det:
        L.append("## Seed determinism (same request twice, back to back)")
        L.append("")
        for (s, cid), hs in det.items():
            same = len({h for h, _ in hs}) == 1
            L.append(f"- {s} {cid}: {len(hs)} runs, {'identical bytes' if same else 'DIFFERENT outputs'} ({', '.join(f'{h} {b}B' for h, b in hs)})")
        L.append("")

    # --------------------------------------------------------------- findings
    L.append("## Findings")
    L.append("")
    findings = []
    for s in servers:
        rs = [r for r in reqs if r["server"] == s and r["expect"] == "ok"]
        rate, ok, judged = pass_rate(rs)
        if judged:
            findings.append(f"**{s}**: {ok}/{judged} expected-ok requests passed ({rate*100:.1f}%).")
        bad = [r for r in rs if r.get("status_final") in ("timeout", "lost")]
        if bad:
            findings.append(f"**{s}**: {len(bad)} request(s) timed out or lost their job: " + ", ".join(sorted({r['case_id'] for r in bad})))
        fj = [r for r in rs if r.get("status_final") == "failed"]
        if fj:
            def _why(r):
                e = r.get("error") or ""
                return e.split("Runner error for task")[-1].split(":", 1)[-1].strip().rstrip('"}')[:70] if "Runner error" in e else e[:70]
            findings.append(f"**{s}**: {len(fj)} job(s) ended `failed`: " + "; ".join(sorted({f"{r['case_id']} ({_why(r)})" for r in fj})[:6]))
        fivexx = [r for r in reqs if r["server"] == s and (r.get("http_code") or 0) >= 500]
        if fivexx:
            findings.append(f"**{s}**: {len(fivexx)} HTTP 5xx responses ({dict(Counter(r['http_code'] for r in fivexx))}).")
        dm = [r for r in rs if r.get("error") and "duration" in r["error"] and r.get("status_final") == "completed"]
        if dm:
            findings.append(f"**{s}**: {len(dm)} completed clip(s) came back with the wrong length: " + "; ".join(sorted({f"{r['case_id']}: {r['error'][:60]}" for r in dm})[:4]))
        na = [r for r in rs if r.get("has_audio") is False]
        if na:
            findings.append(f"**{s}**: {len(na)} clip(s) had no audio track: " + ", ".join(sorted({r['case_id'] for r in na})))
        m = summary["monitor"].get(s) or {}
        if m.get("restart_count") and m["restart_count"][0] is not None and m["restart_count"][1] != m["restart_count"][0]:
            findings.append(f"**{s}**: worker restart_count changed {m['restart_count'][0]} → {m['restart_count'][1]} during the run.")
        if m.get("error_count") and m["error_count"][0] is not None and m["error_count"][1] != m["error_count"][0]:
            findings.append(f"**{s}**: worker error_count changed {m['error_count'][0]} → {m['error_count'][1]} during the run.")
        if m.get("non200"):
            findings.append(f"**{s}**: /tt-liveness failed {m['non200']} of {m['samples']} samples.")
    for s, cid, code in unexpected:
        findings.append(f"**{s}**: `{cid}` should have been rejected but got HTTP {code}.")
    for s, cid, gb, gr in degraded:
        findings.append(f"**{s}**: `{cid}` generation went from {gb:.1f}s (baseline) to {gr:.1f}s (recovery).")
    for s in servers:
        sp = summary["phases"].get(f"{s}/sustain") or {}
        bp = summary["phases"].get(f"{s}/baseline") or {}
        if sp.get("gen") and bp.get("gen") and sp["gen"]["p50"] and bp["gen"]["p50"]:
            findings.append(f"**{s}**: gen p50 {bp['gen']['p50']:.1f}s unloaded vs {sp['gen']['p50']:.1f}s under {sp.get('max_inflight')} in flight; "
                            f"queue p95 under load {sp['queue']['p95']:.1f}s, e2e p95 {sp['e2e']['p95']:.1f}s.")
        bu = summary["phases"].get(f"{s}/burst") or {}
        if bu.get("e2e") and bu["e2e"]["max"]:
            findings.append(f"**{s}**: burst of {bu['n']} drained with e2e p50 {bu['e2e']['p50']:.1f}s, max {bu['e2e']['max']:.1f}s, queue p95 {bu['queue']['p95']:.1f}s.")
    if not findings:
        findings.append("Nothing recorded.")
    L += [f"- {x}" for x in findings]
    L.append("")
    L.append(f"Artifacts: `{out_dir}/results.jsonl`, `results.csv`, `monitor.jsonl`, `stress.log`, sampled MP4s in `out/`.")
    summary["findings"] = findings
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    text = render(sys.argv[1])
    with open(os.path.join(sys.argv[1], "report.md"), "w") as fh:
        fh.write(text)
    print(text)
