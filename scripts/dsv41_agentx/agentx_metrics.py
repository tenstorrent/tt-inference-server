#!/usr/bin/env python3
"""Headline metrics of an AgentX (InferenceX / aiperf) run, computed with InferenceX's own aggregation code.

usage: agentx_metrics.py <dir containing (or above) profile_export.jsonl> [--repo <InferenceX checkout>] [--gpus N] [--json-out FILE]
Run it with the AGENTIC_TRACES venv python (the wrapper run_agentx.sh does). Prints a short summary
(TTFT p50/p90, ITL p50/p90, interactivity, aggregate tok/s, request counts) and writes the full numbers as JSON."""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("artifact_dir")
ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2] / ".workflow_venvs" / ".venv_agentic_traces" / "InferenceX"))
ap.add_argument("--gpus", type=int, default=32, help="chips, only for the per-chip lines")
ap.add_argument("--json-out", default=None)
a = ap.parse_args()
sys.path.insert(0, a.repo)
sys.path.insert(0, str(Path(a.repo) / "utils"))  # 'agentic' package (name 'utils' collides with tt-inference-server's)
try:  # InferenceX layout at the pinned ref (utils/agentic/aggregation); older refs had infx.results.agentic
    from agentic.aggregation.request_metrics import (  # noqa: E402
        compute_latency_stats, compute_throughput_stats, load_records_with_accounting)
except ImportError:
    from infx.results.agentic.artifacts import load_records_with_accounting  # noqa: E402
    from infx.results.agentic.request_metrics import compute_latency_stats, compute_throughput_stats  # noqa: E402

d = Path(a.artifact_dir)
if not (d / "profile_export.jsonl").is_file():
    kids = sorted(d.rglob("profile_export.jsonl"))
    if not kids:
        sys.exit(f"no profile_export.jsonl under {d}")
    d = kids[0].parent
records, acct = load_records_with_accounting(d / "profile_export.jsonl")
flat, _ = compute_latency_stats(records)
tflat, _ = compute_throughput_stats(records)
g = flat.get
out = {
    "artifact_dir": str(d),
    "requests_ok": len(records),
    "accounting": acct,
    "ttft_s": {k: g(f"{k}_ttft") for k in ("p50", "p90", "p95", "mean")},
    "tpot_itl_s": {k: g(f"{k}_itl") for k in ("p50", "p90", "mean")},
    "e2el_s": {k: g(f"{k}_e2el") for k in ("p50", "p90", "mean")},
    "e2e_norm_tok_s_user": {k: g(f"{k}_e2e_norm_intvty") for k in ("p50", "p90", "p95", "mean")},
    "decode_tok_s_user_1_over_itl": {k: g(f"{k}_intvty") for k in ("p50", "p90", "mean")},
    "throughput_tok_s": {k: tflat.get(k) for k in ("input_tput_tps", "output_tput_tps", "total_tput_tps", "duration_seconds")},
}
if tflat.get("total_tput_tps"):
    out["per_chip_total_tok_s"] = tflat["total_tput_tps"] / a.gpus
    out["per_chip_output_tok_s"] = tflat["output_tput_tps"] / a.gpus
js = Path(a.json_out) if a.json_out else d / "ix_metrics.json"
js.write_text(json.dumps(out, indent=2))


def f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


t, i, v, th = out["ttft_s"], out["tpot_itl_s"], out["e2e_norm_tok_s_user"], out["throughput_tok_s"]
print("=" * 64)
print("AgentX metrics (InferenceX aggregation)  " + str(d))
print(f"  requests ok            {out['requests_ok']}   accounting {acct}")
print(f"  TTFT  p50 / p90 / p95  {f(t['p50'])} / {f(t['p90'])} / {f(t['p95'])} s")
print(f"  ITL   p50 / p90        {f(i['p50'], 4)} / {f(i['p90'], 4)} s   (TPOT)")
print(f"  interactivity P50/P90  {f(v['p50'], 1)} / {f(v['p90'], 1)} tok/s/user  (E2E-normalised)")
print(f"  aggregate tok/s        output {f(th['output_tput_tps'], 1)}   total (in+out) {f(th['total_tput_tps'], 1)}   over {f(th['duration_seconds'], 0)} s")
print(f"  full numbers           {js}")
print("=" * 64)
