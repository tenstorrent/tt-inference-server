#!/usr/bin/env python3
"""Per-task accuracy and the BBEH headline (harmonic mean over tasks) from lm-eval samples.

Usage: bbeh_summary.py <samples_gemma4_bbeh*.jsonl> [...]
The harmonic mean follows the BBEH paper: per-task accuracy in percent, +1 smoothing
so a zero task does not collapse the mean."""
import json
import sys
from collections import defaultdict

hits, totals = defaultdict(int), defaultdict(int)
for path in sys.argv[1:]:
    for line in open(path):
        if not line.strip():
            continue
        s = json.loads(line)
        task = s["doc"]["task"]
        totals[task] += 1
        hits[task] += int(s.get("exact_match", 0))
accs = {t: 100.0 * hits[t] / totals[t] for t in sorted(totals)}
for t, a in accs.items():
    print(f"{t:40s} {hits[t]:4d}/{totals[t]:<4d} {a:6.1f}")
n = sum(totals.values())
micro = 100.0 * sum(hits.values()) / n if n else 0.0
hm = len(accs) / sum(1.0 / (a + 1.0) for a in accs.values()) - 1.0 if accs else 0.0
print(f"\nquestions={n} tasks={len(accs)} micro_acc={micro:.1f} harmonic_mean={hm:.1f}")
