#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gsm8k_rescore.py <lm_out dir | samples jsonl> [--json-out F]: rescore a logged GSM8K run (no generation) with the boxed-aware rule (GATE metric) and print it beside
lm-eval's flexible-extract and strict-match (as logged). Reads samples_*.jsonl of `lm_eval --log_samples`. Prints n, the three scores with stderr, and the
gate verdict against HY4_GATE_GSM8K (percent; default 0 = no gate)."""
import glob
import json
import math
import os
import sys
from pathlib import Path

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("hy4_gsm8k_utils", Path(__file__).resolve().parents[2] / "reference_config" / "evals" / "lm_eval_tasks" / "hy4_gsm8k" / "utils.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
extract_number, same = _m.extract_number, _m.same


def rescore(path):
    rows = {}
    for line in open(path):
        x = json.loads(line)
        rows.setdefault(x["doc_id"], {})[x["filter"]] = x
    n = len(rows)
    ok = {"boxed-aware": 0, "flexible-extract": 0, "strict-match": 0}
    for f in rows.values():
        fl = f["flexible-extract"]
        ok["boxed-aware"] += int(same(extract_number(fl["resps"][0][0]), fl["target"]))
        ok["flexible-extract"] += int(bool(fl["exact_match"]))
        if "strict-match" in f:
            ok["strict-match"] += int(bool(f["strict-match"]["exact_match"]))
    return n, ok


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    p = argv[0]
    if os.path.isdir(p):
        p = sorted(glob.glob(f"{p}/**/samples_*.jsonl", recursive=True))[-1]
    n, ok = rescore(p)
    out = {"samples": p, "n": n}
    for k, v in ok.items():
        s = v / n
        out[k] = {"correct": v, "acc": round(s, 4), "stderr": round(math.sqrt(s * (1 - s) / n), 4)}
        print(f"{k}: {v}/{n} = {s:.4f} +- {math.sqrt(s * (1 - s) / n):.4f}")
    gate = float(os.environ.get("HY4_GATE_GSM8K", "0"))
    out["gate_pct"] = gate
    out["gate_pass"] = ok["boxed-aware"] / n * 100 >= gate
    print(f"GATE (boxed-aware >= {gate}%): {'PASS' if out['gate_pass'] else 'FAIL'}")
    if "--json-out" in argv:
        json.dump(out, open(argv[argv.index("--json-out") + 1], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
