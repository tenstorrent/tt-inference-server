#!/usr/bin/env python3
"""Static checks for the Grafana dashboards shipped in the chart.

Catches what a render cannot: a panel querying a template variable nobody
declares (the failure mode is a silent "No data"), a duplicate panel id, and —
with the rule file this writes, fed to `promtool check rules` — a PromQL typo.
Metric *existence* is out of reach here; that needs a running server.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Grafana's built-ins are substituted by Grafana, not declared as variables.
BUILTINS = {"__rate_interval", "__interval", "__range", "__from", "__to",
            "__interval_ms", "__timeFilter", "__name__"}
# Durations promtool can parse in place of Grafana's built-ins. Order matters:
# the _ms name has to be replaced before the prefix it shares with $__interval.
SUBSTITUTIONS = {"$__rate_interval": "5m", "$__interval_ms": "300000",
                 "$__interval": "5m", "$__range": "1h"}
# Grafana interpolates $var, ${var} and [[var]] alike, so all three have to be
# recognised here — an undeclared one in any spelling is the silent "No data"
# this script exists to catch.
VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)|\[\[(\w+)[^\]]*\]\]")


def panels(items):
    for p in items:
        yield p
        yield from panels(p.get("panels") or [])


def check(path: Path, rules: list[dict]) -> list[str]:
    problems: list[str] = []
    try:
        dash = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path.name}: not valid JSON ({e})"]

    declared = {v["name"] for v in dash.get("templating", {}).get("list", [])}
    seen_ids: dict[int, str] = {}
    for p in panels(dash.get("panels", [])):
        title = p.get("title", "<untitled>")
        pid = p.get("id")
        if pid in seen_ids:
            problems.append(f"{path.name}: panel id {pid} used twice "
                            f"({seen_ids[pid]!r} and {title!r})")
        seen_ids[pid] = title
        if p.get("type") == "row":
            continue
        targets = p.get("targets") or []
        if not targets:
            problems.append(f"{path.name}: panel {title!r} has no query")
        for t in targets:
            expr = t.get("expr", "")
            if not expr.strip():
                problems.append(f"{path.name}: panel {title!r} has an empty query")
                continue
            for groups in VAR_RE.findall(expr):
                var = next(g for g in groups if g)
                if var not in BUILTINS and var not in declared:
                    problems.append(f"{path.name}: panel {title!r} uses ${var}, "
                                    f"which is not declared in templating")
            for src, dst in SUBSTITUTIONS.items():
                expr = expr.replace(src, dst)
            # promtool names the rule in its error, so the name carries the
            # origin; a rule ordinal alone cannot be traced back to a panel.
            rules.append({"record": f"dashboard:{path.stem}:{pid}", "expr": expr,
                          "labels": {"dashboard": path.name, "title": title}})
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chart", type=Path)
    ap.add_argument("--rules", type=Path, required=True,
                    help="where to write the promtool rule file")
    args = ap.parse_args()

    files = sorted((args.chart / "dashboards").glob("*.json"))
    if not files:
        print(f"::error::no dashboards found under {args.chart}/dashboards")
        return 1

    problems: list[str] = []
    rules: list[dict] = []
    for f in files:
        problems += check(f, rules)
        print(f"checked {f.name}")

    args.rules.write_text(json.dumps(
        {"groups": [{"name": "dashboards", "rules": rules}]}, indent=2))
    print(f"wrote {len(rules)} queries to {args.rules} for promtool")

    for p in problems:
        print(f"::error::{p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
