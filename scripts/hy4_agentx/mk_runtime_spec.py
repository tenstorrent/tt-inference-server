#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""mk_runtime_spec.py <B> <workflow benchmarks|agentic_traces|evals|agentic> <outdir> [catalog json] [--entry-only] [--model-env FILE]

CPU only. Takes the server's own catalog entry of the model (model.env: key model_specs[<HF id>][<DEVICE>][vLLM][<impl_id><B>]) and writes the
runtime-model-spec JSON that ``run.py --runtime-model-spec-json`` takes (the client side of the grid / AgentX / eval runs uses exactly the
server's entry). Generalisation of the DSV4 mk_runtime_spec.py (image5_release): the model key, the model name and the impl id come from model.env.

--entry-only : only extract the entry (stdlib only; this is what run_agentx.sh does); writes <outdir>/entry_<impl_id>.json and prints one summary line.
otherwise    : also builds the RuntimeConfig for ``workflow`` (needs the tt-inference-server tree on sys.path: env MKSPEC_WT, default = this checkout)
               and, for ``benchmarks``, prints the expanded sweep (cells, num_prompts) so the cell count can be checked before any boot.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness_model import HarnessModel  # noqa: E402


def extract_entry(m: HarnessModel, b: int, catalog: str) -> dict:
    d = json.load(open(catalog))
    try:
        v = d["model_specs"][m.model_id][m.device_key]["vLLM"][m.impl_id(b)]
    except KeyError as ex:
        raise SystemExit(f"catalog {catalog}: no entry model_specs[{m.model_id!r}][{m.device_key!r}]['vLLM'][{m.impl_id(b)!r}] (missing key {ex})")
    if v.get("model_id") != m.catalog_model_id(b):
        raise SystemExit(f"catalog entry model_id {v.get('model_id')!r} != expected {m.catalog_model_id(b)!r} (the AgentX config in reference_config/agentic_traces is keyed by it)")
    return v


def summary_line(m: HarnessModel, b: int, v: dict) -> str:
    dm = v["device_model_spec"]
    return "catalog entry: %s %s max_concurrency %s max_context %s" % (m.impl_id(b), v["model_id"], dm["max_concurrency"], dm["max_context"])


def main(argv) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = [a for a in argv if a.startswith("--")]
    me = None
    if "--model-env" in argv:
        i = argv.index("--model-env")
        me = argv[i + 1]
        args = [a for a in args if a != me]
    entry_only = "--entry-only" in flags
    if len(args) < 3:
        print(__doc__)
        return 2
    m = HarnessModel.load(me)
    b, workflow, out = int(args[0]), args[1], args[2]
    cat = args[3] if len(args) > 3 else m.catalog
    v = extract_entry(m, b, cat)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "entry_%s.json" % m.impl_id(b))
    json.dump(v, open(path, "w"))
    print(summary_line(m, b, v))
    if entry_only:
        print(path)
        return 0
    wt = os.environ.get("MKSPEC_WT", str(HERE.parents[1]))
    sys.path.insert(0, wt)
    os.environ["MODEL_SPECS_ENV"] = "dev"
    from workflows.model_spec import ModelSpec  # noqa: E402
    from workflows.runtime_config import RuntimeConfig  # noqa: E402

    spec = ModelSpec.from_json(path)
    rc = RuntimeConfig(model=m.model_name, workflow=workflow, device=m.device, impl=None, dev_mode=True, no_auth=True, service_port="8000",
                       server_url="http://127.0.0.1:8000", runtime_model_spec_json=None, **({"agentic_traces_mode": "ci"} if workflow == "agentic_traces" else {}))
    print(rc.to_json(spec, "final_%s" % m.impl_id(b), spec.model_id, out))
    print("max_concurrency", spec.device_model_spec.max_concurrency, "max_context", spec.device_model_spec.max_context, "tok_all", spec.device_model_spec.max_tokens_all_users)
    if workflow == "benchmarks":
        from reference_config.benchmarking import benchmark_config as bc  # noqa: E402

        cfg = bc.build_benchmark_config(spec)
        pts = []
        for t in cfg.tasks:
            for _dev, params in t.param_map.items():
                for p in params:
                    pts.append((t.__class__.__name__, p.isl, p.osl, p.max_concurrency, p.num_prompts))
        for x in pts[:80]:
            print(x)
        print("N", len(pts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
