#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Non-`colab` logic of the Colab GPU-reference workflow (see README.md).

Driver: preflight, status-view, balance-below, summary.
Runner (on the VM): serve-plan, plan-argv, served, record-status,
request-evidence, cap-bound, provenance.

The HF token is read only from $HF_TOKEN or the token file; it is never an
argument and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
HF = "https://huggingface.co"
# Flags the runner sets itself (spec limits, bf16, the repo id as model and
# served name, which run.py's eval client requests).
EXPLICIT_SERVE_KEYS = {
    "model",
    "served_model_name",
    "max_model_len",
    "max_num_seqs",
    "dtype",
}
# DeviceModelSpec defaults for Tenstorrent serving that mean nothing on CUDA
# vLLM (TT config payload, TT paged-KV block size, log truncation).
TT_ONLY_SERVE_KEYS = {"additional_config", "block_size", "max-log-len"}
# Colab GPU memory (GiB); vLLM's default gpu_memory_utilization; margin for
# CUDA context, activations and graphs.
GPU_MEMORY_GIB = {"H100": 80.0, "A100": 40.0, "L4": 22.5, "T4": 15.0}
GPU_MEMORY_UTILIZATION = 0.9
OVERHEAD_GIB = 4.0
GIB = float(1 << 30)
REJECTION_PATTERN = re.compile(
    r"maximum context length|truncat|HTTP/1\.1\" 400|400 Bad Request", re.IGNORECASE
)


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# --- spec -> vllm serve -------------------------------------------------------


def load_gpu_spec(model: str, ttis_dir: Path = REPO_ROOT):
    """The model's GPU ModelSpec from TTIS's own loader, dev catalog, exactly as
    `run.py --dev-mode --tt-device gpu` resolves it. MODEL_SPECS is built at
    import time, so call this in a fresh interpreter."""
    os.environ["MODEL_SPECS_ENV"] = "dev"
    sys.path.insert(0, str(ttis_dir))
    from workflows.model_spec import get_runtime_model_spec

    return get_runtime_model_spec(model=model, device="gpu")[0]


def vllm_flags(vllm_args: Dict[str, Any]) -> List[str]:
    """Spec vllm_args as `vllm serve` flags (bools as --x / --no-x)."""
    flags: List[str] = []
    for key, value in vllm_args.items():
        if key in EXPLICIT_SERVE_KEYS | TT_ONLY_SERVE_KEYS or value is None:
            continue
        name = key.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{name}" if value else f"--no-{name}")
        else:
            text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            flags += [f"--{name}", text]
    return flags


def serve_plan(spec, port: int = 8000) -> Dict[str, Any]:
    dms, repo = spec.device_model_spec, spec.hf_model_repo
    argv = ["serve", repo, "--served-model-name", repo, "--port", str(port)]
    argv += ["--max-model-len", str(dms.max_context)]
    argv += ["--max-num-seqs", str(dms.max_concurrency), "--dtype", "bfloat16"]
    return {
        "model": repo,
        "model_id": spec.model_id,
        "max_context": dms.max_context,
        "max_concurrency": dms.max_concurrency,
        "revision": dms.vllm_args.get("revision"),
        "vllm_serve_args": argv + vllm_flags(dms.vllm_args),
        "dropped_tt_only_vllm_args": {
            k: v for k, v in dms.vllm_args.items() if k in TT_ONLY_SERVE_KEYS
        },
    }


# --- Hugging Face + memory fit ------------------------------------------------


def hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and path.exists():
        token = path.read_text().strip()
    return token or None


def hf_get(url: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """GET a Hub JSON document; None on any HTTP/network/parse error."""
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def kv_bytes_per_token(config: Dict[str, Any]) -> int:
    """bf16 K+V bytes per token for a standard decoder config.json."""
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    return 2 * int(config["num_hidden_layers"]) * kv_heads * head_dim * 2


def memory_estimate(
    num_params: int,
    config: Optional[Dict[str, Any]],
    max_context: int,
    max_concurrency: int,
) -> Dict[str, Any]:
    """bf16 weights, KV for one max_context sequence (what vLLM needs to start;
    more sequences only queue), and need = weights + one sequence + overhead.
    Without a readable config.json only the weights are sized."""
    weights = num_params * 2 / GIB
    per_token = kv_bytes_per_token(config) if config else None
    one_seq = per_token * max_context / GIB if per_token else 0.0
    return {
        "weights_gib": round(weights, 2),
        "kv_gib_one_seq": round(one_seq, 2) if per_token else None,
        "kv_gib_all_seqs": round(one_seq * max_concurrency, 2) if per_token else None,
        "need_gib": round(weights + one_seq + OVERHEAD_GIB, 2),
        "kv_bytes_per_token": per_token,
        "kv_unknown": per_token is None,
    }


def gpu_verdicts(estimate: Dict[str, Any], max_context: int) -> Dict[str, Any]:
    """Per GPU: "fits"; {"cap": tokens} when the weights fit but max_context
    does not (the runner then serves at vLLM's own estimate); or "no"."""
    verdicts: Dict[str, Any] = {}
    for gpu, mem in GPU_MEMORY_GIB.items():
        usable = mem * GPU_MEMORY_UTILIZATION
        spare = usable - estimate["weights_gib"] - OVERHEAD_GIB
        per_token = estimate.get("kv_bytes_per_token")
        tokens = int(spare * GIB // per_token) if per_token and spare > 0 else 0
        if estimate["need_gib"] <= usable:
            verdicts[gpu] = "fits"
        else:
            verdicts[gpu] = (
                {"cap": min(tokens, max_context)} if tokens >= 2048 else "no"
            )
    return verdicts


def check_models(models: List[str]) -> List[Dict[str, Any]]:
    """Spec, HF access and memory facts per model (preflight input)."""
    token, rows = hf_token(), []
    for model in models:
        try:
            plan = serve_plan(load_gpu_spec(model))
        except Exception as exc:  # the resolver raises ValueError
            rows.append({"model": model, "ok": False, "error": f"no GPU spec: {exc}"})
            continue
        rev = plan["revision"] or "main"
        info = hf_get(f"{HF}/api/models/{model}")
        gated = None if info is None else bool(info.get("gated"))
        use_token = token if gated is not False else None
        meta = hf_get(f"{HF}/api/models/{model}/revision/{rev}", use_token)
        config = hf_get(f"{HF}/{model}/resolve/{rev}/config.json", use_token)
        params = ((meta or {}).get("safetensors") or {}).get("total")
        memory = None
        if params:
            ctx, conc = plan["max_context"], plan["max_concurrency"]
            memory = memory_estimate(params, config, ctx, conc)
            memory["verdicts"] = gpu_verdicts(memory, ctx)
        row = dict(plan, model=model, ok=True, gated=gated, memory=memory)
        row.update(needs_token=gated is not False, have_token=token is not None)
        if token and gated is not False:
            row["token_can_read"] = meta is not None
        rows.append(row)
    return rows


def preflight(rows: List[Dict[str, Any]], gpus: List[str]):
    """(usable_gpus, ok, messages). Each model must resolve, be readable and
    fit in bf16 on a listed GPU: fully if any GPU allows it, else with a cap.
    A model fitting none is refused (one Colab GPU per runtime, so ~70B is out
    of scope). The run keeps the GPU types every model accepts."""
    usable, ok, msgs = list(gpus), True, []
    for r in rows:
        if not r.get("ok"):
            msgs += [
                f"{r['model']}: {r.get('error')}",
                "add a `- device: GPU` default entry",
            ]
            ok = False
            continue
        gated = {True: "gated", False: "public", None: "gating unknown"}[r["gated"]]
        msgs.append(
            f"{r['model']}: {r['model_id']} max_context={r['max_context']} "
            f"max_concurrency={r['max_concurrency']} revision={r['revision']} ({gated})"
        )
        if r["needs_token"] and not r["have_token"]:
            msgs.append("  needs an HF token: set HF_TOKEN or run `hf auth login`")
            ok = False
        if r.get("token_can_read") is False:
            msgs.append("  the HF token cannot read it: accept the license on the Hub")
            ok = False
        mem = r.get("memory")
        if mem is None:
            msgs.append(
                "  no parameter count on the Hub: cannot check GPU fit, refusing"
            )
            ok = False
            continue
        kv = (
            "KV not sized"
            if mem.get("kv_unknown")
            else f"KV {mem['kv_gib_one_seq']} GiB/seq"
        )
        msgs.append(
            f"  bf16 weights {mem['weights_gib']} GiB + {kv} -> needs ~{mem['need_gib']} GiB"
        )
        verdicts = {g: mem["verdicts"].get(g) for g in gpus}
        for g, v in verdicts.items():
            if isinstance(v, dict):
                msgs.append(
                    f"  {g}: max_context does not fit; ~{v['cap']} tokens would (cap)"
                )
            elif v == "no":
                msgs.append(f"  {g}: bf16 weights do not fit")
        full = {g for g, v in verdicts.items() if v == "fits"}
        capped = {g for g, v in verdicts.items() if isinstance(v, dict)}
        unknown = {g for g, v in verdicts.items() if v is None}  # e.g. G4
        if not (full or capped or unknown):
            hint = (
                "add H100 (80 GB) to --gpu"
                if "H100" not in gpus
                else "models above ~32B (e.g. 70B) need a multi-GPU machine, out of scope"
            )
            msgs.append(f"  refused: bf16 weights do not fit any GPU in --gpu; {hint}")
            ok = False
            continue
        allowed = (full or capped) | unknown
        if any(g not in allowed for g in usable):
            msgs.append(f"  dropping {', '.join(set(usable) - allowed)} from --gpu")
        usable = [g for g in usable if g in allowed]
    if ok and not usable:
        msgs.append("no GPU in --gpu fits every model; split them into separate runs")
        ok = False
    return usable, ok, msgs


# --- driver-side parsing ------------------------------------------------------

STATUS_LABELS = {
    "GPUREF_PHASE=": "  phase: ",
    "GPUREF_GPU=": "  gpu:   ",
    "GPUREF_EVAL=": "  evals: ",
    "GPUREF_LOG| ": "  | ",
}


def parse_status(text: str):
    """(state, display lines) from snippets/status.py output."""
    state, lines = None, []
    for line in text.splitlines():
        if line.startswith("GPUREF_STATE="):
            state = line.split("=", 1)[1].strip() or None
        for prefix, label in STATUS_LABELS.items():
            if line.startswith(prefix):
                lines.append(label + line[len(prefix) :])
    return state, lines


def parse_balance(text: str) -> Optional[float]:
    match = re.search(r"Current balance:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def summary_rows(root: Path) -> List[Dict[str, Any]]:
    """Per-task scores from the newest TTIS eval report per model under root
    (local reports_output/evals/data/report_data_*.json or Shield's
    report_*.json), with the GPU name from each provenance.json."""
    gpus, newest = {}, {}
    for path in sorted(root.rglob("provenance.json")):
        prov = load_json(path) or {}
        gpus[prov.get("model")] = (prov.get("gpu") or {}).get("name")
    for path in sorted(root.rglob("report*.json")):
        report = load_json(path)
        if not isinstance(report, dict) or not isinstance(report.get("sections"), list):
            continue
        meta = report.get("metadata") or {}
        model = meta.get("model_repo") or meta.get("model_name")
        stamp = meta.get("generated_at") or ""
        evals = [
            s
            for s in report["sections"]
            if isinstance(s, dict) and s.get("kind") == "evals"
        ]
        if evals and (model not in newest or stamp >= newest[model][0]):
            newest[model] = (stamp, meta, evals)
    return [
        {
            "model": model,
            "device": meta.get("device"),
            "gpu": gpus.get(model),
            "task": data.get("task_name"),
            "score": data.get("score"),
            "published_score": data.get("published_score"),
            "gpu_reference_score": data.get("gpu_reference_score"),
        }
        for model, (_, meta, evals) in sorted(newest.items())
        for data in (s.get("data") or {} for s in evals)
    ]


def format_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No TTIS eval reports found."
    keys = [
        "model",
        "device",
        "gpu",
        "task",
        "score",
        "published_score",
        "gpu_reference_score",
    ]

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else "-" if v is None else str(v)

    table = [["model", "device", "gpu", "task", "score", "published", "gpu_ref"]]
    table += [[fmt(r.get(k)) for k in keys] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(keys))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in table
    )


# --- runner-side helpers ------------------------------------------------------


def record_status(path: Path, model: str, status: str, note: Optional[str]) -> None:
    data = load_json(path) if path.exists() else None
    data = data if isinstance(data, dict) else {}
    data[model] = {"status": status, "note": note or None}
    path.write_text(json.dumps(data, indent=2) + "\n")


def request_token_stats(root: Path, count_tokens) -> Dict[str, Dict[str, int]]:
    """Per lm-eval task (samples_<task>_<timestamp>.jsonl): samples, longest
    prompt+generation in tokens, empty responses."""
    stats: Dict[str, Dict[str, int]] = {}
    for path in sorted(root.rglob("samples_*.jsonl")):
        task = re.sub(r"_\d{4}-\d{2}-\d{2}T[\d.-]+$", "", path.stem[len("samples_") :])
        entry = stats.setdefault(task, {"samples": 0, "max_tokens": 0, "empty": 0})
        for line in path.open():
            sample = json.loads(line)
            prompt = sample["arguments"]["gen_args_0"]["arg_0"]
            resp = sample["resps"][0][0]
            length = count_tokens(prompt) + count_tokens(resp)
            entry["samples"] += 1
            entry["max_tokens"] = max(entry["max_tokens"], length)
            entry["empty"] += 0 if resp.strip() else 1
    return stats


def rejection_lines(paths: List[Path]) -> List[str]:
    """Log lines that look like a rejected or truncated request."""
    return [
        f"{p.name}: {line.strip()[:200]}"
        for p in paths
        if p.exists()
        for line in p.read_text(errors="replace").splitlines()
        if REJECTION_PATTERN.search(line)
    ]


def cap_bound(evidence: Optional[Dict[str, Any]], cap: int) -> bool:
    """A capped run is unusable without evidence, with any rejection line, or
    with a request within 64 tokens (the harness's reserve) of the cap."""
    if not isinstance(evidence, dict) or evidence.get("rejection_log_lines"):
        return True
    lengths = [
        t["max_tokens"] for t in (evidence.get("max_request_tokens") or {}).values()
    ]
    return not lengths or max(lengths) >= cap - 64


def run_out(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:  # e.g. no nvidia-smi off the VM, no git in an exported tree
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def build_provenance(args: argparse.Namespace) -> Dict[str, Any]:
    ttis = Path(args.ttis_dir)
    plan = load_json(Path(args.plan)) or {}
    query = "--query-gpu=name,driver_version,memory.total"
    gpu = run_out(["nvidia-smi", query, "--format=csv,noheader"]) or ",,"
    gpu = [part.strip() for part in gpu.splitlines()[0].split(",")]
    versions = run_out(
        [
            args.vllm_python,
            "-c",
            (
                "import importlib.metadata as m, json; "
                "print(json.dumps({p: m.version(p) for p in ('vllm', 'torch', 'transformers')}))"
            ),
        ]
    )
    requirements = (ttis / "requirements" / "evals-common.txt").read_text()
    pinned = re.search(r"lm-evaluation-harness\.git@([0-9a-f]{7,40})", requirements)
    # pathlib's glob, unlike glob.glob, also matches the hidden .venv_* dirs.
    pattern = "*/lib/python*/site-packages/lm_eval-*.dist-info/direct_url.json"
    installed = [
        ((load_json(p) or {}).get("vcs_info") or {}).get("commit_id")
        for p in sorted((ttis / ".workflow_venvs").glob(pattern))
    ]
    rev = plan.get("revision") or "main"
    resolved = hf_get(f"{HF}/api/models/{args.model}/revision/{rev}", hf_token()) or {}
    return {
        "model": args.model,
        "model_id": plan.get("model_id"),
        "status": args.status,
        "evals_exit_code": args.evals_rc,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "gpu": dict(zip(["name", "driver", "memory_total"], gpu + [None] * 3)),
        "packages": json.loads(versions) if versions else {},
        "ttis_sha": run_out(["git", "rev-parse", "HEAD"], cwd=ttis),
        "model_revision_pinned": plan.get("revision"),
        "model_revision_sha": resolved.get("sha"),
        "max_context": plan.get("max_context"),
        "max_concurrency": plan.get("max_concurrency"),
        "vllm_serve_command": [args.vllm_bin] + plan.get("vllm_serve_args", []),
        # Set only when vLLM refused max_context on this GPU; lm-eval still
        # sizes prompts to max_context, hence request_evidence below.
        "max_model_len_cap": args.max_model_len_cap,
        "dropped_tt_only_vllm_args": plan.get("dropped_tt_only_vllm_args"),
        "run_py_command": [
            "python3",
            "run.py",
            "--workflow",
            "evals",
            "--tt-device",
            "gpu",
            "--model",
            args.model,
            "--dev-mode",
        ],  # fmt: skip
        "lm_eval_commit_pinned": pinned.group(1) if pinned else None,
        "lm_eval_commit_installed": next((c for c in installed if c), None),
        "note": args.note,
        "request_evidence": load_json(Path(args.evidence)) if args.evidence else None,
    }


# --- CLI ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    req = {"required": True}

    def add(name, *positional, **options):
        p = sub.add_parser(name)
        for arg in positional:
            p.add_argument(arg, nargs="+" if arg == "models" else None)
        for opt, kwargs in options.items():
            p.add_argument("--" + opt.replace("_", "-"), **kwargs)

    add("preflight", "models", gpus=req)
    add("status-view")
    add("balance-below", "floor")
    add("summary", "dir")
    add("serve-plan", model=req, ttis_dir={"default": str(REPO_ROOT)})
    add("plan-argv", "plan")
    add("served", "model")
    add("record-status", "file", "model", "status", "note")
    add("request-evidence", model=req, revision={}, samples_root=req, out=req,
        log={"action": "append", "default": []})  # fmt: skip
    add("cap-bound", "evidence", "cap")
    add("provenance", out=req, model=req, ttis_dir=req, plan=req, vllm_python=req,
        vllm_bin=req, status=req, evals_rc={"type": int}, started_at=req,
        finished_at=req, note={}, max_model_len_cap={"type": int}, evidence={})  # fmt: skip
    args = parser.parse_args(argv)

    if args.cmd == "preflight":
        usable, ok, msgs = preflight(check_models(args.models), args.gpus.split(","))
        print("\n".join(f"[gpuref]   {m}" for m in msgs), file=sys.stderr)
        print(" ".join(usable))
        return 0 if ok else 1
    if args.cmd == "status-view":
        state, lines = parse_status(sys.stdin.read())
        print("\n".join(lines), file=sys.stderr)
        print(state or "")
        return 0 if state else 1
    if args.cmd == "balance-below":
        balance = parse_balance(sys.stdin.read())
        print(balance)
        return 0 if balance is not None and balance < float(args.floor) else 1
    if args.cmd == "summary":
        status = load_json(Path(args.dir) / "status.json") or {}
        for model, info in status.items():
            print(
                f"{model}: {info['status']}"
                + (f" ({info['note']})" if info["note"] else "")
            )
        rows = summary_rows(Path(args.dir))
        print(format_summary(rows))
        return 0 if rows else 1
    if args.cmd == "served":
        try:
            ids = [m.get("id") for m in json.loads(sys.stdin.read()).get("data", [])]
        except (ValueError, AttributeError):
            return 1
        return 0 if args.model in ids else 1
    if args.cmd == "cap-bound":
        return 0 if cap_bound(load_json(Path(args.evidence)), int(args.cap)) else 1
    if args.cmd == "serve-plan":
        plan = serve_plan(load_gpu_spec(args.model, Path(args.ttis_dir)))
        print(json.dumps(plan, indent=2))
    elif args.cmd == "plan-argv":
        print("\n".join(load_json(Path(args.plan))["vllm_serve_args"]))
    elif args.cmd == "record-status":
        record_status(Path(args.file), args.model, args.status, args.note)
    elif args.cmd == "request-evidence":
        from transformers import AutoTokenizer  # the vLLM venv's python has it

        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        evidence = {
            "max_request_tokens": request_token_stats(
                Path(args.samples_root),
                lambda text: len(tok(text, add_special_tokens=False).input_ids),
            ),
            "rejection_log_lines": rejection_lines([Path(p) for p in args.log]),
        }
        Path(args.out).write_text(json.dumps(evidence, indent=2) + "\n")
    elif args.cmd == "provenance":
        Path(args.out).write_text(json.dumps(build_provenance(args), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
