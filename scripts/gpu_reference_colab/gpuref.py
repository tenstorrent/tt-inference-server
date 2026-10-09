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
from types import SimpleNamespace
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
# Prefill chunk for GPU serving. The TT spec default is max_context, but a
# chunk that big makes vLLM's activation profiling OOM (cogito-qwen-14B on an
# A100: 6.75 GiB for 131072 tokens). It only schedules work; outputs match.
GPU_MAX_BATCHED_TOKENS = 16384
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
    vllm_args = dict(dms.vllm_args)
    batched = int(vllm_args.get("max_num_batched_tokens") or dms.max_context)
    vllm_args["max_num_batched_tokens"] = str(min(batched, GPU_MAX_BATCHED_TOKENS))
    return {
        "model": repo,
        "model_id": spec.model_id,
        "max_context": dms.max_context,
        "max_concurrency": dms.max_concurrency,
        "revision": dms.vllm_args.get("revision"),
        "vllm_serve_args": argv + vllm_flags(vllm_args),
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
    # Multimodal and hybrid checkpoints nest the decoder under text_config.
    config = (
        config
        if (config or {}).get("num_attention_heads")
        else (config or {}).get("text_config")
    )
    known = bool(config) and all(
        k in config for k in ("num_attention_heads", "num_hidden_layers")
    )
    per_token = kv_bytes_per_token(config) if known else None
    one_seq = per_token * max_context / GIB if per_token else 0.0
    # Peak MLP activation (gate+up, bf16) for one prefill chunk.
    chunk = min(max_context, GPU_MAX_BATCHED_TOKENS)
    act = chunk * int((config or {}).get("intermediate_size") or 0) * 4 / GIB
    return {
        "weights_gib": round(weights, 2),
        "activation_gib": round(act, 2),
        "kv_gib_one_seq": round(one_seq, 2) if per_token else None,
        "kv_gib_all_seqs": round(one_seq * max_concurrency, 2) if per_token else None,
        "need_gib": round(weights + act + one_seq + OVERHEAD_GIB, 2),
        "kv_bytes_per_token": per_token,
        "kv_unknown": per_token is None,
    }


def gpu_verdicts(estimate: Dict[str, Any], max_context: int) -> Dict[str, Any]:
    """Per GPU: "fits"; {"cap": tokens} when the weights fit but max_context
    does not (the runner then serves at vLLM's own estimate); or "no"."""
    verdicts: Dict[str, Any] = {}
    for gpu, mem in GPU_MEMORY_GIB.items():
        usable = mem * GPU_MEMORY_UTILIZATION
        fixed = estimate["weights_gib"] + estimate.get("activation_gib", 0.0)
        spare = usable - fixed - OVERHEAD_GIB
        per_token = estimate.get("kv_bytes_per_token")
        tokens = int(spare * GIB // per_token) if per_token and spare > 0 else 0
        if estimate["need_gib"] <= usable:
            verdicts[gpu] = "fits"
        else:
            verdicts[gpu] = (
                {"cap": min(tokens, max_context)} if tokens >= 2048 else "no"
            )
    return verdicts


def params_from_config(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """Dense-decoder parameter count from config.json, for checkpoints with no
    safetensors metadata on the Hub (e.g. .bin-only TinyLlama_v1.1)."""
    try:
        h, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
        heads = int(config["num_attention_heads"])
        kv = int(config.get("num_key_value_heads") or heads)
        head_dim = int(config.get("head_dim") or h // heads)
        embed = (
            int(config["vocab_size"])
            * h
            * (1 if config.get("tie_word_embeddings") else 2)
        )
        attn = 2 * h * heads * head_dim + 2 * h * kv * head_dim
        return embed + layers * (attn + 3 * h * int(config["intermediate_size"]))
    except (KeyError, TypeError, ValueError):
        return None


def model_facts(
    model: str, plan: Dict[str, Any], token: Optional[str]
) -> Dict[str, Any]:
    """HF gating/access, config.json and the memory-fit verdicts for a plan."""
    rev = plan.get("revision") or "main"
    info = hf_get(f"{HF}/api/models/{model}")
    gated = None if info is None else bool(info.get("gated"))
    use_token = token if gated is not False else None
    meta = hf_get(f"{HF}/api/models/{model}/revision/{rev}", use_token)
    config = hf_get(f"{HF}/{model}/resolve/{rev}/config.json", use_token)
    params = ((meta or {}).get("safetensors") or {}).get("total") or params_from_config(
        config
    )
    memory = None
    if params:
        ctx, conc = plan["max_context"], plan["max_concurrency"]
        memory = memory_estimate(params, config, ctx, conc)
        memory["verdicts"] = gpu_verdicts(memory, ctx)
    row = dict(plan, model=model, ok=True, gated=gated, memory=memory, config=config)
    row.update(needs_token=gated is not False, have_token=token is not None)
    if token and gated is not False:
        row["token_can_read"] = meta is not None
    return row


def check_models(models: List[str]) -> List[Dict[str, Any]]:
    """Spec, HF access and memory facts per model (preflight input)."""
    token, rows = hf_token(), []
    for model in models:
        try:
            plan = serve_plan(load_gpu_spec(model))
        except Exception as exc:  # the resolver raises ValueError
            rows.append({"model": model, "ok": False, "error": f"no GPU spec: {exc}"})
            continue
        rows.append(model_facts(model, plan, token))
    return rows


def preflight(rows: List[Dict[str, Any]], gpus: List[str]):
    """(usable_gpus, ok, messages). Each model must resolve, be readable and
    fit in bf16 on a listed GPU, fully or with a max_model_len cap (a capped
    run is checked by the runner's request evidence).
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
        allowed = full | capped | unknown
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
    """(state, display lines, models finished) from snippets/status.py output."""
    state, lines, done = None, [], 0
    for line in text.splitlines():
        if line.startswith("GPUREF_STATE="):
            state = line.split("=", 1)[1].strip() or None
        if line.startswith("GPUREF_DONE="):
            done = int(line.split("=", 1)[1] or 0)
        for prefix, label in STATUS_LABELS.items():
            if line.startswith(prefix):
                lines.append(label + line[len(prefix) :])
    return state, lines, done


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


def fmt(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, float) else "-" if v is None else str(v)


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
        "max_num_batched_tokens": next(
            (
                b
                for a, b in zip(
                    plan.get("vllm_serve_args", []), plan.get("vllm_serve_args", [])[1:]
                )
                if a == "--max-num-batched-tokens"
            ),
            None,
        ),  # fmt: skip
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


# --- sweep: pick targets from the catalog, place, run, report ----------------

SWEEP_VENVS = {"EVALS_COMMON"}  # tasks the lm-eval evals path can run on a GPU
SPEC_BEGIN = (
    "# BEGIN gpu-ref sweep: derived GPU specs. Generated by\n"
    "# scripts/gpu_reference_colab/gpuref.py sweep plan --write-specs from each\n"
    "# model's TT row (Quetzal row first): max_context = min(row, native\n"
    "# max_position_embeddings), same max_concurrency and HF revision. Do not hand-edit."
)
SPEC_END = "# END gpu-ref sweep: derived GPU specs"
ISSUE_URL = "https://github.com/tenstorrent/tt-inference-server/issues/5353"


def catalog_facts(ttis_dir: Path = REPO_ROOT):
    """(tasks, rows) per HF repo from TTIS's own eval-config map and dev catalog."""
    os.environ["MODEL_SPECS_ENV"] = "dev"
    sys.path.insert(0, str(ttis_dir))
    from reference_config.evals.eval_config import _eval_config_map
    from workflows.model_spec import MODEL_SPECS

    tasks = {
        repo: [
            {
                "task": t.task_name,
                "published": t.score.published_score,
                "gpu_ref": t.score.gpu_reference_score,
                "requested": getattr(t, "gpu_reference_requested", None),
                "venv": t.workflow_venv_type.name,
                "tolerance": t.score.tolerance,
            }
            for t in cfg.tasks
            if t.score
        ]
        for repo, cfg in _eval_config_map.items()
    }
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for spec in MODEL_SPECS.values():
        d = spec.device_model_spec
        rows.setdefault(spec.hf_model_repo, []).append({
            "device": spec.device_type.name, "impl": spec.impl.impl_id,
            "default": d.default_impl, "max_context": d.max_context,
            "max_concurrency": d.max_concurrency, "revision": d.vllm_args.get("revision"),
        })  # fmt: skip
    return tasks, rows


def select_targets(tasks, rows, refresh: bool = False, audit: frozenset = frozenset()):
    """(targets, skipped). A task is a target when gpu_reference_requested is
    set (a reviewed request to measure, even over an existing reference), when
    it has neither published_score nor gpu_reference_score, with refresh when
    it has a gpu_reference_score, and -- whatever its references -- when its
    model is in `audit` (every task of every Quetzal-row model). It needs an
    lm-eval (EVALS_COMMON) task and a GPU spec, or a TT row to derive one."""
    targets, skipped = [], []
    for repo in sorted(tasks):
        model_rows = rows.get(repo, [])
        has_gpu = any(r["device"] == "GPU" for r in model_rows)
        derivable = any(r["impl"] == "quetzal" or r["default"] for r in model_rows)
        for t in tasks[repo]:
            if t["requested"]:
                reason = f"requested: {t['requested']}"
            elif t["gpu_ref"] is None and t["published"] is None:
                reason = "ungated"
            elif refresh and t["gpu_ref"] is not None:
                reason = "refresh"
            elif repo in audit:
                reason = "audit: Quetzal P300X2 row"
            else:
                continue
            item = {
                "model": repo,
                "task": t["task"],
                "reason": reason,
                "old_gpu_ref": t["gpu_ref"],
                "published": t["published"],
                "tolerance": t.get("tolerance", 0.05),
            }
            if t["venv"] not in SWEEP_VENVS:
                skipped.append(
                    dict(item, skip=f"{t['venv']} task, not run by the lm-eval path")
                )
            elif not (has_gpu or derivable):
                skipped.append(
                    dict(item, skip="no GPU spec and no TT row to derive one")
                )
            else:
                targets.append(dict(item, spec="existing" if has_gpu else "derived"))
    return targets, skipped


def derive_gpu_spec(model_rows, native_ctx: Optional[int]) -> Optional[Dict[str, Any]]:
    """GPU spec from the model's Quetzal row, else its default TT rows (the
    smallest context): max_context capped at the native context."""
    tt = [r for r in model_rows if r["device"] != "GPU"]
    source = [r for r in tt if r["impl"] == "quetzal"] or [
        r for r in tt if r["default"]
    ]
    if not source:
        return None
    row = min(source, key=lambda r: r["max_context"])
    ctx = min(row["max_context"], native_ctx or row["max_context"])
    return {
        "impl": row["impl"], "max_context": ctx, "max_concurrency": row["max_concurrency"],
        "revision": row["revision"],
        "source": f"{row['impl']} {row['device']} row max_context {row['max_context']}, "
        f"native {native_ctx}",
    }  # fmt: skip


def gpu_group(verdicts: Dict[str, Any]) -> Optional[str]:
    """Run list for a model: "A100" if its bf16 weights fit an A100 (fully or
    with a cap) -- an A100 bills ~5.3 CU/hr against ~18 for an H100 and the
    scores are valid on either, so no H100 fallback -- "H100" if it needs
    80 GB, None if nothing fits."""

    def fits(gpu):
        return verdicts.get(gpu) == "fits" or isinstance(verdicts.get(gpu), dict)

    return "A100" if fits("A100") else "H100" if fits("H100") else None


def derive_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """vLLM hf_overrides that keep GPU serving like-for-like with TT. Quetzal
    lowers the HF transformers graph, whose Qwen2 attention ignores
    dual_chunk_attention_config (Qwen2.5-*-1M), so the TT side runs standard
    attention; vLLM 0.13 would switch to dual-chunk attention (and fails to
    start on FlashAttention). Identical for prompts within the chunk size."""
    return (
        {"dual_chunk_attention_config": None}
        if "dual_chunk_attention_config" in config
        else {}
    )


def derived_vllm_args(d: Dict[str, Any]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if d.get("revision"):
        args.update(revision=d["revision"], tokenizer_revision=d["revision"])
    if d.get("hf_overrides"):
        args["hf_overrides"] = d["hf_overrides"]
    return args


def derived_specs_yaml(derived: Dict[str, Dict[str, Any]]) -> str:
    out = []
    for model, d in sorted(derived.items()):
        out += [
            f"- weights:\n  - {model}\n  impl: {d['impl']}\n  inference_engine: VLLM",
            f"  device_model_specs:\n  - device: GPU\n    max_concurrency: {d['max_concurrency']}",
            f"    max_context: {d['max_context']}  # {d['source']}\n    default_impl: true",
        ]
        args = derived_vllm_args(d)
        if args:
            out.append("    vllm_args:")
        for key, value in args.items():
            if key == "hf_overrides":
                out.append(
                    "      # standard attention, as the TT (HF graph) side runs it"
                )
                value = json.dumps(value)
            out.append(f"      {key}: {value}")
        out.append("  status: EXPERIMENTAL")
    return "\n".join(out)


def block_models(text: str) -> List[str]:
    """Models whose GPU spec lives in the generated block of llm.yaml."""
    if SPEC_BEGIN not in text:
        return []
    block = text[text.index(SPEC_BEGIN) : text.index(SPEC_END)]
    return re.findall(r"^- weights:\n  - (\S+)", block, re.M)


def write_derived_specs(text: str, derived: Dict[str, Dict[str, Any]]) -> str:
    """llm.yaml text with the generated block replaced by `derived` (the block
    is regenerated in full, so a rule change reaches every derived spec)."""
    block = (
        f"{SPEC_BEGIN}\n{derived_specs_yaml(derived)}\n{SPEC_END}\n" if derived else ""
    )
    if SPEC_BEGIN in text:
        end = text.index(SPEC_END) + len(SPEC_END) + 1
        return text[: text.index(SPEC_BEGIN)] + block + text[end:]
    return text.rstrip("\n") + "\n\n" + block if block else text


def finished_models(results_root: Path) -> Dict[str, Any]:
    """model -> (finished_at, results dir, provenance) of its newest ok run."""
    done: Dict[str, Any] = {}
    for prov_path in results_root.glob("*/results/*/provenance.json"):
        prov = load_json(prov_path) or {}
        if prov.get("status") != "ok":
            continue
        model, stamp = prov.get("model"), prov.get("finished_at") or ""
        if model not in done or stamp > done[model][0]:
            done[model] = (stamp, prov_path.parent.parent, prov)
    return done


def run_covers(
    prov: Dict[str, Any], results_dir: Path, argv: List[str], tasks: List[str]
) -> bool:
    """A finished run counts for resume only if it served exactly this plan
    (same vllm serve args: revision, context, flags) and scored every task."""
    scored = {
        r["task"]
        for r in summary_rows(results_dir.parent)
        if r["model"] == prov.get("model")
    }
    return (prov.get("vllm_serve_command") or [None])[1:] == argv and set(
        tasks
    ) <= scored


# A100 wall minutes per task for a ~8B model, measured on the 2026-10-08/09 Colab
# runs (full samples, 32 concurrent); a model's estimate scales with its size
# (floor 0.5x) and adds ~10 min of serve/startup. Rough, for budgeting only.
A100_TASK_MINUTES = {
    "leaderboard_ifeval": 3, "ifeval": 3, "leaderboard_math_hard": 8, "mmlu_pro": 25,
    "gpqa_diamond_generative_n_shot": 6, "gpqa_diamond_cot_zeroshot": 6, "r1_aime24": 20,
    "r1_gpqa_diamond": 30, "mbpp_instruct": 5, "humaneval_instruct": 4,
}  # fmt: skip
CU_PER_HOUR = {"A100": 5.3, "H100": 18.05}


def estimate_hours(task_names: List[str], weights_gib: Optional[float]) -> float:
    scale = max(0.5, (weights_gib or 15.0) / 15.0)  # 15 GiB ~ an 8B model in bf16
    minutes = sum(A100_TASK_MINUTES.get(t, 10) for t in task_names) * scale + 10
    return round(minutes / 60, 2)


def quetzal_models(rows, refs: List[str]) -> Dict[str, str]:
    """Models with a Quetzal P300X2 row: here, and in each git ref's dev
    catalog (e.g. main and open row PRs). model -> where it was found."""
    found = {m: "this branch" for m, rs in rows.items()
             if any(r["impl"] == "quetzal" and r["device"] == "P300X2" for r in rs)}  # fmt: skip
    for ref in refs:
        text = run_out(
            ["git", "show", f"{ref}:workflows/model_specs/dev/llm.yaml"], cwd=REPO_ROOT
        )
        try:
            import yaml

            templates = (yaml.safe_load(text or "") or {}).get("templates") or []
        except yaml.YAMLError:
            templates = []
        for tpl in templates:
            devices = {d.get("device") for d in tpl.get("device_model_specs") or []}
            if tpl.get("impl") == "quetzal" and "P300X2" in devices:
                for model in tpl.get("weights") or []:
                    found.setdefault(model, ref)
    return found


def sweep_plan(sha: str, results_root: Path, refresh: bool = False, audit_refs=None):
    tasks, rows = catalog_facts()
    audit: Dict[str, str] = (
        quetzal_models(rows, audit_refs) if audit_refs is not None else {}
    )
    targets, skipped = select_targets(tasks, rows, refresh, frozenset(audit))
    for model, where in sorted(audit.items()):
        if model not in tasks:
            skipped.append({"model": model, "task": "*", "reason": "audit: Quetzal P300X2 row",
                            "skip": f"row on {where}; no EvalConfig on this branch"})  # fmt: skip
    from workflows.model_spec import DeviceModelSpec, DeviceTypes, get_model_id

    yaml_text = (
        REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml"
    ).read_text()
    generated = set(block_models(yaml_text))
    token, derived, groups, placed = hf_token(), {}, {}, {}
    finished = finished_models(results_root)
    for model in sorted({t["model"] for t in targets}):
        has_gpu = any(r["device"] == "GPU" for r in rows[model])
        if has_gpu and model not in generated:
            plan = serve_plan(load_gpu_spec(model))
        else:  # (re)derive: no GPU spec yet, or one this sweep generated
            rev = next(
                (r["revision"] for r in rows[model] if r["impl"] == "quetzal"), None
            )
            config = (
                hf_get(f"{HF}/{model}/resolve/{rev or 'main'}/config.json", token) or {}
            )
            d = derive_gpu_spec(rows[model], config.get("max_position_embeddings"))
            if not d["revision"]:  # pin the Hub's current main
                d["revision"] = (
                    hf_get(f"{HF}/api/models/{model}/revision/main", token) or {}
                ).get("sha")
            d["hf_overrides"] = derive_overrides(config)
            derived[model] = d
            dms = DeviceModelSpec(device=DeviceTypes.GPU, max_concurrency=d["max_concurrency"],
                                  max_context=d["max_context"], default_impl=True,
                                  vllm_args=derived_vllm_args(d))  # fmt: skip
            dms.vllm_args.setdefault("model", model)
            model_id = get_model_id(
                d["impl"].replace("_", "-"), model.split("/")[-1], "gpu"
            )
            plan = serve_plan(
                SimpleNamespace(
                    device_model_spec=dms, hf_model_repo=model, model_id=model_id
                )
            )
        facts = model_facts(model, plan, token)
        memory = facts["memory"] or {}
        model_tasks = [t["task"] for t in targets if t["model"] == model]
        # run.py evaluates every lm-eval task of the model, targeted or not.
        model_tasks_all = {
            model: [t["task"] for t in tasks[model] if t["venv"] in SWEEP_VENVS]
        }
        prior = finished.get(model)
        placed[model] = {
            "group": gpu_group(memory.get("verdicts") or {}),
            "max_context": plan["max_context"],
            "need_gib": memory.get("need_gib"),
            "est_hours": estimate_hours(model_tasks_all.get(model, []), memory.get("weights_gib")),
            "done": bool(prior) and run_covers(prior[2], prior[1], plan["vllm_serve_args"], model_tasks),
        }  # fmt: skip
    kept = []
    for t in targets:
        info = placed[t["model"]]
        if info["group"] is None:
            why = ("bf16 weights fit no Colab GPU (needs a multi-GPU machine)"
                   if info["need_gib"] else "model size unknown (no Hub parameter count or config)")  # fmt: skip
            skipped.append(dict(t, skip=why))
            continue
        kept.append(
            dict(t, **info, spec="derived" if t["model"] in derived else "existing")
        )
        if not info["done"]:
            group = groups.setdefault(info["group"], [])
            if t["model"] not in group:
                group.append(t["model"])
    placeable = {t["model"] for t in kept}
    derived = {m: d for m, d in derived.items() if m in placeable}
    stale = write_derived_specs(yaml_text, derived) != yaml_text
    estimate = {}
    for group, models in groups.items():
        hours = round(sum(placed[m]["est_hours"] for m in models), 1)
        gpu = group.split(",")[0]
        estimate[group] = {"models": len(models), "hours": hours,
                           "cu_at_first_choice": round(hours * CU_PER_HOUR[gpu]),
                           "cu_at_a100_rate": round(hours * CU_PER_HOUR["A100"])}  # fmt: skip
    return {"sha": sha, "targets": kept, "skipped": skipped, "groups": groups,
            "derived": derived, "specs_stale": stale, "estimate": estimate}  # fmt: skip


def patch_eval_config(text: str, results: List[Dict[str, Any]], ref_url: str) -> str:
    """eval_config.py text with each result's gpu_reference_score/_ref set and
    its gpu_reference_requested cleared. Proposed only; never applied here."""
    for r in results:
        start = text.index(f'        hf_model_repo="{r["model"]}",')
        end = text.find("\n    EvalConfig(", start)
        end = len(text) if end < 0 else end
        block = text[start:end]
        at = block.index(f'task_name="{r["task"]}"')
        nxt = block.find("EvalTask(", at)
        nxt = len(block) if nxt < 0 else nxt
        task = block[at:nxt]
        task = re.sub(r"\n\s*gpu_reference_requested=(\"[^\"]*\"|'[^']*'),", "", task)
        if r.get("verdict") == "keep":  # matches the old reference within noise
            text = text[:start] + block[:at] + task + block[nxt:] + text[end:]
            continue
        task = re.sub(
            r"gpu_reference_score=[^,\n]+,",
            f"gpu_reference_score={r['score']:.2f},",
            task,
            1,
        )
        task = re.sub(
            r"gpu_reference_score_ref=[^\n]+,",
            f'gpu_reference_score_ref="{ref_url}",',
            task,
            1,
        )
        text = text[:start] + block[:at] + task + block[nxt:] + text[end:]
    return text


def sweep_results(
    plan, results_root: Path, tt_root: Optional[Path]
) -> List[Dict[str, Any]]:
    """One row per target: GPU score from the newest ok run of its model, the
    evidence/cap/GPU from provenance, and the TT score from Shield reports."""
    done = finished_models(results_root)
    tt = {}
    if tt_root:
        for row in summary_rows(tt_root):
            if row["device"] != "GPU":
                tt[(row["model"], row["task"])] = row["score"]
    rows = []
    for t in plan["targets"]:
        row = dict(t, tt_score=tt.get((t["model"], t["task"])), score=None)
        if t["model"] in done:
            _, results_dir, prov = done[t["model"]]
            # Partial fetches keep the TTIS reports beside results/, so read the session dir.
            rows_ = summary_rows(results_dir.parent)
            scores = {(r["model"], r["task"]): r["score"] for r in rows_}
            evidence = (prov.get("request_evidence") or {}).get(
                "max_request_tokens"
            ) or {}
            math = t["task"] == "leaderboard_math_hard"
            prefix = "leaderboard_math_" if math else t["task"] + "_"
            sub = [
                v for k, v in evidence.items() if k == t["task"] or k.startswith(prefix)
            ]
            serve = prov.get("vllm_serve_command") or []
            chunk = (
                serve[serve.index("--max-num-batched-tokens") + 1]
                if "--max-num-batched-tokens" in serve
                else None
            )
            row.update(
                max_num_batched_tokens=chunk,
                score=scores.get((t["model"], t["task"])),
                samples=sum(v["samples"] for v in sub) or None,
                max_request_tokens=max((v["max_tokens"] for v in sub), default=None),
                cap=prov.get("max_model_len_cap"), gpu=(prov.get("gpu") or {}).get("name"),
                ttis_sha=prov.get("ttis_sha"), revision=prov.get("model_revision_sha"),
                vllm=(prov.get("packages") or {}).get("vllm"), results=str(results_dir),
            )  # fmt: skip
        row["verdict"] = reference_verdict(
            row.get("old_gpu_ref"), row["score"], row.get("samples")
        )
        current = row.get("old_gpu_ref") or row.get("published")
        proposed = current if row["verdict"] in (None, "keep") else row["score"]
        n, tol = row.get("samples"), row.get("tolerance", 0.05)
        row["tt_now"] = tt_grade(row["tt_score"], current, tol, n)
        row["tt_proposed"] = tt_grade(row["tt_score"], proposed, tol, n)
        rows.append(row)
    return rows


def tt_grade(score, ref, tolerance: float, n) -> str:
    """TTIS's rule for a TT score against a reference: PASS on the ratio rule
    (score/ref >= 1 - tolerance) or within noise (ref - score <= 1.96 binomial
    SE at the reference), else FAIL; NA without a score or reference."""
    if score is None or not ref:
        return "NA"
    if score / ref >= 1 - tolerance:
        return "PASS"
    se = 100 * (ref / 100 * (1 - ref / 100) / n) ** 0.5 if n and 0 < ref < 100 else None
    return "PASS" if se is not None and ref - score <= 1.96 * se else "FAIL"


def reference_verdict(
    old: Optional[float], new: Optional[float], n: Optional[int]
) -> Optional[str]:
    """ "new" (no old reference), "keep" (new within TTIS's noise rule of the old:
    |new-old| <= 1.96 binomial SE at the old score), or "replace"."""
    if new is None:
        return None
    if old is None:
        return "new"
    if n and 0 < old < 100:
        se = 100 * (old / 100 * (1 - old / 100) / n) ** 0.5
        if abs(new - old) <= 1.96 * se:
            return "keep"
    return "replace"


def sweep_markdown(rows: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> str:
    """Ready-to-paste #5353 block."""
    out = ["## GPU reference sweep", ""]
    out.append(
        "| model | task | GPU score | samples | TT score | old GPU ref | proposal | TT now -> proposed "
        "| why selected | GPU | chunk | cap (longest request) |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    proposal = {"new": "record", "keep": "keep old (within noise); clear flag",
                "replace": "replace old", None: "not measured"}  # fmt: skip
    for r in rows:
        cap = f"{r['cap']} ({r.get('max_request_tokens')})" if r.get("cap") else "none"
        out.append(
            f"| {r['model']} | {r['task']} | {fmt(r['score'])} | {fmt(r.get('samples'))} | "
            f"{fmt(r['tt_score'])} | {fmt(r.get('old_gpu_ref'))} | {proposal[r.get('verdict')]} | "
            f"{r.get('tt_now', 'NA')} -> {r.get('tt_proposed', 'NA')} | "
            f"{r['reason']} | {fmt(r.get('gpu'))} | {fmt(r.get('max_num_batched_tokens'))} | {cap} |"
        )
    measured = [r for r in rows if r.get("score") is not None]
    for title, now, new in (
        (
            "**TT passes that would FAIL against the proposed reference:**",
            "PASS",
            "FAIL",
        ),
        (
            "**TT failures that would PASS against the proposed reference:**",
            "FAIL",
            "PASS",
        ),
    ):
        hit = [
            r
            for r in measured
            if r.get("tt_now") == now and r.get("tt_proposed") == new
        ]
        lines = [
            f"- {r['model']} {r['task']}: TT {fmt(r['tt_score'])} vs GPU {fmt(r['score'])}"
            for r in hit
        ]
        out += ["", title] + (lines or ["- none"])
    shas = sorted({r["ttis_sha"] for r in rows if r.get("ttis_sha")})
    out += ["", f"TTIS sha(s): {', '.join(shas) or '-'}; vLLM "
            f"{', '.join(sorted({r['vllm'] for r in rows if r.get('vllm')})) or '-'} bf16; "
            "full sample counts; per-model provenance.json (HF revision, exact vllm serve "
            "args, request evidence) in each results dir. GPU prefill chunk "
            "(--max-num-batched-tokens) is capped at 16384: it changes scheduling only, "
            "not results."]  # fmt: skip
    if skipped:
        out += ["", "Skipped:", ""] + [
            f"- {s['model']} {s['task']}: {s['skip']}" for s in skipped
        ]
    return "\n".join(out) + "\n"


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
    p = sub.add_parser("sweep")
    p.add_argument("action", choices=["plan", "report"])
    p.add_argument("--sha", default=None)
    p.add_argument(
        "--plan", required=True, help="plan JSON (written by plan, read by report)"
    )
    p.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "workflow_logs" / "gpu_reference_colab"),
    )
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--refresh-quetzal", nargs="*", default=None, metavar="GIT_REF",
                   help="audit: every task of every model with a Quetzal P300X2 row here "
                   "and in these refs' catalogs (e.g. origin/main, open row PR heads)")  # fmt: skip
    p.add_argument(
        "--write-specs", action="store_true", help="add derived GPU specs to llm.yaml"
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--tt-reports", default=None, help="dir with Shield eval report JSONs"
    )
    p.add_argument("--emit-patch", action="store_true")
    p.add_argument("--ref-url", default=ISSUE_URL)
    add("provenance", out=req, model=req, ttis_dir=req, plan=req, vllm_python=req,
        vllm_bin=req, status=req, evals_rc={"type": int}, started_at=req,
        finished_at=req, note={}, max_model_len_cap={"type": int}, evidence={})  # fmt: skip
    args = parser.parse_args(argv)

    if args.cmd == "sweep" and args.action == "plan":
        plan = sweep_plan(
            args.sha, Path(args.results_root), args.refresh, args.refresh_quetzal
        )
        Path(args.plan).write_text(json.dumps(plan, indent=2) + "\n")
        for t in plan["targets"]:
            state = "done" if t["done"] else f"[{t['group']}]"
            print(f"[gpuref]   {state:12} {t['model']} {t['task']}: {t['reason']}"
                  f" (spec {t['spec']}, need ~{t['need_gib']} GiB)", file=sys.stderr)  # fmt: skip
        for t in plan["skipped"]:
            print(
                f"[gpuref]   skipped      {t['model']} {t['task']}: {t['skip']}",
                file=sys.stderr,
            )
        yaml_path = REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml"
        if plan["specs_stale"] and args.write_specs:
            yaml_path.write_text(
                write_derived_specs(yaml_path.read_text(), plan["derived"])
            )
            print(
                f"[gpuref]   derived GPU specs written to {yaml_path}", file=sys.stderr
            )
        for group, est in plan["estimate"].items():
            print(f"[gpuref]   group {group}: {est['models']} models, ~{est['hours']} h, "
                  f"~{est['cu_at_first_choice']} CU at {group.split(',')[0]} rates "
                  f"(~{est['cu_at_a100_rate']} at A100 rates)", file=sys.stderr)  # fmt: skip
        for group, models in plan["groups"].items():
            print(group + "\t" + " ".join(models))
        return 2 if plan["specs_stale"] else 0  # 2: commit specs first
    if args.cmd == "sweep":  # report
        plan = load_json(Path(args.plan))
        rows = sweep_results(
            plan,
            Path(args.results_root),
            Path(args.tt_reports) if args.tt_reports else None,
        )
        out = Path(args.out_dir or Path(args.plan).parent)
        summary = {"sha": plan["sha"], "results": rows, "skipped": plan["skipped"]}
        (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out / "sweep_5353.md").write_text(sweep_markdown(rows, plan["skipped"]))
        finished = [r for r in rows if r["score"] is not None]
        if args.emit_patch and finished:
            import difflib

            path = REPO_ROOT / "reference_config" / "evals" / "eval_config.py"
            old = path.read_text()
            new = patch_eval_config(old, finished, args.ref_url)
            name = "reference_config/evals/eval_config.py"
            diff = difflib.unified_diff(
                old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"
            )
            (out / "eval_config.patch").write_text("".join(diff))
        print(sweep_markdown(rows, plan["skipped"]))
        print(
            f"{len(finished)}/{len(rows)} targets measured; outputs in {out}",
            file=sys.stderr,
        )
        return 0
    if args.cmd == "preflight":
        usable, ok, msgs = preflight(check_models(args.models), args.gpus.split(","))
        print("\n".join(f"[gpuref]   {m}" for m in msgs), file=sys.stderr)
        print(" ".join(usable))
        return 0 if ok else 1
    if args.cmd == "status-view":
        state, lines, done = parse_status(sys.stdin.read())
        print("\n".join(lines), file=sys.stderr)
        print(f"{state} {done}" if state else "")
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
