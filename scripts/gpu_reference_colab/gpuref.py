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
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
HF = "https://huggingface.co"
REMOTE_DIR = "/content/gpuref"  # the VM work dir (snippets/*.py use the same)
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
# vLLM OpenAI-frontend flags that change what a chat client gets back: how
# reasoning and tool calls are split out of the text, and the chat template's
# defaults. A derived GPU spec copies them from the TT row it derives from, so
# chat requests (agentic evals, use_chat_api tasks) are answered alike.
FRONTEND_SERVE_KEYS = {
    "reasoning_parser", "tool_call_parser", "enable_auto_tool_choice",
    "default_chat_template_kwargs", "chat_template", "chat_template_content_format",
}  # fmt: skip


def frontend_args(vllm_args: Dict[str, Any]) -> Dict[str, Any]:
    """The FRONTEND_SERVE_KEYS of a spec's vllm_args (either spelling)."""
    return {
        k: v
        for k, v in (vllm_args or {}).items()
        if k.replace("-", "_") in FRONTEND_SERVE_KEYS
    }


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
# GPU throughput policy: the TT row's max_concurrency (often 32) leaves the GPU
# mostly idle. Up to ~8B parameters, serve with 256 sequences and send as many
# concurrent requests (the runner exports TT_EVAL_CLIENT_CONCURRENCY); up to
# 16B, as many as the GPU's spare memory holds at KV_TOKENS_PER_SEQ tokens
# each, at most 128 (~126 for Qwen3-14B on an 80 GB H100; on an A100-40GB that
# is below the row's value, which then stays). Larger models keep
# the row's value. Prompts, sampling and scoring are unchanged.
GPU_HIGH_CONCURRENCY = 256
GPU_HIGH_CONCURRENCY_MAX_PARAMS = 9e9
GPU_MID_CONCURRENCY = 128
GPU_MID_CONCURRENCY_MAX_PARAMS = 16e9
# KV tokens one in-flight eval request holds, from Qwen3-0.6B mmlu_pro on an
# A100 (32 running used 19.6% of a ~300k-token cache: ~1.8k each).
KV_TOKENS_PER_SEQ = 2048
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
    # hf_overrides can only set keys, and vLLM 0.13 dereferences
    # dual_chunk_attention_config whenever the attribute exists (even None). So a
    # None override means "remove the key": serve from a config.json without it
    # (--hf-config-path), which the runner writes with materialize-hf-config.
    overrides = vllm_args.pop("hf_overrides", None) or {}
    overrides = json.loads(overrides) if isinstance(overrides, str) else dict(overrides)
    drop = sorted(k for k, v in overrides.items() if v is None)
    if any(v is not None for v in overrides.values()):
        vllm_args["hf_overrides"] = {
            k: v for k, v in overrides.items() if v is not None
        }
    hf_config = None
    if drop:
        path = f"{REMOTE_DIR}/hf_configs/{repo.replace('/', '__')}"
        hf_config = {
            "path": path,
            "drop": drop,
            "revision": dms.vllm_args.get("revision"),
        }
        argv += ["--hf-config-path", path]
    return {
        "hf_config": hf_config,
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


# The only Hub documents gpuref reads; repo ids and revisions are restricted to
# the Hub's own characters, so a model name cannot leave these paths.
HF_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
HF_REPO = re.compile(rf"^{HF_NAME}/{HF_NAME}$")
HF_REV = re.compile(rf"^{HF_NAME}$")
HF_PATH = re.compile(
    rf"^/(?:api/models/{HF_NAME}/{HF_NAME}(?:/revision/{HF_NAME})?"
    rf"|{HF_NAME}/{HF_NAME}/resolve/{HF_NAME}/config\.json)$"
)


def hf_url_ok(url: str) -> bool:
    """True only for https://huggingface.co Hub model API / config.json URLs."""
    parts = urllib.parse.urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.netloc == "huggingface.co"
        and not parts.query
        and not parts.fragment
        and ".." not in parts.path
        and bool(HF_PATH.match(parts.path))
    )


def hf_get(url: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """GET a Hub JSON document; None on any HTTP/network/parse error, and for
    any URL other than a huggingface.co model API or config.json path (the
    token is only ever sent there)."""
    if not hf_url_ok(url):
        return None
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def kv_bytes_per_token(config: Dict[str, Any]) -> int:
    """bf16 K+V bytes per token for a decoder config.json. In a hybrid model
    (layer_types, e.g. Gated DeltaNet + full attention) only the attention
    layers grow with the context; linear-attention layers keep a fixed state."""
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    layers = int(config["num_hidden_layers"])
    types = config.get("layer_types")
    if isinstance(types, list) and types and any("linear" in str(t) for t in types):
        layers = sum(1 for t in types if "linear" not in str(t))
    return 2 * layers * kv_heads * head_dim * 2


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


def model_params(meta: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]]):
    """Parameter count: the Hub's safetensors total, else from config.json."""
    return ((meta or {}).get("safetensors") or {}).get("total") or params_from_config(
        config
    )


def kv_tokens_spare(estimate: Optional[Dict[str, Any]], gpu_mem_gib: Optional[float]):
    """KV-cache tokens left on a GPU after weights, activations and overhead."""
    per_token = (estimate or {}).get("kv_bytes_per_token")
    if not per_token or not gpu_mem_gib:
        return 0
    spare = gpu_mem_gib * GPU_MEMORY_UTILIZATION - OVERHEAD_GIB
    spare -= estimate["weights_gib"] + estimate.get("activation_gib", 0.0)
    return max(0, int(spare * GIB // per_token))


def gpu_concurrency(
    plan: Dict[str, Any],
    params: Optional[int],
    estimate: Optional[Dict[str, Any]] = None,
    gpu_mem_gib: Optional[float] = None,
) -> Dict[str, Any]:
    """The plan with the GPU throughput policy applied: --max-num-seqs and the
    eval client's concurrency (client_concurrency), and the rule that chose them."""
    row = conc = plan["max_concurrency"]
    rule = "row max_concurrency"
    kv_fit = kv_tokens_spare(estimate, gpu_mem_gib) // KV_TOKENS_PER_SEQ
    if params and params <= GPU_HIGH_CONCURRENCY_MAX_PARAMS:
        conc, rule = (
            max(row, GPU_HIGH_CONCURRENCY),
            f"<= {GPU_HIGH_CONCURRENCY_MAX_PARAMS:.0e} params",
        )
    elif params and params <= GPU_MID_CONCURRENCY_MAX_PARAMS and kv_fit > row:
        conc = min(GPU_MID_CONCURRENCY, kv_fit)
        rule = f"<= {GPU_MID_CONCURRENCY_MAX_PARAMS:.0e} params, KV fits {kv_fit} x {KV_TOKENS_PER_SEQ} tokens on {gpu_mem_gib:g} GiB"
    argv = list(plan["vllm_serve_args"])
    argv[argv.index("--max-num-seqs") + 1] = str(conc)
    return dict(
        plan, vllm_serve_args=argv, max_num_seqs=conc, client_concurrency=conc,
        concurrency_rule=rule,
    )  # fmt: skip


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
    params = model_params(meta, config)
    memory = None
    if params:
        ctx, conc = plan["max_context"], plan["max_concurrency"]
        memory = memory_estimate(params, config, ctx, conc)
        memory["verdicts"] = gpu_verdicts(memory, ctx)
    row = dict(plan, model=model, ok=True, gated=gated, memory=memory, config=config)
    row["params"] = params
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
            # No GPU spec at this commit: the runner derives one from the TT row.
            d = derive_entry(model, catalog_facts()[1].get(model, []), token)
            if d is None:
                rows.append(
                    {"model": model, "ok": False, "error": f"no GPU spec: {exc}"}
                )
                continue
            plan = dict(derived_plan(model, d), derived_gpu_spec=d)
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
    """(state, display lines, models finished, phase) from snippets/status.py
    output. A --serve runner whose model is up behind the tunnel is SERVING."""
    state, lines, done, phase, serving = None, [], 0, "", False
    for line in text.splitlines():
        if line.startswith("GPUREF_PHASE="):
            phase = line.split("=", 1)[1].strip()
        if line.startswith("GPUREF_STATE="):
            state = line.split("=", 1)[1].strip() or None
        if line.startswith("GPUREF_DONE="):
            done = int(line.split("=", 1)[1] or 0)
        if line.strip() == "GPUREF_SERVING=1":
            serving = True
        for prefix, label in STATUS_LABELS.items():
            if line.startswith(prefix):
                lines.append(label + line[len(prefix) :])
    if serving and state == "RUNNING":
        state = "SERVING"
    return state, lines, done, phase


def parse_balance(text: str) -> Optional[float]:
    match = re.search(r"Current balance:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def summary_rows(root: Path) -> List[Dict[str, Any]]:
    """Per-task scores under root (local reports_output/evals/data/report_data_*.json
    or Shield's report_*.json), each from the newest report with that model and
    task (agentic tasks run as one Shield job each, so one model can have
    several reports), with the GPU name from each provenance.json."""
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
        for section in report["sections"]:
            if not isinstance(section, dict) or section.get("kind") != "evals":
                continue
            data = section.get("data") or {}
            key = (model, data.get("task_name"))
            if key not in newest or stamp >= newest[key][0]:
                newest[key] = (stamp, meta, data)
    return [
        {
            "model": model,
            "device": meta.get("device"),
            "gpu": gpus.get(model),
            "task": data.get("task_name"),
            "score": data.get("score"),
            "published_score": data.get("published_score"),
            "gpu_reference_score": data.get("gpu_reference_score"),
            # The n TTIS graded this score with (None: ratio rule only, or a
            # report from before the noise rule).
            "noise_n": data.get("noise_n"),
            # Agentic (Harbor) blocks: trials run, for the full-set check.
            "n_trials": data.get("n_trials"),
        }
        for (model, _), (_, meta, data) in sorted(
            newest.items(), key=lambda kv: tuple(str(x) for x in kv[0])
        )
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


def run_out(
    cmd: List[str], cwd: Optional[Path] = None, timeout: int = 120
) -> Optional[str]:
    """stdout of an argv list (never a shell), or None if it fails, times out
    or is missing (e.g. no nvidia-smi off the VM, no git in an exported tree)."""
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# EvalTask fields that change what a task scores (after GPU device overrides).
# Transport-only fields (max_concurrent, batch_size, timeouts, log_samples,
# priority) and limits (full runs use none) are left out on purpose.
SCORE_FIELDS = (
    "task_name", "workflow_venv_type", "eval_class", "tokenizer_backend",
    "num_fewshot", "seed", "use_chat_api", "apply_chat_template",
    "fewshot_as_multiturn", "gen_kwargs", "propagate_seed_to_gen_kwargs",
    "model_kwargs", "include_path", "custom_dataset_kwargs", "request_body",
    "thinking", "thinking_kwarg", "capture_reasoning",
)  # fmt: skip


# Serving stack per model: the vLLM (and optionally transformers) a GPU
# reference is measured with, plus the server environment. The default is
# TTIS's own vLLM client pin. An architecture that vLLM 0.13.0 cannot serve
# maps to the minimum vLLM release that supports it, and to the transformers
# the TT side lowers it with. The runner builds one venv per stack, so a sweep
# with mixed stacks runs in one session. Provenance and the eval fingerprint
# record the stack.
DEFAULT_STACK = {
    "vllm": "0.13.0",
    "transformers": None,  # what vLLM resolves
    # FlashInfer's top-k/top-p sampler JIT-compiles at warmup and needs ninja
    # and a matching nvcc, which the Colab VM lacks; vLLM's PyTorch sampler
    # draws from the same distribution.
    "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
}
ARCH_STACKS = {
    # vllm-project/vllm#51655 (first in v0.28.0); TT side lowers with 5.15.0.
    "MuseGlimmerForConditionalGeneration": {"vllm": "0.28.0", "transformers": "5.15.0"},
    "MuseGlimmerForCausalLM": {"vllm": "0.28.0", "transformers": "5.15.0"},
    # gemma4 / gemma4_mm (first in v0.20.0), with the transformers v0.20.0 is
    # tested with (requirements/test/cuda.txt: 5.5.3). Not 5.15.0: its per-layer
    # configs raise AmbiguousGlobalPerLayerAttributeError on vLLM 0.20.0's
    # config.head_dim, so the server never starts (gemma-4 run, 2026-10-12).
    "Gemma4ForConditionalGeneration": {"vllm": "0.20.0", "transformers": "5.5.3"},
    "Gemma4ForCausalLM": {"vllm": "0.20.0", "transformers": "5.5.3"},
    # gemma4_unified (gemma-4-12B-it): first in vLLM v0.23.0, whose model code
    # imports transformers' gemma4_unified config, first released in 5.10.0
    # (vLLM's test pin 5.5.3 predates it; 5.14+ adds the per-layer configs
    # that break Gemma 4 serving, see above).
    "Gemma4UnifiedForConditionalGeneration": {"vllm": "0.23.0", "transformers": "5.10.0"},
    # qwen3_5: hybrid Gated DeltaNet + full attention (Qwen3.5/3.6/3.8-27B), first
    # in v0.17.0, with the transformers v0.17.0 is tested with
    # (requirements/test.txt: 4.57.5; its common.txt requires < 5).
    "Qwen3_5ForConditionalGeneration": {"vllm": "0.17.0", "transformers": "4.57.5"},
}  # fmt: skip


def serving_stack(
    config: Optional[Dict[str, Any]],
    vllm: Optional[str] = None,
    transformers: Optional[str] = None,
) -> Dict[str, Any]:
    """The stack for a model from its HF config's architectures (ARCH_STACKS,
    else DEFAULT_STACK). vllm/transformers override it (--vllm-version etc.)."""
    stack = dict(DEFAULT_STACK, env=dict(DEFAULT_STACK["env"]))
    for arch in (config or {}).get("architectures") or []:
        if arch in ARCH_STACKS:
            stack.update(ARCH_STACKS[arch])
            break
    if vllm:
        stack["vllm"] = vllm
    if transformers:
        stack["transformers"] = transformers
    return stack


def stack_installed(stack: Dict[str, Any], packages: Dict[str, Any]) -> bool:
    """True if the installed packages are the requested stack."""
    return packages.get("vllm") == stack.get("vllm") and (
        not stack.get("transformers")
        or packages.get("transformers") == stack.get("transformers")
    )


def pinned_lm_eval_commit(ttis_dir: Path = REPO_ROOT) -> Optional[str]:
    text = (Path(ttis_dir) / "requirements" / "evals-common.txt").read_text()
    found = re.search(r"lm-evaluation-harness\.git@([0-9a-f]{7,40})", text)
    return found.group(1) if found else None


def _canon(value: Any) -> Any:
    """JSON-stable form of an EvalTask value (enums by name, callables by name)."""
    if hasattr(value, "name") and hasattr(value, "value"):  # Enum
        return value.name
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


# HarborEvalConfig fields read from the runner's environment (where the
# sandboxes run, Harbor's own process timeout): not part of what is measured.
AGENTIC_ENV_FIELDS = {
    "environment_type",
    "environment_kwargs",
    "harbor_timeout_sec",
    "quiet",
    "yes",
}


def _agentic_canon(config: Any) -> Any:
    import dataclasses

    if config is None or not dataclasses.is_dataclass(config):
        return _canon(config)
    return {
        f.name: _canon(getattr(config, f.name))
        for f in dataclasses.fields(config)
        if f.name not in AGENTIC_ENV_FIELDS
    }


def eval_fingerprint(
    model: str,
    lm_eval_commit: Optional[str],
    stack: Optional[Dict[str, Any]],
    ttis_dir: Path = REPO_ROOT,
    venvs: Optional[set] = None,
) -> Optional[str]:
    """sha256 over everything that decides this model's GPU scores: the
    score-affecting fields of each lm-eval task (GPU overrides applied), its
    scorer, the lm-eval commit and the serving stack (vLLM, transformers pin,
    server env). A run counts for resume and for the report only when its
    fingerprint equals the plan's."""
    if not (lm_eval_commit and stack and stack.get("vllm")):
        return None
    sys.path.insert(0, str(ttis_dir))
    from reference_config.evals.eval_config import (
        _eval_config_map,
        resolve_task_for_device,
    )

    cfg = _eval_config_map.get(model)
    if cfg is None:
        return None
    tasks = []
    for task in cfg.tasks:
        if task.workflow_venv_type.name not in (venvs or SWEEP_VENVS):
            continue
        task = resolve_task_for_device(task, "GPU")
        entry = {f: _canon(getattr(task, f, None)) for f in SCORE_FIELDS}
        if task.workflow_venv_type.name in AGENTIC_VENVS:  # the Harbor job is the eval
            entry["agentic_eval_config"] = _agentic_canon(task.agentic_eval_config)
        if task.score:
            entry["score_func"] = _canon(task.score.score_func)
            entry["score_func_kwargs"] = _canon(task.score.score_func_kwargs)
        tasks.append(entry)
    blob = json.dumps(
        {"tasks": sorted(tasks, key=lambda t: t["task_name"]),
         "lm_eval": lm_eval_commit,
         "stack": {"vllm": stack["vllm"], "transformers": stack.get("transformers"),
                   "env": dict(sorted((stack.get("env") or {}).items()))}},
        sort_keys=True, default=str,
    )  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()


def task_kind(model: str, tasks: List[str], ttis_dir: Path = REPO_ROOT) -> str:
    """ "lm-eval" when every task is one of the model's lm-eval tasks (run on the
    VM by run.py), "agentic" when every task is an agentic one (served from the
    VM, run by Shield's endpoint tests: --serve). Raises ValueError otherwise."""
    sys.path.insert(0, str(ttis_dir))
    from reference_config.evals.eval_config import _eval_config_map

    cfg = _eval_config_map.get(model)
    venv = {t.task_name: t.workflow_venv_type.name for t in (cfg.tasks if cfg else [])}
    kinds = {
        "lm-eval" if venv.get(t) in SWEEP_VENVS else
        "agentic" if venv.get(t) in AGENTIC_VENVS else None
        for t in tasks
    }  # fmt: skip
    if not tasks or None in kinds or len(kinds) != 1:
        raise ValueError(
            f"{model}: --task {','.join(tasks)} must be all lm-eval or all agentic "
            f"tasks of the model ({', '.join(f'{k} {v}' for k, v in sorted(venv.items()))})"
        )
    return kinds.pop()


def task_selection(model: str, tasks: List[str], ttis_dir: Path = REPO_ROOT) -> str:
    """run.py --eval-samples JSON that runs only these tasks of this model, in
    full: a task mapped to null is selected but gets no --samples, so lm-eval
    runs every doc (llm_module/eval_configs.py, eval_command.py). Raises
    ValueError when a task is not one of the model's lm-eval tasks."""
    sys.path.insert(0, str(ttis_dir))
    from reference_config.evals.eval_config import _eval_config_map

    cfg = _eval_config_map.get(model)
    have = sorted(
        t.task_name
        for t in (cfg.tasks if cfg else [])
        if t.workflow_venv_type.name in SWEEP_VENVS
    )
    missing = [t for t in tasks if t not in have]
    if not tasks or missing:
        raise ValueError(
            f"{model}: task(s) {missing or tasks} not among its lm-eval tasks {have}"
        )
    return json.dumps({t: None for t in tasks}, sort_keys=True)


def client_concurrency_used(run_log: Path) -> List[int]:
    """Distinct num_concurrent values the lm-eval commands in run_py.log used."""
    try:
        text = run_log.read_text(errors="replace")
    except OSError:
        return []
    return sorted({int(n) for n in re.findall(r"num_concurrent=(\d+)", text)})


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
    packages = json.loads(versions) if versions else {}
    lm_eval_installed = next((c for c in installed if c), None)
    return {
        "model": args.model,
        "model_id": plan.get("model_id"),
        "status": args.status,
        "evals_exit_code": args.evals_rc,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "gpu": dict(zip(["name", "driver", "memory_total"], gpu + [None] * 3)),
        "packages": packages,
        # Server environment from the stack (set by remote_runner.sh).
        "vllm_env": (plan.get("stack") or {}).get("env") or {},
        "ttis_sha": run_out(["git", "rev-parse", "HEAD"], cwd=ttis),
        "model_revision_pinned": plan.get("revision"),
        "model_revision_sha": resolved.get("sha"),
        "max_context": plan.get("max_context"),
        "max_concurrency": plan.get("max_concurrency"),
        # GPU throughput policy (gpu_concurrency); None on older plans.
        "max_num_seqs": plan.get("max_num_seqs"),
        "client_concurrency": plan.get("client_concurrency"),
        # What the eval client actually used per task (num_concurrent in the
        # lm-eval commands): lower than client_concurrency when the pinned
        # commit predates TT_EVAL_CLIENT_CONCURRENCY. Scheduling only.
        "client_concurrency_effective": client_concurrency_used(
            Path(args.plan).parent / "run_py.log"
        ),
        "concurrency_rule": plan.get("concurrency_rule"),
        "placement": plan.get("placement"),
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
        # Config keys removed for GPU serving (e.g. dual_chunk_attention_config).
        "hf_config_dropped_keys": (plan.get("hf_config") or {}).get("drop"),
        # --serve: the tasks Shield runs against this server, and its URL
        # (redacted); the driver adds each Shield run (shield-provenance).
        "endpoint": args.endpoint,
        "run_py_command": [
            "python3",
            "run.py",
            "--workflow",
            "agentic" if args.agentic_tasks else "evals",
            "--tt-device",
            "gpu",
            "--model",
            args.model,
            "--dev-mode",
        ]
        + (
            ["--eval-samples", args.eval_samples] if args.eval_samples else []
        ),  # fmt: skip
        # Set when the run was restricted to some tasks (--task); each runs in full.
        "tasks_selected": sorted(json.loads(args.eval_samples))
        if args.eval_samples
        else sorted(args.agentic_tasks.split(","))
        if args.agentic_tasks
        else None,
        "lm_eval_commit_pinned": pinned.group(1) if pinned else None,
        "lm_eval_commit_installed": lm_eval_installed,
        # The requested serving stack, and the fingerprint of what ran: this
        # checkout's eval settings, the installed lm-eval and that stack, but
        # only if the installed packages are that stack (else None).
        "stack": plan.get("stack"),
        # Set when the pinned commit had no GPU spec and the runner derived one
        # from that commit's TT row (into the VM's clone only).
        "derived_gpu_spec": load_json(Path(args.plan).parent / "derived_gpu_spec.json"),
        # Agentic tasks: Shield's run.py drives Harbor, not lm-eval, so the
        # fingerprint covers the agentic configs and stack (lm-eval: pinned).
        "eval_fingerprint": (
            eval_fingerprint(
                args.model,
                (pinned.group(1) if pinned else None)
                if args.agentic_tasks
                else lm_eval_installed,
                plan.get("stack"),
                ttis,
                venvs=AGENTIC_VENVS if args.agentic_tasks else None,
            )
            if plan.get("stack") and stack_installed(plan["stack"], packages)
            else None
        ),  # fmt: skip
        "note": args.note,
        "request_evidence": load_json(Path(args.evidence)) if args.evidence else None,
    }


# --- serve: agentic evals through Shield's endpoint tests -----------------------
# Agentic tasks (Harbor: terminal-bench, SWE-bench) run every trial in its own
# sandbox, which a Colab VM cannot host. In --serve mode the VM only serves the
# model (vLLM behind --api-key, published through an ngrok tunnel); Shield's
# endpoint-test workflow runs TTIS's own `run.py --workflow agentic` against
# that URL with its sandboxes on the KIX1 cluster (HARBOR_ENV_TYPE=kubernetes),
# exactly as it runs them for a TT deployment.

SHIELD_REPO = "tenstorrent/tt-shield"
SHIELD_WORKFLOW = "on-dispatch-tt-blaze-exabox-tests.yml"
# The tt-shield repo secret holding the endpoint's API key (the TEST_* names are
# selectable from env-variables). Its value is the driver's key file.
ENDPOINT_KEY_SECRET = "TEST_GPUREF_ENDPOINT_API_KEY"
# Every checkbox off: the one test is the agentic workflow on this task, on the
# model's GPU spec (--device gpu overrides the workflow's super_cluster).
SHIELD_CHECKBOXES = (
    "test-sanity", "test-spec-tests", "test-evals-smoke", "test-evals-fast",
    "test-evals-full", "test-agentic", "test-agentic-full", "test-benchmark",
    "test-agentic-traces",
)  # fmt: skip


def shield_test(task: str) -> str:
    return (
        f"inference-workflow-agentic --dev-mode --device gpu --agentic-benchmark {task}"
    )


def shield_inputs(
    model: str, task: str, url: str, ttis_sha: str, secret: str = ENDPOINT_KEY_SECRET
) -> Dict[str, str]:
    """workflow_dispatch inputs (`gh workflow run --json`) for one agentic task
    against the served URL, at the pinned TTIS commit. The API key goes by
    secret name only (KEY=SECRET_<NAME>), never by value."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", task) or not re.fullmatch(
        r"[0-9a-f]{40}", ttis_sha
    ):
        raise ValueError("bad task name or TTIS sha")
    if not re.fullmatch(r"https://[A-Za-z0-9.-]+/?", url):
        raise ValueError("the endpoint must be a bare https://host URL")
    inputs = {box: "false" for box in SHIELD_CHECKBOXES}
    inputs.update({
        "service": "v1-old-stack", "model": model, "target": "None", "test_url": url,
        "env-variables": f"HARBOR_ENV_TYPE=kubernetes API_KEY=SECRET_{secret} "
                         f"OPENAI_API_KEY=SECRET_{secret}",
        "inference-server-git-ref": ttis_sha,
        "extra-tests": json.dumps([shield_test(task)]),
    })  # fmt: skip
    return inputs


def shield_find_run(runs: List[Dict[str, Any]], model: str, url: str, since: str):
    """databaseId of the dispatch this driver started: the newest
    workflow_dispatch run created at or after `since` whose run-name names the
    model and the URL ("Run tests | <service> | <model> | <url>")."""
    found = [
        r for r in runs
        if (r.get("createdAt") or "") >= since
        and model in (r.get("displayTitle") or "")
        and (r.get("displayTitle") or "").rstrip("/").endswith(url.rstrip("/"))
    ]  # fmt: skip
    return max(found, key=lambda r: r["createdAt"])["databaseId"] if found else None


def redact_url(url: Optional[str]) -> Optional[str]:
    """https://abcd…<tld>: enough to tell two tunnels apart, not to call one."""
    if not url:
        return None
    host = urllib.parse.urlsplit(url).hostname or ""
    parts = host.split(".")
    return (
        f"https://{parts[0][:4]}…{'.'.join(parts[-2:])}"
        if len(parts) > 2
        else "https://…"
    )


def shield_provenance(
    prov: Dict[str, Any], records: List[Dict[str, Any]], shield_root: Path
) -> Dict[str, Any]:
    """The served model's provenance with each task's Shield run: run id/url,
    conclusion, the dispatch inputs and the run.py report it wrote.

    Two statuses, never conflated: ``serving_status`` is whether the VM served
    (the runner's own ``status``, copied on the first grading and never
    rewritten), and ``status`` is the overall result, recomputed on every call
    from ``serving_status`` and the reports present now. So a pass that grades
    before a report arrives (``failed``) does not stop a later pass that
    retrieves it from grading ``ok``. status is ok only when the VM served and
    every task's report (device GPU) has a score."""
    rows = {
        r["task"]: r
        for r in summary_rows(shield_root)
        if r["model"] == prov.get("model")
        and r["device"] == "GPU"
        and r["score"] is not None
    }
    prov = dict(prov)
    prov["agentic"] = [
        dict(rec, score=(rows.get(rec.get("task")) or {}).get("score"),
             n_trials=(rows.get(rec.get("task")) or {}).get("n_trials"),
             report_files=sorted(str(p.relative_to(shield_root))
                                 for p in (shield_root / rec.get("task", "")).rglob("report*.json")))
        for rec in records
    ]  # fmt: skip
    prov["run_py_command"] = [shield_test(rec.get("task", "")) for rec in records]
    prov["tasks_selected"] = sorted(rec.get("task") for rec in records)
    if "serving_status" not in prov:
        prov["serving_status"] = prov.get("status")
    served = prov["serving_status"] == "ok"
    # A completed run counts whether or not the model met its acceptance
    # criteria (Shield fails the job on a miss); a cancelled or crashed one not.
    prov["status"] = "ok" if served and records and all(
        rec.get("conclusion") in ("success", "failure") and rec.get("task") in rows
        for rec in records
    ) else "failed"  # fmt: skip
    return prov


# --- unattended monitoring ----------------------------------------------------


def driver_status(
    path: Path, log: Optional[Path], fields: Dict[str, Any]
) -> Dict[str, Any]:
    """Write the driver's state as JSON (atomically) and append one summary line
    to the shared log, so an agent or person reads state instead of tailing
    driver output. States: STARTING, WAITING_CAPACITY, RUNNING, DONE, FAILED,
    NO_CAPACITY, LOW_BALANCE (and DEFERRED / SWEEP_* for a sweep)."""
    import datetime

    record = {k: v for k, v in fields.items() if v not in (None, "")}
    record["updated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    if log is not None:
        line = " ".join(
            f"{k}={str(v).replace(' ', '_')}"
            for k, v in record.items()
            if k != "updated_at"
        )
        with open(log, "a") as handle:
            handle.write(f"{record['updated_at']} {line}\n")
    return record


# --- sweep: pick targets from the catalog, place, run, report ----------------

SWEEP_VENVS = {"EVALS_COMMON"}  # tasks the lm-eval evals path can run on a GPU
AGENTIC_VENVS = {"EVALS_AGENTIC"}  # served from the VM, run by Shield (--serve)
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
                # For TTIS's noise rule (binomial_noise_n): the score's keys and unit.
                "result_keys": t.score.score_func_kwargs.get("result_keys"),
                "unit": t.score.score_func_kwargs.get("unit"),
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
            "frontend": frontend_args(d.vllm_args),
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
                "result_keys": t.get("result_keys"),
                "unit": t.get("unit"),
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
        "revision": row["revision"], "frontend": row.get("frontend") or {},
        "source": f"{row['impl']} {row['device']} row max_context {row['max_context']}, "
        f"native {native_ctx}",
    }  # fmt: skip


def derive_entry(
    model: str, model_rows, token: Optional[str]
) -> Optional[Dict[str, Any]]:
    """A derived GPU spec entry for a model from its TT rows and HF config:
    derive_gpu_spec, the pinned revision, hf_overrides and text-only serving."""
    rev = next((r["revision"] for r in model_rows if r["impl"] == "quetzal"), None)
    config = hf_get(f"{HF}/{model}/resolve/{rev or 'main'}/config.json", token) or {}
    native = config.get("max_position_embeddings") or (
        config.get("text_config") or {}
    ).get("max_position_embeddings")
    d = derive_gpu_spec(model_rows, native)
    if d is None:
        return None
    if not d["revision"]:  # pin the Hub's current main
        d["revision"] = (
            hf_get(f"{HF}/api/models/{model}/revision/main", token) or {}
        ).get("sha")
    d["hf_overrides"] = derive_overrides(config)
    d["limit_mm_per_prompt"] = derive_text_only(config)
    return d


def derived_plan(model: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """The serve plan a derived entry gives (as if it were in the catalog)."""
    from workflows.model_spec import DeviceModelSpec, DeviceTypes, get_model_id

    dms = DeviceModelSpec(device=DeviceTypes.GPU, max_concurrency=d["max_concurrency"],
                          max_context=d["max_context"], default_impl=True,
                          vllm_args=derived_vllm_args(d))  # fmt: skip
    dms.vllm_args.setdefault("model", model)
    model_id = get_model_id(d["impl"].replace("_", "-"), model.split("/")[-1], "gpu")
    return serve_plan(
        SimpleNamespace(device_model_spec=dms, hf_model_repo=model, model_id=model_id)
    )


def ensure_gpu_spec(model: str, ttis_dir: Path) -> Optional[Dict[str, Any]]:
    """Make the checkout at ttis_dir serve `model` on GPU: if its catalog has no
    GPU spec, derive one from that checkout's TT row and merge it into its
    llm.yaml (the VM's clone only). Returns the derived entry, or None when a
    spec already existed. Raises ValueError when there is no row to derive from."""
    try:
        load_gpu_spec(model, ttis_dir)
        return None
    except Exception:
        pass
    _, rows = catalog_facts(ttis_dir)
    d = derive_entry(model, rows.get(model, []), hf_token())
    if d is None:
        raise ValueError(f"{model}: no GPU spec and no TT row to derive one from")
    path = Path(ttis_dir) / "workflows" / "model_specs" / "dev" / "llm.yaml"
    path.write_text(write_derived_specs(path.read_text(), {model: d}))
    return d


def gpu_group(verdicts: Dict[str, Any]) -> Optional[str]:
    """Run list for a model: "A100" if its bf16 weights fit an A100 (fully or
    with a cap) -- an A100 bills ~5.3 CU/hr against ~18 for an H100 and the
    scores are valid on either, so no H100 fallback -- "H100" if it needs
    80 GB, None if nothing fits."""

    def fits(gpu):
        return verdicts.get(gpu) == "fits" or isinstance(verdicts.get(gpu), dict)

    return "A100" if fits("A100") else "H100" if fits("H100") else None


def gpu_placement(
    verdicts: Dict[str, Any],
    params: Optional[int] = None,
    estimate: Optional[Dict[str, Any]] = None,
    row_concurrency: Optional[int] = None,
):
    """(group, reason). gpu_group, except that a 10-16B model whose A100 KV
    room (KV_TOKENS_PER_SEQ per request) is below its row's max_concurrency
    goes to the H100 group: there it serves up to 128 sequences, at about the
    same CU in a third of the wall time (cost is why A100 comes first)."""
    group = gpu_group(verdicts)
    if group != "A100":
        return group, None if group is None else "needs an 80 GB GPU"
    fit = kv_tokens_spare(estimate, GPU_MEMORY_GIB["A100"]) // KV_TOKENS_PER_SEQ
    if (
        params
        and GPU_HIGH_CONCURRENCY_MAX_PARAMS < params <= GPU_MID_CONCURRENCY_MAX_PARAMS
        and row_concurrency
        and fit < row_concurrency
        and gpu_group({"A100": "no", "H100": verdicts.get("H100")}) == "H100"
    ):
        return "H100", (
            f"{params / 1e9:.1f}B params: an A100 holds {fit} sequences of KV "
            f"(< row {row_concurrency}); an H100 serves up to {GPU_MID_CONCURRENCY}"
        )
    return "A100", "fits an A100 (cheapest)"


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


def derive_text_only(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """vLLM limit_mm_per_prompt for a multimodal checkpoint: zero images and
    videos. The TT row serves the text decoder and the evals send text only,
    so the GPU server skips the vision encoder's memory profiling. Text
    prompts are processed identically."""
    if "vision_config" in config:
        return {"image": 0, "video": 0}
    return None


def derived_vllm_args(d: Dict[str, Any]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if d.get("revision"):
        args.update(revision=d["revision"], tokenizer_revision=d["revision"])
    if d.get("hf_overrides"):
        args["hf_overrides"] = d["hf_overrides"]
    if d.get("limit_mm_per_prompt"):
        args["limit_mm_per_prompt"] = d["limit_mm_per_prompt"]
    args.update(d.get("frontend") or {})
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
            elif key == "limit_mm_per_prompt":
                out.append("      # text only, as the TT row serves the text decoder")
                value = json.dumps(value)
            elif isinstance(value, str) and not re.fullmatch(
                r"[A-Za-z0-9._/-]+", value
            ):
                value = json.dumps(value)  # a YAML-safe scalar (e.g. a JSON object)
            elif isinstance(value, bool):
                value = str(value).lower()
            out.append(f"      {key}: {value}")
        out.append("  status: EXPERIMENTAL")
    return "\n".join(out)


def block_entries(text: str) -> Dict[str, str]:
    """model -> its verbatim YAML entry inside the generated block of llm.yaml."""
    if SPEC_BEGIN not in text:
        return {}
    block = text[text.index(SPEC_BEGIN) + len(SPEC_BEGIN) : text.index(SPEC_END)]
    starts = [
        (m.start(), m.group(1))
        for m in re.finditer(r"^- weights:\n  - (\S+)", block, re.M)
    ]
    ends = [start for start, _ in starts[1:]] + [len(block)]
    return {
        name: block[start:end].strip("\n") for (start, name), end in zip(starts, ends)
    }


def handwritten_gpu_models(text: str) -> set:
    """Models with a GPU spec in llm.yaml outside the generated block (reviewed
    templates). The generator never derives or keeps an entry for these."""
    import yaml

    if SPEC_BEGIN in text:
        end = text.index(SPEC_END) + len(SPEC_END)
        text = text[: text.index(SPEC_BEGIN)] + text[end:]
    found = set()
    for tpl in (yaml.safe_load(text) or {}).get("templates") or []:
        if any(d.get("device") == "GPU" for d in tpl.get("device_model_specs") or []):
            found.update(tpl.get("weights") or [])
    return found


def block_models(text: str) -> List[str]:
    """Models whose GPU spec lives in the generated block of llm.yaml."""
    return list(block_entries(text))


def write_derived_specs(
    text: str, derived: Dict[str, Dict[str, Any]], prune: bool = False
) -> str:
    """llm.yaml text with `derived` merged into the generated block by model:
    those entries are added or replaced, every other generated entry is kept
    as it is (a sweep selects only the models it measures), unless prune,
    which drops the entries not in `derived`. Models with a GPU spec outside
    the block (handwritten_gpu_models) never get one. Entries are kept sorted."""
    entries = {} if prune else block_entries(text)
    entries.update({m: derived_specs_yaml({m: d}) for m, d in derived.items()})
    # A reviewed GPU template always wins: never add (or keep) a generated copy.
    for model in handwritten_gpu_models(text) & set(entries):
        del entries[model]
    body = "\n".join(entries[m] for m in sorted(entries))
    block = f"{SPEC_BEGIN}\n{body}\n{SPEC_END}\n" if entries else ""
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


def serves_plan(prov: Dict[str, Any], argv: Optional[List[str]]) -> bool:
    """The run served exactly this vllm serve argv (revision, context, flags).
    --max-num-seqs is ignored: it schedules requests, not what they compute."""

    def semantic(args):
        args = list(args)
        if "--max-num-seqs" in args:
            del args[args.index("--max-num-seqs") : args.index("--max-num-seqs") + 2]
        return args

    served = (prov.get("vllm_serve_command") or [None])[1:]
    return argv is not None and semantic(served) == semantic(argv)


def run_covers(
    prov: Dict[str, Any],
    results_dir: Path,
    argv: List[str],
    tasks: List[str],
    fingerprint: Optional[str],
) -> bool:
    """A finished run counts for resume only if it served exactly this plan
    (same vllm serve args: revision, context, flags), ran the same eval
    settings, lm-eval and vLLM (eval_fingerprint; runs recorded without one
    never count) and has a score for every task. --max-num-seqs is ignored:
    it schedules requests, not what they compute."""

    scored = {
        r["task"]
        for r in summary_rows(results_dir.parent)
        if r["model"] == prov.get("model") and r["score"] is not None
    }
    return (
        fingerprint is not None
        and prov.get("eval_fingerprint") == fingerprint
        and serves_plan(prov, argv)
        and set(tasks) <= scored
    )


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
        # A ref starting with "-" would reach git as an option.
        if ref.startswith("-") or not re.match(r"^[A-Za-z0-9._/-]+$", ref):
            raise ValueError(f"not a git ref: {ref!r}")
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


def sweep_plan(
    sha: str,
    results_root: Path,
    refresh: bool = False,
    audit_refs=None,
    vllm_version: Optional[str] = None,
    prune: bool = False,
    transformers_version: Optional[str] = None,
):
    tasks, rows = catalog_facts()
    lm_eval = pinned_lm_eval_commit()
    audit: Dict[str, str] = (
        quetzal_models(rows, audit_refs) if audit_refs is not None else {}
    )
    targets, skipped = select_targets(tasks, rows, refresh, frozenset(audit))
    for model, where in sorted(audit.items()):
        if model not in tasks:
            skipped.append({"model": model, "task": "*", "reason": "audit: Quetzal P300X2 row",
                            "skip": f"row on {where}; no EvalConfig on this branch"})  # fmt: skip
    yaml_text = (
        REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml"
    ).read_text()
    generated = set(block_models(yaml_text)) - handwritten_gpu_models(yaml_text)
    token, derived, groups, placed = hf_token(), {}, {}, {}
    finished = finished_models(results_root)
    for model in sorted({t["model"] for t in targets}):
        has_gpu = any(r["device"] == "GPU" for r in rows[model])
        if has_gpu and model not in generated:
            plan = serve_plan(load_gpu_spec(model))
        else:  # (re)derive: no GPU spec yet, or one this sweep generated
            d = derive_entry(model, rows[model], token)
            derived[model] = d
            plan = derived_plan(model, d)
        facts = model_facts(model, plan, token)
        memory = facts["memory"] or {}
        group, placement = gpu_placement(
            memory.get("verdicts") or {},
            facts["params"],
            memory,
            plan["max_concurrency"],
        )
        plan = gpu_concurrency(
            plan, facts["params"], memory, GPU_MEMORY_GIB.get(group or "")
        )
        model_tasks = [t["task"] for t in targets if t["model"] == model]
        # run.py evaluates every lm-eval task of the model, targeted or not.
        model_tasks_all = {
            model: [t["task"] for t in tasks[model] if t["venv"] in SWEEP_VENVS]
        }
        prior = finished.get(model)
        placed[model] = {
            "group": group,
            "placement": placement,
            "moved_to_h100": group != gpu_group(memory.get("verdicts") or {}),
            "max_num_seqs": plan["max_num_seqs"],
            "max_context": plan["max_context"],
            "need_gib": memory.get("need_gib"),
            "est_hours": estimate_hours(model_tasks_all.get(model, []), memory.get("weights_gib")),
            "stack": (stack := serving_stack(facts.get("config"), vllm_version, transformers_version)),
            "eval_fingerprint": (fp := eval_fingerprint(model, lm_eval, stack)),
            "vllm_serve_args": plan["vllm_serve_args"],
            "done": bool(prior) and run_covers(prior[2], prior[1], plan["vllm_serve_args"], model_tasks, fp),
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
    probe_first(groups, placed)
    placeable = {t["model"] for t in kept}
    derived = {m: d for m, d in derived.items() if m in placeable}
    stale = write_derived_specs(yaml_text, derived, prune) != yaml_text
    estimate = {}
    for group, models in groups.items():
        hours = round(sum(placed[m]["est_hours"] for m in models), 1)
        gpu = group.split(",")[0]
        estimate[group] = {"models": len(models), "hours": hours,
                           "cu_at_first_choice": round(hours * CU_PER_HOUR[gpu]),
                           "cu_at_a100_rate": round(hours * CU_PER_HOUR["A100"])}  # fmt: skip
    return {"sha": sha, "targets": kept, "skipped": skipped, "groups": groups,
            "derived": derived, "specs_stale": stale, "estimate": estimate,
            "lm_eval_commit": lm_eval, "vllm_override": vllm_version,
            "transformers_override": transformers_version, "prune": prune,
            "legacy_runs": sorted(m for m in placed if m in finished
                                  and not finished[m][2].get("eval_fingerprint"))}  # fmt: skip


def probe_first(groups: Dict[str, List[str]], placed: Dict[str, Any]) -> None:
    """Measurement probe first: in each group, the costliest model that
    gpu_placement moved to the H100 runs first, so its measured CU can decide
    the rest before they run. The other models keep their order."""
    for models in groups.values():
        moved = [m for m in models if placed[m].get("moved_to_h100")]
        if moved:
            probe = max(moved, key=lambda m: placed[m]["est_hours"])
            models.remove(probe)
            models.insert(0, probe)


def patch_eval_config(text: str, results: List[Dict[str, Any]], ref_url: str) -> str:
    """eval_config.py text with each result's gpu_reference_score/_ref set and
    its gpu_reference_requested cleared. Proposed only; never applied here.
    A result whose model or task this file does not have is left out."""
    for r in results:
        start = text.find(f'        hf_model_repo="{r["model"]}",')
        if start < 0:
            continue
        end = text.find("\n    EvalConfig(", start)
        end = len(text) if end < 0 else end
        block = text[start:end]
        at = block.find(f'task_name="{r["task"]}"')
        if at < 0:
            continue
        nxt = block.find("EvalTask(", at)
        nxt = len(block) if nxt < 0 else nxt
        task = block[at:nxt]
        task = re.sub(r"\n\s*gpu_reference_requested=(\"[^\"]*\"|'[^']*'),", "", task)
        if r.get("verdict") == "keep":  # matches the old reference within noise
            text = text[:start] + block[:at] + task + block[nxt:] + text[end:]
            continue
        if "gpu_reference_score=" not in task:
            # The task omits both fields (dataclass defaults): add them after
            # published_score_ref, at its indent, so a "record" is not dropped.
            m = re.search(r"\n(\s*)published_score_ref=[^\n]+,", task)
            if m is None:
                continue
            ind = m.group(1)
            task = (task[: m.end()] + f"\n{ind}gpu_reference_score=None,"
                    f"\n{ind}gpu_reference_score_ref=None," + task[m.end():])  # fmt: skip
        task = re.sub(
            r"gpu_reference_score=[^,\n]+,",
            f"gpu_reference_score={r['score']:.2f},",
            task,
            count=1,
        )
        task = re.sub(
            r"gpu_reference_score_ref=[^\n]+,",
            f'gpu_reference_score_ref="{ref_url}",',
            task,
            count=1,
        )
        text = text[:start] + block[:at] + task + block[nxt:] + text[end:]
    return text


def sweep_results(
    plan, results_root: Path, tt_root: Optional[Path], strict: bool = True
) -> List[Dict[str, Any]]:
    """One row per target: GPU score from the newest ok run of its model, the
    evidence/cap/GPU from provenance, and the TT score from Shield reports.
    strict (a sweep plan): only a run whose eval fingerprint equals the plan's
    counts. Not strict (explicit models): the newest ok run, as recorded."""
    done = finished_models(results_root)
    tt = {}
    if tt_root:
        for row in summary_rows(tt_root):
            if row["device"] != "GPU":
                tt[(row["model"], row["task"])] = (row["score"], row["noise_n"])
    rows = []
    for t in plan["targets"]:
        tt_score, tt_n = tt.get((t["model"], t["task"]), (None, None))
        row = dict(t, tt_score=tt_score, tt_noise_n=tt_n, score=None)
        prior = done.get(t["model"])
        if (
            strict
            and prior
            and prior[2].get("eval_fingerprint") != t.get("eval_fingerprint")
        ):
            # Run with other eval settings, lm-eval or vLLM (or recorded before
            # fingerprints): not a reference for this plan.
            row["note"] = "stale: eval fingerprint differs from the plan"
            prior = None
        elif strict and prior and not serves_plan(prior[2], t.get("vllm_serve_args")):
            # Same eval settings but other serving (revision, context, flags), or
            # a plan written before it recorded them: re-plan and re-run.
            row["note"] = "stale: vllm serve args differ from the plan"
            prior = None
        if prior:
            _, results_dir, prov = prior
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
            counts = {k: v["samples"] for k, v in evidence.items()}
            counts.setdefault(t["task"], sum(v["samples"] for v in sub) or None)
            row.update(
                max_num_batched_tokens=chunk,
                score=scores.get((t["model"], t["task"])),
                samples=sum(v["samples"] for v in sub) or None,
                noise_n=noise_n(t, counts),
                max_request_tokens=max((v["max_tokens"] for v in sub), default=None),
                cap=prov.get("max_model_len_cap"), gpu=(prov.get("gpu") or {}).get("name"),
                ttis_sha=prov.get("ttis_sha"), revision=prov.get("model_revision_sha"),
                vllm=(prov.get("packages") or {}).get("vllm"), results=str(results_dir),
                transformers=(prov.get("packages") or {}).get("transformers"),
                vllm_env=prov.get("vllm_env") or {},
            )  # fmt: skip
        row["verdict"] = reference_verdict(
            row.get("old_gpu_ref"), row["score"], row.get("noise_n")
        )
        current = row.get("old_gpu_ref") or row.get("published")
        proposed = current if row["verdict"] in (None, "keep") else row["score"]
        # The GPU run's n decides the reference (above); a TT score is graded
        # with the n of its own report, as TTIS grades it.
        tol = row.get("tolerance", 0.05)
        row["tt_now"] = tt_grade(row["tt_score"], current, tol, tt_n)
        row["tt_proposed"] = tt_grade(row["tt_score"], proposed, tol, tt_n)
        rows.append(row)
    return rows


def eval_noise():
    """TTIS's noise rule (workflow_module/eval_noise.py, no third-party
    imports). Imported lazily: on the VM gpuref.py runs outside the checkout."""
    sys.path.insert(0, str(REPO_ROOT))
    from workflow_module import eval_noise as mod

    return mod


def noise_n(target: Dict[str, Any], counts: Dict[str, Any]) -> Optional[float]:
    """The n TTIS's noise rule uses for this task (binomial_noise_n), or None
    when TTIS grades it by the ratio rule alone (not percent, not binomial)."""
    if target.get("unit") != "percent":
        return None
    keys = [
        tuple(k) if isinstance(k, list) else k for k in target.get("result_keys") or []
    ]
    return eval_noise().binomial_noise_n(keys, target["task"], counts)


def tt_grade(score, ref, tolerance: float, n) -> str:
    """TTIS's rule for a TT score against a reference (noise_aware_check): PASS
    on the ratio rule (score/ref >= 1 - tolerance) or within noise, else FAIL;
    NA without a score or reference. n is noise_n (None: ratio rule only)."""
    if score is None or not ref:
        return "NA"
    passed, _, _ = eval_noise().noise_aware_check(
        score / ref >= 1 - tolerance, score, ref, n
    )
    return "PASS" if passed else "FAIL"


def reference_verdict(
    old: Optional[float], new: Optional[float], n: Optional[float]
) -> Optional[str]:
    """ "new" (no old reference), "keep" (new within TTIS's noise band of the
    old, either direction: |new-old| <= NOISE_Z binomial SE at the old score),
    or "replace". n is noise_n; without one only an identical score keeps."""
    if new is None:
        return None
    if old is None:
        return "new"
    mod = eval_noise()
    _, _, se = mod.noise_aware_check(False, new, old, n)
    return "keep" if abs(new - old) <= mod.NOISE_Z * (se or 0.0) else "replace"


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
    stacks = sorted({
        f"vLLM {r['vllm']} / transformers {r.get('transformers') or '-'}"
        + "".join(f", {k}={v}" for k, v in sorted((r.get("vllm_env") or {}).items()) if v)
        for r in rows if r.get("vllm")
    })  # fmt: skip
    out += ["", f"TTIS sha(s): {', '.join(shas) or '-'}; "
            f"{'; '.join(stacks) or 'vLLM -'}; bf16; "
            "full sample counts; per-model provenance.json (HF revision, exact vllm serve "
            "args, request evidence) in each results dir. GPU prefill chunk "
            "(--max-num-batched-tokens) is capped at 16384: it changes scheduling only, "
            "not results."]  # fmt: skip
    if skipped:
        out += ["", "Skipped:", ""] + [
            f"- {s['model']} {s['task']}: {s['skip']}" for s in skipped
        ]
    return "\n".join(out) + "\n"


# --- publish: #5353 comment + proposed eval_config.patch ---------------------

ISSUE_REPO = "tenstorrent/tt-inference-server"
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def explicit_targets(models: List[str], results_root: Path) -> List[Dict[str, Any]]:
    """Targets for a run of explicit models (not a sweep plan): every task its
    newest ok run scored, with the catalog's references when this checkout
    has the model's EvalConfig (otherwise none, and the ratio rule only)."""
    try:
        tasks, _ = catalog_facts()
    except Exception:  # an exported tree without the catalog imports
        tasks = {}
    catalog = {(repo, t["task"]): t for repo, ts in tasks.items() for t in ts}
    done = finished_models(results_root)
    targets = []
    for model in models:
        if model not in done:
            continue
        for row in summary_rows(done[model][1].parent):
            if row["model"] != model:  # a results dir holds only its GPU runs
                continue
            t = catalog.get((model, row["task"]), {})
            targets.append({
                "model": model, "task": row["task"], "reason": "explicit run",
                "old_gpu_ref": t.get("gpu_ref"), "published": t.get("published"),
                "tolerance": t.get("tolerance", 0.05),
                "result_keys": t.get("result_keys"), "unit": t.get("unit"),
            })  # fmt: skip
    return targets


def render_publication(
    out: Path,
    results_root: Path,
    plan: Optional[Dict[str, Any]] = None,
    models: Optional[List[str]] = None,
    tt_root: Optional[Path] = None,
    eval_config: Optional[Path] = None,
    ref_url: str = ISSUE_URL,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Write sweep_summary.json, the issue comment (sweep_5353.md) and the
    proposed eval_config.patch (against eval_config, default this checkout's)
    into out. Nothing is posted or applied."""
    import difflib

    if plan is not None:
        rows = sweep_results(plan, results_root, tt_root, strict=True)
        skipped, sha = plan.get("skipped", []), plan.get("sha")
    else:
        targets = explicit_targets(models or [], results_root)
        rows = sweep_results({"targets": targets}, results_root, tt_root, strict=False)
        skipped, sha = [], None
    out.mkdir(parents=True, exist_ok=True)
    summary = {"sha": sha, "results": rows, "skipped": skipped}
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = sweep_markdown(rows, skipped)
    if note:  # context for the reader, after the heading
        head, _, rest = markdown.partition("\n\n")
        markdown = f"{head}\n\n{note.strip()}\n\n{rest}"
    (out / "sweep_5353.md").write_text(markdown)
    finished = [r for r in rows if r["score"] is not None]
    patch_path = out / "eval_config.patch"
    if finished:
        path = (
            eval_config or REPO_ROOT / "reference_config" / "evals" / "eval_config.py"
        )
        old = Path(path).read_text()
        new = patch_eval_config(old, finished, ref_url)
        name = "reference_config/evals/eval_config.py"
        diff = difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"
        )
        patch_path.write_text("".join(diff))
    return {"rows": rows, "markdown": markdown, "finished": len(finished),
            "patch": str(patch_path) if finished else None}  # fmt: skip


def post_publication(out: Path, issue: int, render) -> str:
    """Post out/sweep_5353.md as a comment on the issue (gh), then re-render the
    patch citing that comment. Refuses a token-like string and a second post."""
    state_path = out / "publish.json"
    state = load_json(state_path) or {}
    if state.get("posted_url"):
        raise SystemExit(f"already posted: {state['posted_url']} (see {state_path})")
    body = out / "sweep_5353.md"
    if TOKEN_PATTERN.search(body.read_text()):
        raise SystemExit(f"refusing to post: {body} contains a token-like string")
    posted = subprocess.run(
        ["gh", "issue", "comment", str(issue), "--repo", ISSUE_REPO,
         "--body-file", str(body)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    url = posted.stdout.strip().splitlines()[-1] if posted.returncode == 0 else ""
    if not url.startswith("https://github.com/"):
        raise SystemExit(f"gh issue comment failed: {posted.stderr.strip()}")
    render(url)  # the patch now cites the posted comment
    state.update(posted_url=url, issue=issue)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return url


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
    add("serve-plan", model=req, ttis_dir={"default": str(REPO_ROOT)},
        vllm_version={}, transformers_version={})  # fmt: skip
    add("plan-argv", "plan")
    add("ensure-gpu-spec", model=req, ttis_dir=req, out=req)
    add("task-selection", "model", "tasks", ttis_dir={"default": str(REPO_ROOT)})
    p = sub.add_parser("plan-stack", help="a serve plan's stack: 'VLLM [TRANSFORMERS]'")
    p.add_argument("plan")
    p.add_argument(
        "--env", action="store_true", help="print its env as KEY=VALUE lines"
    )
    add("served", "model")
    add("record-status", "file", "model", "status", "note")
    add("request-evidence", model=req, revision={}, samples_root=req, out=req,
        log={"action": "append", "default": []})  # fmt: skip
    add("cap-bound", "evidence", "cap")
    p = driver_status_parser = sub.add_parser(
        "driver-status", help="write driver_status.json + a log line"
    )
    p.add_argument("file")
    p.add_argument("--log", default=None)
    p.add_argument("fields", nargs="*", metavar="KEY=VALUE")
    add("materialize-hf-config", "plan")
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
        "--write-specs",
        action="store_true",
        help="merge this run's derived GPU specs into llm.yaml (others are kept)",
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help="with --write-specs: drop generated specs of models not selected this run",
    )
    p.add_argument(
        "--vllm-version",
        default=None,
        help="vLLM the runner will use (default: remote_runner.sh's pin)",
    )
    p.add_argument(
        "--transformers-version",
        default=None,
        help="transformers the runner will use (default: each model's stack)",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--tt-reports", default=None, help="dir with Shield eval report JSONs"
    )
    p.add_argument("--emit-patch", action="store_true", help="(always written now)")
    p.add_argument("--ref-url", default=ISSUE_URL)
    p = sub.add_parser(
        "publish",
        help="render the issue comment and proposed eval_config.patch from results",
    )
    p.add_argument("--plan", default=None, help="sweep plan JSON (strict fingerprints)")
    p.add_argument("--models", nargs="+", default=None, help="explicit runs instead")
    p.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "workflow_logs" / "gpu_reference_colab"),
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tt-reports", default=None)
    p.add_argument(
        "--eval-config", default=None, help="eval_config.py to patch (default: here)"
    )
    p.add_argument("--ref-url", default=ISSUE_URL)
    p.add_argument("--note", default=None, help="context paragraph for the comment")
    p.add_argument("--issue", type=int, default=int(ISSUE_URL.rsplit("/", 1)[1]))
    p.add_argument(
        "--post",
        action="store_true",
        help="post the comment with gh (only after approval) and cite it in the patch",
    )
    add("provenance", out=req, model=req, ttis_dir=req, plan=req, vllm_python=req,
        vllm_bin=req, status=req, evals_rc={"type": int}, started_at=req,
        finished_at=req, note={}, max_model_len_cap={"type": int}, evidence={},
        eval_samples={}, agentic_tasks={}, endpoint={})  # fmt: skip
    # --serve (agentic tasks): see the "serve" section above.
    add("task-kind", "model", "tasks", ttis_dir={"default": str(REPO_ROOT)})
    p = sub.add_parser(
        "write-gpu-specs", help="derive GPU specs from the TT rows into llm.yaml"
    )
    p.add_argument("models", nargs="+")
    add("shield-inputs", model=req, task=req, url_file=req, ttis_sha=req,
        secret={"default": ENDPOINT_KEY_SECRET})  # fmt: skip
    add("shield-find-run", model=req, url_file=req, since=req)
    add("redact-url", url_file=req)
    add("shield-workflow")
    p = sub.add_parser(
        "shield-provenance", help="add the Shield runs to provenance.json"
    )
    p.add_argument("--prov", required=True)
    p.add_argument("--shield-dir", required=True)
    p.add_argument("--records", nargs="+", required=True)
    p.add_argument("--rollovers", type=int, default=0)
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["driver-status"]:
        # FILE --log LOG KEY=VALUE...: older argparse (Python 3.12.3, the system
        # python on exabox) binds the empty fields list next to FILE and then
        # rejects every KEY=VALUE after --log, so no status file was written.
        # 3.12.12 and 3.14 accept it; intermixed parsing works on all of them.
        args = driver_status_parser.parse_intermixed_args(argv[1:])
        args.cmd = "driver-status"
    else:
        args = parser.parse_args(argv)
    for name in [getattr(args, "model", None)] + (getattr(args, "models", None) or []):
        if name is not None and not HF_REPO.match(name):
            parser.error(f"not a Hugging Face repo id (org/name): {name!r}")
    if getattr(args, "revision", None) and not HF_REV.match(args.revision):
        parser.error(f"not a Hugging Face revision: {args.revision!r}")

    if args.cmd == "sweep" and args.action == "plan":
        plan = sweep_plan(
            args.sha,
            Path(args.results_root),
            args.refresh,
            args.refresh_quetzal,
            args.vllm_version,
            args.prune,
            args.transformers_version,
        )
        Path(args.plan).write_text(json.dumps(plan, indent=2) + "\n")
        for t in plan["targets"]:
            state = "done" if t["done"] else f"[{t['group']}]"
            stack = t.get("stack") or {}
            print(f"[gpuref]   {state:12} {t['model']} {t['task']}: {t['reason']}"
                  f" (spec {t['spec']}, need ~{t['need_gib']} GiB, vLLM {stack.get('vllm')}"
                  f"{' / transformers ' + stack['transformers'] if stack.get('transformers') else ''})",
                  file=sys.stderr)  # fmt: skip
        for t in plan["skipped"]:
            print(
                f"[gpuref]   skipped      {t['model']} {t['task']}: {t['skip']}",
                file=sys.stderr,
            )
        if plan.get("legacy_runs"):
            print(f"[gpuref]   NOTE: {len(plan['legacy_runs'])} model(s) have ok runs recorded "
                  "without an eval fingerprint (older tooling); they count as not done and "
                  f"will be re-run: {', '.join(plan['legacy_runs'])}", file=sys.stderr)  # fmt: skip
        yaml_path = REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml"
        if plan["specs_stale"] and args.write_specs:
            yaml_path.write_text(
                write_derived_specs(yaml_path.read_text(), plan["derived"], args.prune)
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
    if args.cmd == "sweep" or args.cmd == "publish":  # sweep report == publish --plan
        plan = load_json(Path(args.plan)) if args.plan else None
        if args.cmd == "publish" and not (plan or args.models):
            parser.error("publish needs --plan (a sweep) or --models (explicit runs)")
        out = Path(args.out_dir or (Path(args.plan).parent if args.plan else "."))

        def render(ref_url):
            return render_publication(
                out, Path(args.results_root), plan, getattr(args, "models", None),
                Path(args.tt_reports) if args.tt_reports else None,
                Path(args.eval_config) if getattr(args, "eval_config", None) else None,
                ref_url, getattr(args, "note", None),
            )  # fmt: skip

        result = render(args.ref_url)
        print(result["markdown"])
        print(f"{result['finished']}/{len(result['rows'])} targets measured; outputs in {out}"
              + ("" if getattr(args, "post", False) else " (not posted; --post posts after approval)"),
              file=sys.stderr)  # fmt: skip
        if getattr(args, "post", False):
            url = post_publication(out, args.issue, render)
            print(f"[gpuref] posted {url}; {out / 'eval_config.patch'} now cites it",
                  file=sys.stderr)  # fmt: skip
        return 0
    if args.cmd == "driver-status":
        fields = dict(f.split("=", 1) for f in args.fields if "=" in f)
        driver_status(Path(args.file), Path(args.log) if args.log else None, fields)
        return 0
    if args.cmd == "preflight":
        usable, ok, msgs = preflight(check_models(args.models), args.gpus.split(","))
        print("\n".join(f"[gpuref]   {m}" for m in msgs), file=sys.stderr)
        print(" ".join(usable))
        return 0 if ok else 1
    if args.cmd == "status-view":
        state, lines, done, phase = parse_status(sys.stdin.read())
        print("\n".join(lines), file=sys.stderr)
        print(f"{state} {done} {phase}".rstrip() if state else "")
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
    if args.cmd == "materialize-hf-config":
        plan = load_json(Path(args.plan)) or {}
        spec = plan.get("hf_config")
        if not spec:
            return 0
        rev = spec.get("revision") or "main"
        config = hf_get(f"{HF}/{plan['model']}/resolve/{rev}/config.json", hf_token())
        if config is None:
            return 1
        for key in spec["drop"]:
            config.pop(key, None)
        Path(spec["path"]).mkdir(parents=True, exist_ok=True)
        (Path(spec["path"]) / "config.json").write_text(json.dumps(config, indent=2))
        return 0
    if args.cmd == "cap-bound":
        return 0 if cap_bound(load_json(Path(args.evidence)), int(args.cap)) else 1
    if args.cmd == "serve-plan":
        plan = serve_plan(load_gpu_spec(args.model, Path(args.ttis_dir)))
        rev, token = plan.get("revision") or "main", hf_token()
        meta = hf_get(f"{HF}/api/models/{args.model}/revision/{rev}", token)
        config = hf_get(f"{HF}/{args.model}/resolve/{rev}/config.json", token)
        params = model_params(meta, config)
        estimate = (
            memory_estimate(
                params, config, plan["max_context"], plan["max_concurrency"]
            )
            if params
            else None
        )
        mib = run_out(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        )
        gpu_mem = int(mib.splitlines()[0]) / 1024 if mib else None
        plan = gpu_concurrency(plan, params, estimate, gpu_mem)
        plan["stack"] = serving_stack(
            config, args.vllm_version, args.transformers_version
        )
        if estimate:
            verdicts = gpu_verdicts(estimate, plan["max_context"])
            plan["placement"] = gpu_placement(
                verdicts, params, estimate, plan["max_concurrency"]
            )[1]
        print(json.dumps(plan, indent=2))
    elif args.cmd == "ensure-gpu-spec":
        try:
            d = ensure_gpu_spec(args.model, Path(args.ttis_dir))
        except ValueError as exc:
            print(f"[gpuref] {exc}", file=sys.stderr)
            return 1
        if d is not None:
            Path(args.out).write_text(json.dumps(d, indent=2) + "\n")
            print(
                f"[gpuref] derived a GPU spec for {args.model} from its TT row",
                file=sys.stderr,
            )
    elif args.cmd == "task-selection":
        try:
            print(
                task_selection(args.model, args.tasks.split(","), Path(args.ttis_dir))
            )
        except ValueError as exc:
            print(f"[gpuref] {exc}", file=sys.stderr)
            return 1
    elif args.cmd == "plan-stack":
        stack = (load_json(Path(args.plan)) or {}).get("stack") or DEFAULT_STACK
        if args.env:
            print(
                "\n".join(
                    f"{k}={v}" for k, v in sorted((stack.get("env") or {}).items())
                )
            )
        else:
            print(f"{stack['vllm']} {stack.get('transformers') or ''}".rstrip())
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
    elif args.cmd == "task-kind":
        try:
            print(task_kind(args.model, args.tasks.split(","), Path(args.ttis_dir)))
        except ValueError as exc:
            print(f"[gpuref] {exc}", file=sys.stderr)
            return 1
    elif args.cmd == "write-gpu-specs":
        _, rows = catalog_facts()
        derived = {}
        for model in args.models:
            d = derive_entry(model, rows.get(model, []), hf_token())
            if d is None:
                print(
                    f"[gpuref] {model}: no TT row to derive a GPU spec from",
                    file=sys.stderr,
                )
                return 1
            derived[model] = d
        path = REPO_ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml"
        path.write_text(write_derived_specs(path.read_text(), derived))
        print(f"[gpuref] wrote {len(derived)} derived GPU spec(s) to {path}; review and commit",
              file=sys.stderr)  # fmt: skip
    elif args.cmd == "shield-inputs":
        url = Path(args.url_file).read_text().strip()
        try:
            print(
                json.dumps(
                    shield_inputs(
                        args.model, args.task, url, args.ttis_sha, args.secret
                    )
                )
            )
        except ValueError as exc:
            print(f"[gpuref] {exc}", file=sys.stderr)
            return 1
    elif args.cmd == "shield-find-run":
        url = Path(args.url_file).read_text().strip()
        found = shield_find_run(json.load(sys.stdin), args.model, url, args.since)
        if found is None:
            return 1
        print(found)
    elif (
        args.cmd == "redact-url"
    ):  # "<redacted> <sha12>": log the first, compare the second
        url = Path(args.url_file).read_text().strip()
        print(redact_url(url) or "", hashlib.sha256(url.encode()).hexdigest()[:12])
    elif args.cmd == "shield-workflow":
        print(SHIELD_REPO, SHIELD_WORKFLOW)
    elif args.cmd == "shield-provenance":
        prov = load_json(Path(args.prov)) or {}
        records = [load_json(Path(r)) or {} for r in args.records]
        out = shield_provenance(prov, records, Path(args.shield_dir))
        out["serving_rollovers"] = args.rollovers  # sessions replaced mid-run
        Path(args.prov).write_text(json.dumps(out, indent=2) + "\n")
        print(out["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
