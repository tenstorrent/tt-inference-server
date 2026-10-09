#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""All non-`colab` logic of the Colab GPU-reference workflow (see README.md).

Driver side (local):
  preflight --gpus G[,G] MODEL...  GPU spec, HF access and memory-fit checks;
                                   prints the usable GPU types
  snippet NAME [...]               Python code sent to the VM with `colab exec`
  status-view                      parse the status snippet's output (stdin)
  balance-below CU                 exit 0 if `colab usage` (stdin) is below CU
  summary DIR                      per-model, per-task score table
Runner side (on the VM):
  serve-plan --model M             the exact `vllm serve` argv for M's GPU spec
  plan-argv PLAN                   that argv, one item per line
  served MODEL                     exit 0 if /v1/models JSON (stdin) lists MODEL
  record-status FILE MODEL STATUS [NOTE]
  provenance ...                   write provenance.json for one model

The HF token is only ever read from $HF_TOKEN or the token file; it is never
taken as an argument and never printed.
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
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_API = "https://huggingface.co/api/models"
HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"

# vllm serve flags the runner sets itself from the spec (max_context,
# max_concurrency) or fixes for every model (bf16, the HF repo id as both the
# model and its served name, which is what run.py's eval client requests).
EXPLICIT_SERVE_KEYS = frozenset(
    {"model", "served_model_name", "max_model_len", "max_num_seqs", "dtype"}
)
# Defaults DeviceModelSpec injects for Tenstorrent serving that have no
# meaning (or a different one) on a CUDA vLLM: the TT override_tt_config
# payload, the TT paged-KV block granularity, and a log-truncation knob. None
# of them changes what the model computes.
TT_ONLY_SERVE_KEYS = frozenset({"additional_config", "block_size", "max-log-len"})


# --------------------------------------------------------------------------
# Spec reading
# --------------------------------------------------------------------------


def load_gpu_spec(model: str, ttis_dir: Optional[Path] = None):
    """Resolve ``model``'s GPU ModelSpec with TTIS's own loader (dev catalog).

    Mirrors ``run.py --dev-mode --tt-device gpu --model <model>``: the dev
    catalog is selected through MODEL_SPECS_ENV before workflows.model_spec is
    imported, because MODEL_SPECS is built at import time.
    """
    root = Path(ttis_dir or REPO_ROOT).resolve()
    os.environ["MODEL_SPECS_ENV"] = "dev"
    loaded = sys.modules.get("workflows.model_spec")
    if loaded is not None:
        # MODEL_SPECS is fixed at import time; a prod-catalog import (or one
        # from another checkout) cannot be switched, so refuse rather than
        # silently resolve against the wrong catalog.
        if (
            getattr(loaded, "_MODEL_SPECS_ENV", None) != "dev"
            or root not in Path(loaded.__file__).resolve().parents
        ):
            raise RuntimeError(
                "workflows.model_spec is already imported with another catalog; "
                "run gpuref.py in a fresh interpreter"
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from workflows.model_spec import get_runtime_model_spec

    spec, _impl, _engine = get_runtime_model_spec(model=model, device="gpu")
    return spec


def _flag_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def vllm_flags(vllm_args: Dict[str, Any]) -> List[str]:
    """Convert a spec's vllm_args mapping into `vllm serve` CLI flags.

    Keys the runner sets explicitly and TT-only keys are skipped. Booleans
    follow vLLM's BooleanOptionalAction convention (--x / --no-x).
    """
    flags: List[str] = []
    for key, value in vllm_args.items():
        if key in EXPLICIT_SERVE_KEYS or key in TT_ONLY_SERVE_KEYS:
            continue
        if value is None:
            continue
        name = key.replace("_", "-")
        if isinstance(value, bool):
            flags.append(f"--{name}" if value else f"--no-{name}")
            continue
        flags.extend([f"--{name}", _flag_value(value)])
    return flags


def serve_plan(spec, port: int = 8000) -> Dict[str, Any]:
    """Everything the runner needs to serve ``spec`` on a GPU, as JSON data."""
    dms = spec.device_model_spec
    repo = spec.hf_model_repo
    argv = [
        "serve",
        repo,
        "--served-model-name",
        repo,
        "--port",
        str(port),
        "--max-model-len",
        str(dms.max_context),
        "--max-num-seqs",
        str(dms.max_concurrency),
        "--dtype",
        "bfloat16",
    ] + vllm_flags(dms.vllm_args)
    return {
        "model": repo,
        "model_id": spec.model_id,
        "impl": spec.impl.impl_name,
        "device": spec.device_type.to_string(),
        "max_context": dms.max_context,
        "max_concurrency": dms.max_concurrency,
        "revision": dms.vllm_args.get("revision"),
        "vllm_serve_args": argv,
        "dropped_tt_only_vllm_args": {
            k: v for k, v in dms.vllm_args.items() if k in TT_ONLY_SERVE_KEYS
        },
    }


# --------------------------------------------------------------------------
# Hugging Face access (token never printed)
# --------------------------------------------------------------------------


def _read_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        token = HF_TOKEN_FILE.read_text().strip()
    except OSError:
        return None
    return token or None


def _hf_get(url: str, token: Optional[str] = None, timeout: float = 30.0):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def hf_gated(repo: str) -> Optional[bool]:
    """True/False from the public HF model API, None when it cannot be read."""
    try:
        info = _hf_get(f"{HF_API}/{repo}")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return bool(info.get("gated"))


def hf_can_read(repo: str, revision: Optional[str], token: Optional[str]) -> bool:
    """Whether ``token`` can read ``repo`` at ``revision`` (HTTP 200)."""
    url = f"{HF_API}/{repo}/revision/{revision or 'main'}"
    try:
        _hf_get(url, token=token)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return True


def hf_revision_sha(repo: str, revision: Optional[str]) -> Optional[str]:
    """The commit sha ``revision`` (default main) resolves to, or None."""
    url = f"{HF_API}/{repo}/revision/{revision or 'main'}"
    try:
        return _hf_get(url, token=_read_hf_token()).get("sha")
    except (urllib.error.URLError, OSError, ValueError):
        return None


# Colab GPU memory (GiB) for the fit check; G4 is left out (not characterised).
GPU_MEMORY_GIB = {"H100": 80.0, "A100": 40.0, "L4": 22.5, "T4": 15.0}
# vLLM's default --gpu-memory-utilization, and a margin for CUDA context,
# activations and CUDA graphs on top of weights + KV cache.
GPU_MEMORY_UTILIZATION = 0.9
OVERHEAD_GIB = 4.0
GIB = float(1 << 30)


def kv_bytes_per_token(config: Dict[str, Any], dtype_bytes: int = 2) -> int:
    """bf16 K+V bytes per token for a standard decoder config.json."""
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def memory_estimate(
    num_params: int, config: Dict[str, Any], max_context: int, max_concurrency: int
) -> Dict[str, float]:
    """bf16 serving memory: weights, KV for one max_context sequence, KV for all.

    vLLM refuses to start only when one max_context sequence does not fit; the
    full max_concurrency x max_context pool is a throughput concern (vLLM
    queues/preempts), so `need_gib` uses the one-sequence figure.
    """
    weights = num_params * 2 / GIB
    per_token = kv_bytes_per_token(config)
    one_seq = per_token * max_context / GIB
    all_seqs = one_seq * max_concurrency
    return {
        "weights_gib": round(weights, 2),
        "kv_gib_one_seq": round(one_seq, 2),
        "kv_gib_all_seqs": round(all_seqs, 2),
        "need_gib": round(weights + one_seq + OVERHEAD_GIB, 2),
        "kv_bytes_per_token": per_token,
    }


def gpu_verdicts(estimate: Dict[str, Any], max_context: int) -> Dict[str, Any]:
    """Per Colab GPU: "fits", {"cap": tokens} or "no".

    "fits": bf16 weights + one max_context sequence + overhead fit.
    {"cap": n}: the weights fit but max_context does not; about n tokens of
    KV would (the runner then serves at vLLM's own estimate and records it).
    "no": the bf16 weights plus overhead do not fit at all. Quantized
    references are out of scope, so such a model cannot run on that GPU.
    """
    verdicts: Dict[str, Any] = {}
    for gpu, mem in GPU_MEMORY_GIB.items():
        usable = mem * GPU_MEMORY_UTILIZATION
        if estimate["need_gib"] <= usable:
            verdicts[gpu] = "fits"
            continue
        spare = usable - estimate["weights_gib"] - OVERHEAD_GIB
        per_token = estimate.get("kv_bytes_per_token")
        if not per_token:  # KV unknown: only the weights were checked
            verdicts[gpu] = "no"
            continue
        tokens = int(spare * GIB // per_token) if spare > 0 else 0
        verdicts[gpu] = {"cap": min(tokens, max_context)} if tokens >= 2048 else "no"
    return verdicts


def params_from_config(config: Dict[str, Any]) -> int:
    """Approximate parameter count of a dense decoder from config.json.

    Fallback when the Hub has no safetensors metadata. Embeddings (+ untied
    LM head) and, per layer, q/k/v/o projections and a gated MLP; norms and
    biases are negligible. Not valid for MoE configs.
    """
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or hidden // heads)
    inter = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    layers = int(config["num_hidden_layers"])
    embed = vocab * hidden * (1 if config.get("tie_word_embeddings") else 2)
    attn = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    return embed + layers * (attn + 3 * hidden * inter)


def hf_memory_estimate(
    repo: str,
    revision: Optional[str],
    max_context: int,
    max_concurrency: int,
    token: Optional[str],
) -> Optional[Dict[str, Any]]:
    """memory_estimate from the Hub, or None when not even the size is known.

    The parameter count comes from the Hub's safetensors metadata (public even
    for gated repos), else from config.json. Without a readable config.json
    (e.g. a gated repo the token cannot read) the KV term is unknown: the
    estimate then covers the weights only and sets kv_unknown.
    """
    base = f"{revision or 'main'}"
    errors = (urllib.error.URLError, OSError, ValueError, KeyError, TypeError)
    num_params = config = None
    try:
        info = _hf_get(f"{HF_API}/{repo}/revision/{base}", token=token)
        num_params = int((info.get("safetensors") or {})["total"])
    except errors:
        pass
    try:
        config = _hf_get(
            f"https://huggingface.co/{repo}/resolve/{base}/config.json", token=token
        )
        if num_params is None:
            num_params = params_from_config(config)
    except errors:
        config = None
    if num_params is None:
        return None
    if config is not None:
        try:
            estimate = memory_estimate(num_params, config, max_context, max_concurrency)
            estimate["kv_unknown"] = False
        except errors:
            config = None
    if config is None:
        weights = round(num_params * 2 / GIB, 2)
        estimate = {
            "weights_gib": weights,
            "kv_gib_one_seq": None,
            "kv_gib_all_seqs": None,
            "need_gib": round(weights + OVERHEAD_GIB, 2),
            "kv_bytes_per_token": None,
            "kv_unknown": True,
        }
    estimate["num_params"] = num_params
    estimate["verdicts"] = gpu_verdicts(estimate, max_context)
    return estimate


def check_models(models: Iterable[str]) -> List[Dict[str, Any]]:
    token = _read_hf_token()
    rows = []
    for model in models:
        row: Dict[str, Any] = {"model": model}
        try:
            spec = load_gpu_spec(model)
        except Exception as exc:  # resolver raises ValueError; report any failure
            row.update(ok=False, error=f"no GPU spec: {exc}")
            rows.append(row)
            continue
        plan = serve_plan(spec)
        gated = hf_gated(model)
        row.update(
            ok=True,
            model_id=plan["model_id"],
            max_context=plan["max_context"],
            max_concurrency=plan["max_concurrency"],
            revision=plan["revision"],
            gated=gated,
            # Unknown gating is treated as gated: better to demand a token.
            needs_token=gated is not False,
            have_token=token is not None,
        )
        if token is not None and gated is not False:
            row["token_can_read"] = hf_can_read(model, plan["revision"], token)
        row["memory"] = hf_memory_estimate(
            model,
            plan["revision"],
            plan["max_context"],
            plan["max_concurrency"],
            token if gated is not False else None,
        )
        rows.append(row)
    return rows


def preflight(rows: List[Dict[str, Any]], gpus: List[str]):
    """Apply the run policy to check_models() rows.

    Returns (usable_gpus, ok, messages). A model must resolve, have HF access,
    and fit in bf16: per listed GPU it "fits" (weights + one max_context
    sequence), needs a max_model_len cap (weights fit, max_context does not),
    or does not fit. It may use the GPUs that fit fully, else a capped one; with
    neither it is refused (quantized references are out of scope, and one
    Colab runtime has one GPU, so ~70B models cannot run). The run keeps the
    GPU types every model accepts.
    """
    usable, ok, msgs = list(gpus), True, []
    for r in rows:
        model = r["model"]
        if not r.get("ok"):
            msgs += [
                f"{model}: {r.get('error')}",
                "add a `- device: GPU` entry with default_impl: true to its dev spec",
            ]
            ok = False
            continue
        gated = {True: "gated", False: "public", None: "gating unknown"}[r["gated"]]
        msgs.append(
            f"{model}: {r['model_id']} max_context={r['max_context']} "
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
        if mem.get("kv_unknown"):
            msgs.append(
                f"  bf16 weights {mem['weights_gib']} GiB (config.json unreadable)"
            )
        else:
            msgs.append(
                f"  bf16 weights {mem['weights_gib']} GiB + KV {mem['kv_gib_one_seq']} GiB/seq "
                f"-> needs ~{mem['need_gib']} GiB"
            )
        verdicts = {g: mem["verdicts"].get(g) for g in gpus}
        full = {g for g, v in verdicts.items() if v == "fits"}
        capped = {g for g, v in verdicts.items() if isinstance(v, dict)}
        unknown = {g for g, v in verdicts.items() if v is None}
        for g, v in verdicts.items():
            if isinstance(v, dict):
                msgs.append(
                    f"  {g}: max_context does not fit; ~{v['cap']} tokens would (cap recorded)"
                )
            elif v == "no":
                msgs.append(f"  {g}: bf16 weights do not fit")
        if not (full or capped) and not unknown:
            hint = (
                "add H100 (80 GB) to --gpu"
                if "H100" not in gpus
                else "models above ~32B (e.g. 70B) need a multi-GPU machine, out of scope on Colab"
            )
            msgs.append(f"  refused: bf16 weights do not fit any GPU in --gpu; {hint}")
            ok = False
            continue
        allowed = (full or capped) | unknown
        dropped = [g for g in usable if g not in allowed]
        if dropped:
            msgs.append(f"  dropping {', '.join(dropped)} from --gpu for this run")
        usable = [g for g in usable if g in allowed]
    if ok and not usable:
        msgs.append("no GPU in --gpu fits every model; split them into separate runs")
        ok = False
    return usable, ok, msgs


# --------------------------------------------------------------------------
# VM snippets (sent with `colab exec`; none of them handles the token's value)
# --------------------------------------------------------------------------

REMOTE_DIR = "/content/gpuref"

_SNIPPETS = {
    "prepare": """
import os
os.makedirs(W, exist_ok=True)
hf = os.path.expanduser("~/.cache/huggingface")
os.makedirs(hf, mode=0o700, exist_ok=True)
os.chmod(hf, 0o700)
print("GPUREF_PREPARED")
""",
    # The upload lands in the non-hidden work dir (Jupyter's contents API may
    # refuse hidden paths such as ~/.cache); this moves it into place.
    "install_token": """
import os
src, dst = os.path.join(W, "hf_token.upload"), os.path.expanduser("~/.cache/huggingface/token")
os.chmod(src, 0o600)
os.replace(src, dst)
os.chmod(dst, 0o600)
print("GPUREF_TOKEN_INSTALLED mode=%o bytes=%d" % (os.stat(dst).st_mode & 0o777, os.stat(dst).st_size))
""",
    # start_new_session=True is setsid(): no controlling terminal, its own
    # process group, so the runner outlives this kernel call (nohup-like).
    "launch": """
import os, subprocess
for name in ("DONE", "FAILED", "phase"):
    try:
        os.remove(os.path.join(W, name))
    except OSError:
        pass
log = open(os.path.join(W, "runner.log"), "ab")
proc = subprocess.Popen(["bash", os.path.join(W, "remote_runner.sh")] + ARGS,
                        cwd=W, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True)
with open(os.path.join(W, "runner.pid"), "w") as f:
    f.write(str(proc.pid))
print("GPUREF_LAUNCHED pid=%d" % proc.pid)
""",
    "status": """
import glob, os, subprocess
def read(name):
    try:
        with open(os.path.join(W, name)) as f:
            return f.read().strip()
    except OSError:
        return ""
def alive(pid):
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False
pid = read("runner.pid")
for state, cond in (("DONE", os.path.exists(os.path.join(W, "DONE"))),
                    ("FAILED", os.path.exists(os.path.join(W, "FAILED"))),
                    ("RUNNING", bool(pid) and alive(pid)), ("DIED", bool(pid)), ("ABSENT", True)):
    if cond:
        break
print("GPUREF_STATE=" + state)
print("GPUREF_PHASE=" + read("phase"))
try:
    print("GPUREF_GPU=" + subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip())
except Exception:
    pass
for line in read("runner.log").splitlines()[-12:]:
    print("GPUREF_LOG| " + line)
logs = sorted(glob.glob(os.path.join(W, "results", "*", "run_py.log")), key=os.path.getmtime)
if logs:
    with open(logs[-1], "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4096))
        tail = f.read().decode("utf-8", "replace").replace("\\r", "\\n").splitlines()
    if tail:
        print("GPUREF_EVAL=" + os.path.basename(os.path.dirname(logs[-1])) + ": " + tail[-1][-200:])
""",
    # A run stopped early (--min-balance) has not copied TTIS's workflow_logs.
    "pack": """
import os, subprocess
tarball, paths = os.path.join(W, "gpuref-results.tar.gz"), ["results"]
if not os.path.isdir(os.path.join(W, "results", "workflow_logs")) and os.path.isdir(
        os.path.join(W, "tt-inference-server", "workflow_logs")):
    paths.append("tt-inference-server/workflow_logs")
subprocess.run(["tar", "-czf", tarball, "-C", W] + paths, check=True)
print("GPUREF_PACKED bytes=%d" % os.path.getsize(tarball))
""",
}


def render_snippet(name: str, runner_args: Optional[List[str]] = None) -> str:
    """Python source for one VM step. Only `launch` takes arguments (the
    runner's argv: sha, vLLM pin, models), embedded with repr()."""
    head = f"W = {REMOTE_DIR!r}\n"
    if name == "launch":
        head += f"ARGS = {list(runner_args or [])!r}\n"
    return head + _SNIPPETS[name].lstrip("\n")


def parse_status(text: str):
    """(state, display_lines) from the status snippet's output."""
    state, lines = None, []
    labels = {
        "GPUREF_PHASE=": "  phase: ",
        "GPUREF_GPU=": "  gpu:   ",
        "GPUREF_EVAL=": "  evals: ",
    }
    for line in text.splitlines():
        if line.startswith("GPUREF_STATE="):
            state = line.split("=", 1)[1].strip() or None
        elif line.startswith("GPUREF_LOG| "):
            lines.append("  | " + line[len("GPUREF_LOG| ") :])
        else:
            for prefix, label in labels.items():
                if line.startswith(prefix):
                    lines.append(label + line[len(prefix) :])
    return state, lines


def parse_balance(text: str) -> Optional[float]:
    """Compute-unit balance from `colab usage` output, or None."""
    match = re.search(r"Current balance:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def record_status(path: Path, model: str, status: str, note: Optional[str]) -> None:
    data = _load_json(path) if path.exists() else None
    data = data if isinstance(data, dict) else {}
    data[model] = {"status": status, "note": note or None}
    path.write_text(json.dumps(data, indent=2) + "\n")


def served_models(payload: Any) -> List[str]:
    """Model ids in an OpenAI /v1/models response."""
    data = payload.get("data") if isinstance(payload, dict) else None
    return [m.get("id") for m in data or [] if isinstance(m, dict)]


# --------------------------------------------------------------------------
# Request-length evidence (for runs served under a max_model_len cap)
# --------------------------------------------------------------------------

# Markers of a request the server rejected or the harness cut short.
REJECTION_PATTERN = re.compile(
    r"maximum context length|truncat|HTTP/1\.1\" 400|400 Bad Request", re.IGNORECASE
)


def sample_task_name(path: Path) -> str:
    """'samples_mmlu_pro_law_2026-10-09T00-14-03.525271.jsonl' -> 'mmlu_pro_law'."""
    return re.sub(r"_\d{4}-\d{2}-\d{2}T[\d.-]+$", "", path.stem[len("samples_") :])


def request_token_stats(root: Path, count_tokens) -> Dict[str, Dict[str, int]]:
    """Per lm-eval task: samples, longest prompt+generation in tokens, and
    empty responses, from every samples_*.jsonl under root.

    ``count_tokens(text) -> int`` is the served model's tokenizer.
    """
    stats: Dict[str, Dict[str, int]] = {}
    for path in sorted(root.rglob("samples_*.jsonl")):
        entry = stats.setdefault(
            sample_task_name(path), {"samples": 0, "max_tokens": 0, "empty": 0}
        )
        with path.open() as f:
            for line in f:
                sample = json.loads(line)
                args = sample.get("arguments") or {}
                first = args.get("gen_args_0") if isinstance(args, dict) else None
                prompt = (
                    (first or {}).get("arg_0")
                    if first
                    else (args[0][0] if args else "")
                )
                resp = (sample.get("resps") or [[""]])[0]
                resp = resp[0] if isinstance(resp, list) else resp
                total = count_tokens(str(prompt or "")) + count_tokens(str(resp or ""))
                entry["samples"] += 1
                entry["max_tokens"] = max(entry["max_tokens"], total)
                entry["empty"] += 0 if str(resp or "").strip() else 1
    return stats


def cap_bound(evidence: Optional[Dict[str, Any]], cap: int) -> bool:
    """True when a capped run cannot be trusted: no evidence, a rejected or
    truncated request in the logs, or a request at the cap (prompt+generation
    within 64 tokens of it, the harness's context reserve)."""
    if not isinstance(evidence, dict):
        return True
    if evidence.get("rejection_log_lines"):
        return True
    lengths = [
        t.get("max_tokens", 0)
        for t in (evidence.get("max_request_tokens") or {}).values()
    ]
    return not lengths or max(lengths) >= cap - 64


def rejection_lines(paths: Iterable[Path]) -> List[str]:
    """Log lines that look like a rejected or truncated request."""
    hits = []
    for path in paths:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        hits += [
            f"{path.name}: {ln.strip()[:200]}"
            for ln in text.splitlines()
            if REJECTION_PATTERN.search(ln)
        ]
    return hits


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def lm_eval_commit_pinned(ttis_dir: Path) -> Optional[str]:
    """The lm-evaluation-harness commit TTIS pins for the evals-common venv."""
    req = ttis_dir / "requirements" / "evals-common.txt"
    try:
        text = req.read_text()
    except OSError:
        return None
    match = re.search(r"lm-evaluation-harness\.git@([0-9a-f]{7,40})", text)
    return match.group(1) if match else None


def lm_eval_commit_installed(ttis_dir: Path) -> Optional[str]:
    """The commit recorded by pip in an installed lm_eval's direct_url.json."""
    # pathlib's glob, unlike glob.glob, also matches the hidden .venv_* dirs.
    venvs = ttis_dir / ".workflow_venvs"
    pattern = "*/lib/python*/site-packages/lm_eval-*.dist-info/direct_url.json"
    for path in sorted(venvs.glob(pattern)):
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        commit = (info.get("vcs_info") or {}).get("commit_id")
        if commit:
            return commit
    return None


def gpu_info() -> Dict[str, Optional[str]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if not out:
        return {"name": None, "driver": None, "memory_total": None}
    first = [part.strip() for part in out.splitlines()[0].split(",")]
    first += [None] * (3 - len(first))
    return {"name": first[0], "driver": first[1], "memory_total": first[2]}


def package_versions(python: str) -> Dict[str, Optional[str]]:
    code = (
        "import importlib.metadata as m, json\n"
        "out = {}\n"
        "for p in ('vllm', 'torch', 'transformers'):\n"
        "    try:\n"
        "        out[p] = m.version(p)\n"
        "    except m.PackageNotFoundError:\n"
        "        out[p] = None\n"
        "print(json.dumps(out))\n"
    )
    out = _run([python, "-c", code])
    try:
        return json.loads(out) if out else {}
    except ValueError:
        return {}


def build_provenance(args: argparse.Namespace) -> Dict[str, Any]:
    ttis_dir = Path(args.ttis_dir)
    plan = json.loads(Path(args.plan).read_text()) if args.plan else {}
    pinned = plan.get("revision")
    return {
        "model": args.model,
        "model_id": plan.get("model_id"),
        "status": args.status,
        "evals_exit_code": args.evals_rc,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "gpu": gpu_info(),
        "packages": package_versions(args.vllm_python),
        "ttis_sha": _run(["git", "rev-parse", "HEAD"], cwd=ttis_dir),
        "model_revision_pinned": pinned,
        "model_revision_sha": hf_revision_sha(args.model, pinned),
        "max_context": plan.get("max_context"),
        "max_concurrency": plan.get("max_concurrency"),
        "vllm_serve_command": [args.vllm_bin] + plan.get("vllm_serve_args", []),
        # Set only when vLLM refused max_context on this GPU and the runner
        # retried with vLLM's own estimate; lm-eval still uses max_context.
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
        ],
        "lm_eval_commit_pinned": lm_eval_commit_pinned(ttis_dir),
        "lm_eval_commit_installed": lm_eval_commit_installed(ttis_dir),
        "note": args.note,
        # Longest prompt+generation per task and any rejection/truncation log
        # lines: the evidence that a max_model_len cap did not bind.
        "request_evidence": _load_json(Path(args.evidence)) if args.evidence else None,
    }


# --------------------------------------------------------------------------
# Score summary
# --------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def find_eval_reports(root: Path) -> List[Dict[str, Any]]:
    """TTIS report payloads (metadata + sections) with eval blocks under root.

    Matches both the local layout (reports_output/evals/data/report_data_*.json)
    and Shield's report_evals_*/report_*.json. Only the newest report per model
    is kept, so a re-run supersedes an older one.
    """
    newest: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("report*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        sections = payload.get("sections")
        metadata = payload.get("metadata") or {}
        if not isinstance(sections, list):
            continue
        if not any(isinstance(s, dict) and s.get("kind") == "evals" for s in sections):
            continue
        model = metadata.get("model_repo") or metadata.get("model_name") or str(path)
        stamp = metadata.get("generated_at") or ""
        if model not in newest or stamp >= newest[model]["metadata"].get(
            "generated_at", ""
        ):
            payload = dict(payload)
            payload["_path"] = str(path)
            newest[model] = payload
    return [newest[k] for k in sorted(newest)]


def gpu_names_by_model(root: Path) -> Dict[str, str]:
    """model -> GPU name recorded in each provenance.json under root."""
    names = {}
    for path in sorted(root.rglob("provenance.json")):
        payload = _load_json(path)
        if isinstance(payload, dict) and payload.get("model"):
            name = (payload.get("gpu") or {}).get("name")
            if name:
                names[payload["model"]] = name
    return names


def summary_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    gpus = gpu_names_by_model(root)
    for report in find_eval_reports(root):
        meta = report.get("metadata") or {}
        model = meta.get("model_repo") or meta.get("model_name")
        for section in report["sections"]:
            if not isinstance(section, dict) or section.get("kind") != "evals":
                continue
            data = section.get("data") or {}
            rows.append(
                {
                    "model": model,
                    "device": meta.get("device"),
                    "gpu": gpus.get(model),
                    "task": data.get("task_name"),
                    "score": data.get("score"),
                    "published_score": data.get("published_score"),
                    "gpu_reference_score": data.get("gpu_reference_score"),
                    "accuracy_check": data.get("accuracy_check"),
                    "report": report["_path"],
                }
            )
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "-" if value is None else str(value)


def format_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No TTIS eval reports found."
    header = ["model", "device", "gpu", "task", "score", "published", "gpu_ref"]
    table = [header] + [
        [
            _fmt(r["model"]),
            _fmt(r["device"]),
            _fmt(r.get("gpu")),
            _fmt(r["task"]),
            _fmt(r["score"]),
            _fmt(r["published_score"]),
            _fmt(r["gpu_reference_score"]),
        ]
        for r in rows
    ]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in table
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-models")
    p.add_argument("models", nargs="+")
    p = sub.add_parser("preflight")
    p.add_argument("--gpus", required=True, help="comma-separated, in preference order")
    p.add_argument("models", nargs="+")
    p = sub.add_parser("snippet")
    p.add_argument("name", choices=sorted(_SNIPPETS))
    p.add_argument("runner_args", nargs="*", help="launch only: remote_runner.sh argv")
    sub.add_parser("status-view")
    p = sub.add_parser("balance-below")
    p.add_argument("floor", type=float)
    p = sub.add_parser("summary")
    p.add_argument("dir")
    p.add_argument("--json", action="store_true", help="print rows as JSON")

    p = sub.add_parser("serve-plan")
    p.add_argument("--model", required=True)
    p.add_argument("--ttis-dir", default=str(REPO_ROOT))
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("plan-argv")
    p.add_argument("plan")
    p = sub.add_parser("served")
    p.add_argument("model")
    p = sub.add_parser("record-status")
    p.add_argument("file")
    p.add_argument("model")
    p.add_argument("status")
    p.add_argument("note", nargs="?", default=None)
    p = sub.add_parser("request-evidence")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--samples-root", required=True)
    p.add_argument("--log", action="append", default=[], help="log files to scan")
    p.add_argument("--out", required=True, help="JSON file to write")
    p = sub.add_parser("cap-bound")
    p.add_argument("evidence")
    p.add_argument("cap", type=int)
    p = sub.add_parser("provenance")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ttis-dir", required=True)
    p.add_argument("--plan", default=None)
    p.add_argument("--vllm-python", required=True)
    p.add_argument("--vllm-bin", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--evals-rc", type=int, default=None)
    p.add_argument("--started-at", required=True)
    p.add_argument("--finished-at", required=True)
    p.add_argument("--note", default=None)
    p.add_argument("--max-model-len-cap", type=int, default=None)
    p.add_argument("--evidence", default=None, help="request-evidence JSON to embed")

    args = parser.parse_args(argv)

    if args.cmd == "check-models":
        rows = check_models(args.models)
        print(json.dumps(rows, indent=2))
        return 0 if all(r["ok"] for r in rows) else 1
    if args.cmd == "preflight":
        usable, ok, msgs = preflight(check_models(args.models), args.gpus.split(","))
        for msg in msgs:
            print(f"[gpuref]   {msg}", file=sys.stderr)
        print(" ".join(usable))
        return 0 if ok else 1
    if args.cmd == "snippet":
        print(render_snippet(args.name, args.runner_args), end="")
        return 0
    if args.cmd == "status-view":
        state, lines = parse_status(sys.stdin.read())
        for line in lines:
            print(line, file=sys.stderr)
        if state is None:
            return 1
        print(state)
        return 0
    if args.cmd == "balance-below":
        balance = parse_balance(sys.stdin.read())
        print(f"{balance:.2f}" if balance is not None else "unknown")
        return 0 if balance is not None and balance < args.floor else 1
    if args.cmd == "summary":
        root = Path(args.dir)
        status = _load_json(root / "status.json")
        if isinstance(status, dict) and not args.json:
            for model, info in status.items():
                print(
                    f"{model}: {info.get('status')}"
                    + (f" ({info['note']})" if info.get("note") else "")
                )
        rows = summary_rows(root)
        print(json.dumps(rows, indent=2) if args.json else format_summary(rows))
        return 0 if rows else 1
    if args.cmd == "serve-plan":
        spec = load_gpu_spec(args.model, Path(args.ttis_dir))
        print(json.dumps(serve_plan(spec, port=args.port), indent=2))
        return 0
    if args.cmd == "plan-argv":
        print("\n".join(json.loads(Path(args.plan).read_text())["vllm_serve_args"]))
        return 0
    if args.cmd == "served":
        try:
            payload = json.loads(sys.stdin.read())
        except ValueError:
            return 1
        return 0 if args.model in served_models(payload) else 1
    if args.cmd == "record-status":
        record_status(Path(args.file), args.model, args.status, args.note)
        return 0
    if args.cmd == "request-evidence":
        # Needs transformers: run with the vLLM venv's python.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        evidence = {
            "max_request_tokens": request_token_stats(
                Path(args.samples_root),
                lambda t: len(tok(t, add_special_tokens=False).input_ids),
            ),
            "rejection_log_lines": rejection_lines(Path(p) for p in args.log),
        }
        Path(args.out).write_text(json.dumps(evidence, indent=2) + "\n")
        return 0
    if args.cmd == "cap-bound":
        return 0 if cap_bound(_load_json(Path(args.evidence)), args.cap) else 1
    if args.cmd == "provenance":
        Path(args.out).write_text(json.dumps(build_provenance(args), indent=2) + "\n")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
