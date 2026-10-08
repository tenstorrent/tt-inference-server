#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Python helpers for the Colab GPU-reference workflow (see README.md here).

Subcommands, all printing JSON or plain text on stdout:

  check-models MODEL...   resolve each model's GPU DeviceModelSpec (dev catalog)
                          and report whether its HF repo is gated and whether
                          the local HF token can read it (driver preflight)
  serve-plan  --model M   the exact `vllm serve` argv for M's GPU spec
                          (runner, on the VM)
  provenance  ...         write provenance.json for one model (runner)
  summary     DIR         per-model, per-task score table parsed from the
                          TTIS eval report JSON under DIR (driver)

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


def gpu_fits(need_gib: float) -> Dict[str, bool]:
    return {
        gpu: need_gib <= mem * GPU_MEMORY_UTILIZATION
        for gpu, mem in GPU_MEMORY_GIB.items()
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
    estimate["fits"] = gpu_fits(estimate["need_gib"])
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

    p_check = sub.add_parser("check-models")
    p_check.add_argument("models", nargs="+")

    p_plan = sub.add_parser("serve-plan")
    p_plan.add_argument("--model", required=True)
    p_plan.add_argument("--ttis-dir", default=str(REPO_ROOT))
    p_plan.add_argument("--port", type=int, default=8000)

    p_prov = sub.add_parser("provenance")
    p_prov.add_argument("--out", required=True)
    p_prov.add_argument("--model", required=True)
    p_prov.add_argument("--ttis-dir", required=True)
    p_prov.add_argument("--plan", default=None)
    p_prov.add_argument("--vllm-python", required=True)
    p_prov.add_argument("--vllm-bin", required=True)
    p_prov.add_argument("--status", required=True)
    p_prov.add_argument("--evals-rc", type=int, default=None)
    p_prov.add_argument("--started-at", required=True)
    p_prov.add_argument("--finished-at", required=True)
    p_prov.add_argument("--note", default=None)
    p_prov.add_argument("--max-model-len-cap", type=int, default=None)

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("dir")
    p_sum.add_argument("--json", action="store_true", help="print rows as JSON")

    args = parser.parse_args(argv)

    if args.cmd == "check-models":
        rows = check_models(args.models)
        print(json.dumps(rows, indent=2))
        return 0 if all(r["ok"] for r in rows) else 1
    if args.cmd == "serve-plan":
        spec = load_gpu_spec(args.model, Path(args.ttis_dir))
        print(json.dumps(serve_plan(spec, port=args.port), indent=2))
        return 0
    if args.cmd == "provenance":
        Path(args.out).write_text(json.dumps(build_provenance(args), indent=2) + "\n")
        return 0
    if args.cmd == "summary":
        rows = summary_rows(Path(args.dir))
        print(json.dumps(rows, indent=2) if args.json else format_summary(rows))
        return 0 if rows else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
