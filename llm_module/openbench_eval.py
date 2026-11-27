# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""One openbench / inspect-ai benchmark against the served model, with lm-eval-shaped
output for the eval workflow.

The two benchmarks OpenRouter's Auto Exacto routes by, with its settings:

- ``gpqa_diamond_openrouter``: GPQA-Diamond, temperature 0.5, 10 epochs, options
  shuffled per question (llm_module/openbench_tasks/gpqa_shuffled.py);
- ``tau2_airline``: openbench's tau_bench_airline (temperature 0, 1 epoch), its user
  simulator on ``--user-model`` (OpenRouter does not name its own).

Run with the EVALS_OPENBENCH venv's python. Writes, under ``<output-dir>/<repo>/``,
``results_<stamp>.json`` (``results`` / ``configs`` / ``n-samples`` as lm-eval
writes them, so the workflow's result discovery and scoring read it unchanged) and
``samples_<task>_<stamp>.jsonl`` (one row per sample and epoch; an errored sample's
answer is ``__INFERENCE_ERROR__: ...``, which grades the run incomplete). The
inspect log stays under ``<output-dir>/inspect_logs``. Exit status 0 only when
inspect finished the run.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INFERENCE_ERROR_MARKER = "__INFERENCE_ERROR__"
DATASETS = {
    "gpqa_diamond_openrouter": "nmayorga7/gpqa_diamond",
    "tau2_airline": "sierra-research/tau2-bench airline",
}
METRIC_NAMES = ("accuracy", "mean")


def build_task(name, epochs, user_model):
    if name == "gpqa_diamond_openrouter":
        sys.path.insert(0, str(HERE / "openbench_tasks"))
        from gpqa_shuffled import gpqa_diamond_shuffled

        return gpqa_diamond_shuffled(epochs=epochs or 10)
    if name == "tau2_airline":
        # tau2 reads its data dir at import time: the venv's staged copy
        # (workflows/workflow_venvs.py setup_evals_openbench).
        os.environ.setdefault("TAU2_DATA_DIR", str(Path(sys.prefix) / "tau2-data"))
        from openbench.evals.tau_bench import tau_bench_airline

        return tau_bench_airline(user_model=user_model)
    raise SystemExit(f"unknown openbench task {name!r}: {sorted(DATASETS)}")


def resolve_limit(limit, n):
    """lm-eval's --limit: a fraction of the dataset below 1, else a count."""
    if limit is None:
        return None
    value = float(limit)
    return max(1, math.ceil(value * n)) if value < 1 else int(value)


def summarize(log):
    """(metrics, sample rows, distinct samples) of an inspect EvalLog."""
    metrics = {}
    for score in (log.results.scores if log.results else []) or []:
        for metric_name, metric in score.metrics.items():
            if metric_name in METRIC_NAMES and "accuracy" not in metrics:
                metrics["accuracy"] = metric.value
    rows, ids = [], set()
    for sample in log.samples or []:
        ids.add(sample.id)
        if sample.error is not None:
            answer = f"{INFERENCE_ERROR_MARKER}: {sample.error.message}"[:2000]
        else:
            answer = sample.output.completion if sample.output else ""
        scores = {k: v.value for k, v in (sample.scores or {}).items()}
        rows.append(
            {
                "doc_id": sample.id,
                "epoch": sample.epoch,
                "target": sample.target,
                "resps": [[answer]],
                "filtered_resps": [[answer]],
                "scores": scores,
            }
        )
    return metrics, rows, len(ids)


def write_outputs(out, task_name, metrics, rows, n, n_dataset, config):
    """``samples_<task>_<stamp>.jsonl`` always; ``results_<stamp>.json`` only with
    metrics (an unfinished run leaves no score: the workflow grades it failed)."""
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    with (out / f"samples_{task_name}_{stamp}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    if not metrics:
        return None
    results = {
        "results": {task_name: {f"{k},none": v for k, v in metrics.items()}},
        "configs": {
            task_name: {"task": task_name, "dataset_path": DATASETS[task_name]}
        },
        "n-samples": {task_name: {"original": n_dataset, "effective": n}},
        "config": config,
    }
    path = out / f"results_{stamp}.json"
    path.write_text(json.dumps(results, indent=1, default=str))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(DATASETS))
    ap.add_argument("--base-url", required=True, help="the server's /v1 URL")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--user-model", default="openrouter/openai/gpt-4.1")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument(
        "--inspect-model",
        default=None,
        help="inspect model string instead of openai-api/tt/<model> (mockllm/model)",
    )
    args = ap.parse_args(argv)

    # inspect's OpenAI-compatible provider: openai-api/<service>/<model> reads
    # <SERVICE>_BASE_URL / <SERVICE>_API_KEY.
    os.environ["TT_BASE_URL"] = args.base_url
    os.environ["TT_API_KEY"] = os.environ.get("OPENAI_API_KEY") or "none"
    if args.task == "tau2_airline" and args.user_model.startswith("openrouter/"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit(
                "tau2_airline: the user simulator "
                f"{args.user_model} needs OPENROUTER_API_KEY"
            )

    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log

    task = build_task(args.task, args.epochs, args.user_model)
    limit = resolve_limit(args.limit, len(task.dataset))
    log_dir = args.output_dir / "inspect_logs"
    (log,) = inspect_eval(
        task,
        model=args.inspect_model or f"openai-api/tt/{args.model}",
        log_dir=str(log_dir),
        max_connections=args.concurrency,
        limit=limit,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=3,
        fail_on_error=False,
        display="plain",
    )
    log = read_eval_log(log.location)
    metrics, rows, n = summarize(log)

    config = {
        "model": args.inspect_model or f"openai-api/tt/{args.model}",
        "epochs": log.eval.config.epochs,
        "limit": limit,
        "user_model": args.user_model if args.task == "tau2_airline" else None,
        "inspect_log": log.location,
    }
    write_outputs(
        args.output_dir / args.hf_repo.replace("/", "__"),
        args.task,
        metrics if log.status == "success" else {},
        rows,
        n,
        len(task.dataset),
        config,
    )
    print(json.dumps({"task": args.task, "status": log.status, "metrics": metrics}))
    return 0 if log.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
