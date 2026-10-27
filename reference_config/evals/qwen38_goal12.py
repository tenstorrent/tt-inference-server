# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Versioned goal-12 controls, shipped with each immutable CI arm ref."""
import copy
import json
from dataclasses import replace
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B"

def controls():
    c = json.loads(Path(__file__).with_suffix(".json").read_text())
    if type(c.get("record_suite_metrics", False)) is not bool:
        raise ValueError("record_suite_metrics must be a boolean")
    if c.get("swe_reasoning_effort") not in (None, "xhigh", "medium", "low"):
        raise ValueError("swe_reasoning_effort must be xhigh, medium or low")
    return c


def configure(config, limit_mode):
    c = controls()
    tasks = []
    for original in config.tasks:
        name = original.task_name
        # Preserve the requested version; old immutable TB2 runs remain reproducible.
        if name == "terminal_bench_2_1" and name not in c["evals"] and "terminal_bench_2" in c["evals"]:
            name = "terminal_bench_2"
        if name not in c["evals"]:
            continue
        # Scores remain measurements; reference comparison belongs to the report.
        score = replace(original.score, published_score=None, gpu_reference_score=None)
        common = dict(task_name=name, score=score, seed=c["seed"],
                      limit_samples_map={limit_mode.CI_NIGHTLY: c["sample_count"],
                                         limit_mode.SMOKE_TEST: c["sample_count"]})
        if original.agentic_eval_config is None:
            common.update(max_concurrent=c["max_concurrent"], capture_reasoning=True,
                          doc_ids=None if c["full_suite"] else c["gpqa_handoff"]["doc_ids"][:c["sample_count"]],
                          wall_clock_timeout_seconds=c.get("gpqa_suite_timeout_sec", c["suite_timeout_sec"]),
                          model_kwargs={**original.model_kwargs, "timeout": c.get("gpqa_llm_timeout_sec", c["llm_timeout_sec"])},
                          gen_kwargs={**{k: v for k, v in c["gpqa_handoff"]["generation"].items() if k != "chat_template_kwargs"}, "stream": "false", "max_gen_toks": c["output_tokens"]},
                          request_body={"chat_template_kwargs": c["gpqa_handoff"]["generation"]["chat_template_kwargs"]})
        else:
            cfg = original.agentic_eval_config
            kwargs = copy.deepcopy(cfg.agent_kwargs)
            if cfg.agent == "terminus-2":
                kwargs["session_id_headers"] = ["X-Session-ID"]
                kwargs["model_info"]["max_output_tokens"] = c["output_tokens"]
                kwargs["llm_kwargs"].update(seed=c["seed"], max_tokens=c["output_tokens"], timeout=c["llm_timeout_sec"])
            else:
                kwargs["max_tokens"] = c["output_tokens"]
                kwargs["config"]["model"]["model_kwargs"]["seed"] = c["seed"]
                if c.get("swe_reasoning_effort") is not None:
                    model_kwargs = kwargs["config"]["model"]["model_kwargs"]
                    template_kwargs = model_kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
                    template_kwargs["reasoning_effort"] = c["swe_reasoning_effort"]
            names = c["task_names"][name]
            common["agentic_eval_config"] = replace(
                cfg, dataset="terminal-bench/terminal-bench-2" if name == "terminal_bench_2" else cfg.dataset,
                dataset_ref=c["terminal_dataset_ref"] if name in ("terminal_bench_2", "terminal_bench_2_1") else cfg.dataset_ref, n_tasks=None, n_concurrent_trials=c["max_concurrent"],
                agent_timeout_sec=c["agent_timeout_sec"], llm_timeout_sec=c["llm_timeout_sec"],
                harbor_timeout_sec=c["suite_timeout_sec"], agent_kwargs=kwargs,
                task_names_map={limit_mode.CI_NIGHTLY: names, limit_mode.SMOKE_TEST: names},
                task_names=[])
        tasks.append(replace(original, **common))
    return replace(config, tasks=tasks)
