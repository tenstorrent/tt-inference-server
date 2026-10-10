#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Write the first N GSM8K test questions as 8-shot prompts (the exemplars, format and text
of the dsv41_gsm8k_cot eval task) to a JSONL ({"prompt": ...} per line) for
``TT_BENCH_DATASET_PATH``. Raw content only: the chat endpoint applies the server's template.

Run with the EVALS_COMMON venv python:
  .workflow_venvs/.venv_evals_common/bin/python scripts/build_gsm8k_bench_prompts.py OUT.jsonl [N] [TOKENIZER_DIR]
"""
import json
import sys
from pathlib import Path

from lm_eval.tasks import TaskManager, get_task_dict

out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
tok_dir = sys.argv[3] if len(sys.argv) > 3 else None
include = Path(__file__).resolve().parent.parent / "reference_config/evals/lm_eval_tasks"
task = get_task_dict(["dsv41_gsm8k_cot"], TaskManager(include_path=str(include)))["dsv41_gsm8k_cot"]
task.set_fewshot_seed(1234)
docs = list(task.test_docs())[:n]
prompts = [
    task.fewshot_context(doc, num_fewshot=8, system_instruction=None, apply_chat_template=False)
    for doc in docs
]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for p in prompts:
        f.write(json.dumps({"prompt": p}) + "\n")
print(f"wrote {len(prompts)} prompts to {out}")
if tok_dir:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    lens = [len(tok(p)["input_ids"]) for p in prompts]
    print(f"tokens: min {min(lens)} max {max(lens)} mean {sum(lens) / len(lens):.0f}")
