#!/usr/bin/env python3
"""Fill an HF cache with the datasets of the DeepSeek-V4.1-Flash evals, so a client without internet
(or a re-run) loads them offline.

usage (the workflow's EVALS_COMMON venv has `datasets`):
  HF_HOME=/mnt/tt-data/ssinghal/datasets/hf_home .workflow_venvs/.venv_evals_common/bin/python \
      scripts/dsv41_prefetch_eval_datasets.py [group ...]      # groups: text, longbench, vision (default: all)
Then run the evals with that directory as the client's HF cache (run_gsm8k_eval.sh: HF_CACHE=...) and HF_HUB_OFFLINE=1.
Idavidrein/gpqa is gated: it needs an HF token with access (HF_TOKEN or ~/.cache/huggingface/token).
"""
import sys

import datasets

GROUPS = {
    "text": [  # (path, config or None, split, trust_remote_code)
        ("openai/gsm8k", "main", "test"),
        ("TIGER-Lab/MMLU-Pro", None, "test"),
        ("TIGER-Lab/MMLU-Pro", None, "validation"),
        ("openai/openai_humaneval", None, "test"),
        ("Idavidrein/gpqa", "gpqa_diamond", "train"),
        *[("EleutherAI/hendrycks_math", c, "test") for c in (
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")],
        *[("EleutherAI/hendrycks_math", c, "train") for c in (
            "algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")],
        *[("SaylorTwift/bbh", c, "test") for c in (
            "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages",
            "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
            "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
            "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
            "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks",
            "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies",
            "word_sorting")],
    ],
    "longbench": [("recursal/longbench-v2", c, "train") for c in (
        "academic_multi", "academic_single", "agent_history_qa", "code_repo_qa", "detective", "dialogue_history_qa",
        "event_ordering", "financial_multi", "financial_single", "government_multi", "government_single",
        "graph_reasoning", "legal_multi", "legal_single", "literary", "manyshot_learning", "multinews",
        "new_language_translation", "table_qa", "user_guide_qa")],
    "vision": [  # ~1.1 GB; the older chartqa / docvqa / mmmu tasks fetch their own (DocVQA is several GB)
        ("MMMU/MMMU_Pro", "standard (10 options)", "test"),
        ("nyu-visionx/CV-Bench", None, "test"),
    ],
}

if __name__ == "__main__":
    for g in sys.argv[1:] or list(GROUPS):
        for path, cfg, split in GROUPS[g]:
            try:
                d = datasets.load_dataset(path, cfg, split=split, trust_remote_code=True)
                print(f"ok   {path} {cfg or ''} {split}: {len(d)} rows", flush=True)
            except Exception as e:  # keep going: one gated / missing set must not stop the rest
                print(f"FAIL {path} {cfg or ''} {split}: {type(e).__name__}: {str(e)[:120]}", flush=True)
