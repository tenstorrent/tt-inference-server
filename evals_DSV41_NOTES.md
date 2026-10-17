# DeepSeek-V4.1-Flash: additional evals (long context, reasoning/knowledge/code, image input)

Branch `ssinghal/dsv4p1-evals2` (on `ssinghal/dsv4p1`). Nothing here has been run against a server or a device: config
parsing, dataset loading, prompt formatting and answer extraction were dry-run on the shell machine
(`tests/test_dsv41_eval_tasks.py`, `/mnt/tt-data/ssinghal/datasets/dsv41_evals/dryrun_*.py`).

Source of every number: the model card (`/mnt/tt-data/ssinghal/deepseek-v41-flash/README.md`, same as
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). **All numbers below are from the card's "Base Model" table**
(card's internal framework, base model, few-shot, no chat template). Our runs use the instruct/chat server (non-thinking
chat template, temperature 1.0 / top-p 0.95 for the text tasks, greedy for the lmms-eval image tasks), so none is like for
like; the gate is a sanity gate: `accuracy >= published * 0.95` (one-sided, the workflow's published-score fallback).

## What is selected how

`DSV41_EVAL_TASKS` (comma list) selects tasks. The default run (variable unset) is unchanged: the same 9 tasks as before
(`gpqa_diamond_cot_zeroshot, mmlu_pro, humaneval_instruct, longbench2_generate, dsv41_gsm8k_cot, minerva_math, chartqa,
docvqa_val, mmmu_val`). The new tasks (`_DSV41_OPT_IN_TASKS` in `reference_config/evals/eval_config.py`) run only when named.
`DSV41_EVAL_CONCURRENCY` overrides `max_concurrent` of every selected task (set it to the server's user count).

## Table

| Family | Eval (opt-in task name) | Card number and setting | Our setting | Dataset (HF id) | Server config | Est. runtime | Gate (pass if >=) |
|---|---|---|---|---|---|---|---|
| 1 long ctx | LongBench-V2 at 262144 (`dsv41_longbench2_262k`) | 45.2 EM, 1-shot, DeepSeek-V4.1-Flash-Base | 0-shot, chat, non-thinking, T=1.0/top_p 0.95, max 4096 gen tokens, "reason step by step, answer in \boxed{}", **404 of 503 questions** (context <= 256000 tokens by the V4.1 tokenizer: 179/180 short, 204/215 medium, 21/108 long) | `recursal/longbench-v2` (20 subsets, what `longbench2_generate` loads) | 16 users x 262144: tt-metal `assets/runtime_spec_b16_k5_262144_vision.json` | prefill 33.9M tokens: about 1.7 h at the measured ~5.5k tok/s aggregate (128k x4 TTFT 95 s) plus decode: plan 2-4 h (estimate) | 42.9 |
| 1 long ctx | LongBench-V2 at 1M (`dsv41_longbench2_1m`) | same 45.2 | same, **472 of 503** (context <= 1,040,000 tokens; 31 questions of up to 4.6M tokens cannot fit any server) | same | 4 users x 1048576: `assets/runtime_spec_b4_k5_1048576.json` | prefill 64.1M tokens; at the measured 100,020-token prefill of 164 s (610 tok/s, 1M server) that is about 29 h: not practical in full. `DSV41_LONGBENCH_MIN_ISL=520000` keeps the 17 questions > 520k tokens (12.2M tokens, about 6 h, but 17 questions is +-12 points of noise: smoke only). Skipped automatically on a server with max_context < 1048576 | 42.9 (meaningful only on the full set) |
| 2 reasoning | BBH, 3-shot CoT (`bbh_cot_fewshot`, 27 subtasks, 6511 q) | 86.1 EM, 3-shot, base | lm-eval `bbh_cot_fewshot` (the 3 exemplars of the task), chat template, T=1.0, max 2048, stop `Q:`, filter `get-answer` (text after "the answer is") | `SaylorTwift/bbh` | the 16-user 131072/262144 or the B=32 server | about 1-2 h at 16-32 concurrency (estimate; GSM8K, 1319 q, takes 20 min) | 81.8 |
| 2 reasoning | already in the default run, unchanged: `mmlu_pro` 74.1 (5-shot EM), `humaneval_instruct` 79.4 (0-shot pass@1), `minerva_math` 61.1 (MATH, 4-shot EM), `gpqa_diamond_cot_zeroshot` 90.9 (instruct, **max reasoning effort**, pass@1: our server is non-thinking, so it is not comparable) | | | `TIGER-Lab/MMLU-Pro`, `openai/openai_humaneval`, `EleutherAI/hendrycks_math`, `Idavidrein/gpqa` (gated) | | | |
| 3 image | MMMU-Pro, 10-option standard (`mmmu_pro_standard`, lmms-eval) | 56.5 EM, 4-shot, base (card does not say standard or vision) | 0-shot, greedy, "Answer with the option letter", all 1730 q, up to 7 images each | `MMMU/MMMU_Pro` config `standard (10 options)` (682 MB) | vision spec: `runtime_spec_b16_k5_131072_vision.json` (images <= 1026 tokens each; limit 16 per prompt) | about 20-40 min (estimate) | 53.7 |
| 3 image | CV-Bench (`dsv41_cv_bench`, custom lmms-eval task `reference_config/evals/lmms_eval_tasks/dsv41_cv_bench`) | 77.9 EM, 4-shot, base | 0-shot, greedy, option letter, all 2638 q, score = official mean of 2D (mean of ADE20K and COCO) and 3D accuracy; also `cvbench_2d`, `cvbench_3d` | `nyu-visionx/CV-Bench` (405 MB) | same vision spec | about 10-20 min (estimate) | 74.0 |

Card numbers for the image family that are **not** used because the setting does not match what the harness computes:
DocVQA 95.6 is an LLM-judge score (the workflow's `docvqa_val` reports ANLS, so its `published_score` stays None);
RefCOCO-avg 86.0 (Acc@0.5, grounding boxes) has no lmms-eval task wired here. The card's instruct table has no ChartQA,
MathVista or MMMU number (only the agentic Chartography / BabyVision / ZeroBench with tools, which need an agent harness).

## What the card does not have (so nothing is gated)

* No needle-in-a-haystack, multi-needle, MRCR, CorpusQA or RULER number. LongBench-V2 (45.2) is the only long-context
  row. lm-eval's `ruler` tasks exist in the EVALS_COMMON venv but would be ungated; not added. (The tt-metal side has its own
  100,020-token needle probe.)
* No GPQA / MMLU-Pro / HumanEval / MATH number for the **instruct** model except GPQA Diamond 90.9 (max effort); the
  other rows are base-model few-shot rows.
* MGSM (80.2) is in the base table, but lm-eval's `mgsm_cot_native` is a tag with no aggregate score; not added.

## Comparing with the card

* LongBench-V2: our run is a subset (the questions that fit), zero-shot and sampled; the 262k subset keeps 80 % of the
  questions but only 21 of the 108 "long" ones, i.e. it is easier than the full set. The 5 % one-sided gate is therefore a
  sanity check; an expected score is in the 40s, not exactly 45.2.
* BBH / MMMU-Pro / CV-Bench: base-model few-shot numbers against an instruct, zero-shot (images) run.

## Datasets and offline use

The hosts' client container downloads datasets from the HF hub into its HF cache (as for GSM8K; `HF_CACHE` of
`run_gsm8k_eval.sh`). To run without internet, a cache is prefilled at **`/mnt/tt-data/ssinghal/datasets/hf_home`** (3.5 GB):
GSM8K, MMLU-Pro, HumanEval, GPQA diamond (gated: uses the user's token; do not redistribute), hendrycks_math (7 subjects),
BBH (27), `recursal/longbench-v2` (20), MMMU-Pro standard, CV-Bench. Mount it as the client's `/hf` and set
`HF_HUB_OFFLINE=1`. Loading `dsv41_longbench2_262k` with `HF_HUB_OFFLINE=1` was verified (404 kept). Not prefetched (large, older
tasks): `lmms-lab/DocVQA` (several GB), `lmms-lab/ChartQA`, `lmms-lab/MMMU`. Rebuild or refill the cache with
`scripts/dsv41_prefetch_eval_datasets.py [text|longbench|vision]` (needs an HF token for GPQA). The token counts per
LongBench-V2 question are in `/mnt/tt-data/ssinghal/datasets/dsv41_evals/longbench2_isl.jsonl`.
No subsets were built: every task uses its full set (the LongBench tasks the length-filtered set), so the card
comparison is not diluted by subsampling. `--limit-samples-mode smoke-test|ci-nightly` takes the first N questions; for CV-Bench
these are all 2D (the 3D ones come last; the aggregate then falls back to 2D).

## Commands (against a running server)

Host-side (from this worktree; flags as in tt-metal `assets/run_gsm8k_eval.sh`; `CKPT=/mnt/tt-data/ssinghal/deepseek-v41-flash`):

```
export MODEL_SPECS_ENV=dev HF_HOME=/mnt/tt-data/ssinghal/datasets/hf_home HF_HUB_OFFLINE=1
RUN="python run.py --model DeepSeek-V4.1-Flash --tt-device blackhole_galaxy --dev-mode --no-auth --skip-system-sw-validation \
  --host-weights-dir $CKPT --server-url http://127.0.0.1:8000 --service-port 8000 --workflow evals"
# 1  long context, 16 users x 262144
DSV41_EVAL_TASKS=dsv41_longbench2_262k DSV41_EVAL_CONCURRENCY=16 $RUN --runtime-model-spec-json <tt-metal>/.../assets/runtime_spec_b16_k5_262144_vision.json
# 1  long context, 4 users x 1M (add DSV41_LONGBENCH_MIN_ISL=520000 for the 17 longest questions only)
DSV41_EVAL_TASKS=dsv41_longbench2_1m DSV41_EVAL_CONCURRENCY=4 $RUN --runtime-model-spec-json <tt-metal>/.../assets/runtime_spec_b4_k5_1048576.json
# 2  BBH
DSV41_EVAL_TASKS=bbh_cot_fewshot DSV41_EVAL_CONCURRENCY=16 $RUN --runtime-model-spec-json <...>/runtime_spec_b16_k5_131072.json
# 3  images
DSV41_EVAL_TASKS=mmmu_pro_standard,dsv41_cv_bench DSV41_EVAL_CONCURRENCY=16 $RUN --runtime-model-spec-json <...>/runtime_spec_b16_k5_131072_vision.json
```

(Smoke check: append `--limit-samples-mode smoke-test`.) The first run of each task installs its workflow venv
(EVALS_COMMON / EVALS_VISION). Reports: `workflow_logs/reports_output/evals/`.

In the client container: the image's tt-inference-server tree must contain these files (rebuild the image from this branch, or bind-mount
`reference_config/evals` of this worktree over `/home/container_app_user/tt-inference-server/reference_config/evals`), and tt-metal's
`assets/run_gsm8k_eval.sh` hardcodes `-e DSV41_EVAL_TASKS=dsv41_gsm8k_cot` (one line to change to `${TASKS:-dsv41_gsm8k_cot}`; not done here).

## Harness limits (plainly)

* The 1M run is limited by prefill speed, not by the harness (about 29 h for the full 472-question set at the measured 610 tok/s).
* lm-eval sends one question per request: each long request is a single 100k-1M-token chat message; the client needs
  `timeout` 28800 / 86400 s (set) and the server's `max_model_len` must cover prompt + 4096.
* The image tasks need the vision build (`DSV41_VISION=1`); text-only specs reject `image_url` parts. Whether the workflow
  accepts a `model_type: LLM` runtime spec for the lmms-eval venv is unverified on a device; if it refuses, use
  `--impl deepseek-v41-flash-vlm` (workflows/model_specs/dev/vlm.yaml) instead of `--runtime-model-spec-json`.
* lmms-eval's `openai_compatible` model sends greedy requests (temperature 0) and caps `max_new_tokens` at 4096.
* Few-shot settings of the card (4-shot image rows) are not reproduced: lmms-eval image tasks are zero-shot.
