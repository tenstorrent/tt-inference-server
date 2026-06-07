# #2 — Uplift the BENCHMARKS_VLLM client venv to vllm 0.19.1 / transformers 5.5.1

Part of the forge-TP-p300x2 umbrella, but **cross-cutting** — its own PR, qualified separately.
Prereq for gemma-4-31b-it (and any new-tokenizer model) benchmarks; **not** in the initial-support PR (#1).

## Problem
The benchmark client venv pins `vllm==0.13.0` → `transformers<5` → 4.57.6, which **can't load the
gemma-4 tokenizer** (`extra_special_tokens` is a list, not a dict →
`AttributeError: 'list' object has no attribute 'keys'`). All gemma benchmark runs crash (CI #5203 + local).

Note: only the **benchmark client** lagged — the **server** ships transformers 5.5.1 and the **eval**
client uses 5.10.2 (both load gemma fine). So serving/evals are unaffected.

## Fix
`requirements/benchmarks-vllm.txt`: `vllm 0.13.0 → 0.19.1`, pin `transformers==5.5.1`;
`workflows/workflow_venvs.py`: `VLLM_PIN_VERSION 0.13.0 → 0.19.1` (matching qualification + the serving
image). Prototyped/validated on `kmabee/gemma4_31b_it_forge` as commit `6e6c22bb` (reverted off that
branch in `7ef618a9` to keep #1 independent — cherry-pick `6e6c22bb` for this PR).

## Scope / risk
`BENCHMARKS_VLLM` is the **single shared client venv** for the default `tools=vllm` path → affects
**all** benchmarked LLMs (forge + TTNN `tt_transformers`/galaxy). But it's **client-only**
(token-counting + `vllm bench serve`) — no serving image, weights, or accuracy impact.

## Qualification before merge
Re-run benchmarks and compare deltas on: forge P150 (e.g. Qwen3-4B, Llama-3.1-8B), one T3K + one galaxy
TTNN model, and the two new forge-TP p300x2 models. Watch for: `vllm bench serve` arg/result-JSON drift
(0.13→0.19) and transformers 4→5 tokenizer/token-count changes.

## Validated
Local (vllm 0.19.1 + transformers 5.5.1): gemma-4 tokenizer loads; `vllm bench serve` accepts all
`run_benchmarks.py` args and reaches the request phase. Rollback = revert the 2 pins.
