# Uplift the BENCHMARKS_VLLM client venv to vllm 0.19.1 / transformers 5.5.1

Part of the forge-TP-p300x2 umbrella, but **cross-cutting** — its own PR, qualified separately.
Prereq for gemma-4-31b-it (and any new-tokenizer model) benchmarks; **not** in the initial-support PR.

## Problem
The benchmark client venv pins `vllm==0.13.0` → `transformers<5` → 4.57.6, which **can't load the
gemma-4 tokenizer** (`extra_special_tokens` is a list, not a dict →
`AttributeError: 'list' object has no attribute 'keys'`). All gemma benchmark runs crash (CI + local).

Only the **benchmark client** lagged — the **server** ships transformers 5.5.1 and the **eval** client
uses 5.10.2 (both load gemma fine). So serving/evals are unaffected.

## Fix
`requirements/benchmarks-vllm.txt`: `vllm 0.13.0 → 0.19.1`, pin `transformers==5.5.1`;
`workflows/workflow_venvs.py`: `VLLM_PIN_VERSION 0.13.0 → 0.19.1` (matching qualification + the serving
image). Prototyped/validated on `kmabee/gemma4_31b_it_forge` (commit `6e6c22bb`, reverted off that
branch in `7ef618a9` to keep the initial-support PR independent — cherry-pick `6e6c22bb` for this PR).

## Scope / risk
`BENCHMARKS_VLLM` is the **single shared client venv** for the default `tools=vllm` path → affects
**all** benchmarked LLMs (forge + TTNN `tt_transformers`/galaxy). But it's **client-only**
(token-counting + `vllm bench serve`) — no serving image, weights, or accuracy impact.

## Qualification before merge
Re-run benchmarks and compare deltas on: forge P150 (e.g. Qwen3-4B, Llama-3.1-8B), one T3K + one galaxy
TTNN model, and the two new forge-TP p300x2 models. Watch for: `vllm bench serve` arg/result-JSON drift
(0.13→0.19) and transformers 4→5 tokenizer/token-count changes.

## Validated (local, p01t05)
Before/after on the gemma-4-31b-it tokenizer:
- **Before** (vllm 0.13.0 / transformers 4.57.6): `AutoTokenizer.from_pretrained("google/gemma-4-31B-it")`
  → `AttributeError: 'list' object has no attribute 'keys'`; CI release run 5203 crashed all 14
  benchmark runs the same way.
- **After** (vllm 0.19.1 / transformers 5.5.1): tokenizer loads (vocab 262144); **`vllm bench serve`
  against a live gemma server completed 2/2 requests** (output throughput 3.90 tok/s, TPOT 204 ms),
  **no AttributeError**. The crash was at tokenizer-init (pre-request), so a completed request proves
  the fix.

Caveats: the "after" run used a throwaway 0.19.1/5.5.1 venv (same versions + same `vllm bench serve`
command run.py issues), not the committed branch venv (reverted to 0.13.0) and not the full
`run.py --workflow release` orchestration; it was a 2-request smoke, not the full sweep. Rollback =
revert the two pins.
