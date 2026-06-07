# Issue: uplift the BENCHMARKS_VLLM client venv to vllm 0.19.1 / transformers 5.5.1

**Type:** cross-cutting CI/benchmark change · **Inspired by:** gemma-4-31b-it bring-up
**Proposed change is small (2 pins) but shared across all benchmarked LLMs — needs qualification.**

## Summary
The benchmark client venv (`BENCHMARKS_VLLM`, built from `requirements/benchmarks-vllm.txt`)
pins `vllm==0.13.0`, which hard-pins `transformers<5` → 4.57.6. That transformers can't load the
**gemma-4-31b-it** tokenizer (`extra_special_tokens` is a list, not a dict →
`AttributeError: 'list' object has no attribute 'keys'` in `tokenization_utils_base`). Uplift the
client to the pairing gemma-4 was qualified with in tt-xla and that the serving images ship:
**vllm 0.19.1 + transformers 5.5.1** (+ `VLLM_PIN_VERSION 0.13.0 → 0.19.1`).

## Why this is needed (gemma-4-31b-it–inspired)
- gemma-4-31b-it CI release benchmarks fail 100% (all runs crash at tokenizer load) — see
  `ADD_gemma4_31b_it_to_CI.md`. The blocker is purely the benchmark client's transformers version.
- Everything else is already on transformers 5.x: the **serving** images use transformers **5.5.1**,
  and the **eval** client venv uses **5.10.2**. Only the **benchmark** client lags on 4.57.6.
- New model families (gemma-4, and likely future ones) ship tokenizer configs that only transformers
  5.x parses. The benchmark client must keep pace or it silently blocks every new model's perf numbers.

## Scope / impact
- `BENCHMARKS_VLLM` is the **single shared venv** for the default `tools=vllm` benchmark path, used by
  **all** benchmarked LLMs:
  - **Forge** (`forge_vllm_plugin`): gemma-4-31b-it, Qwen3-32B (p300x2) + the P150 single-chip LLMs
    (Qwen3-4B, Llama-3.2-3B, Llama-3.1-8B, Qwen3-8B, Falcon3-7B).
  - **TTNN-based** (`tt_transformers` + galaxy-specialized `llama`/`qwen`/`deepseek_r`/`gpt_oss`):
    Llama-3.x, Qwen2.5/3, DeepSeek-R1, gpt-oss, Mistral, etc. on N150/N300/T3K/galaxy.
- **Client-only change.** It does NOT modify any serving image, model weights, or accuracy path — only
  how the benchmark harness tokenizes (token counting / prompt construction) and the `vllm bench serve`
  tool version. So the blast radius is benchmark *measurement*, not serving correctness.

## Proposed change
- `requirements/benchmarks-vllm.txt`: `vllm==0.13.0` → `vllm==0.19.1`, pin `transformers==5.5.1`.
- `workflows/workflow_venvs.py`: `VLLM_PIN_VERSION = "0.13.0"` → `"0.19.1"`
  (keeps `fetch_structured_output_scripts()` pulling matching drivers from `vllm@v0.19.1` — verified
  all 3 fetched files exist at that tag).
- (Already prototyped on branch `kmabee/gemma4_31b_it_forge`, commit `6e6c22bb` — to be cherry-picked
  to its own branch for this PR.)

## What else to consider / qualification plan
1. **`vllm bench serve` arg/behavior drift 0.13 → 0.19:** all flags `run_benchmarks.py` builds
   (`--backend --endpoint --dataset-name --max-concurrency --num-prompts --random-input-len
   --random-output-len --percentile-metrics --save-result --save-detailed --extra-body`) are accepted
   on 0.19.1 (verified by dry run). Watch for changed result-JSON keys consumed by `guidellm_report.py`
   / `summary_report.py`.
2. **transformers 4 → 5 tokenizer behavior:** token counts for some models could shift slightly →
   isl/osl targeting. Re-run a representative benchmark set and compare to current numbers.
3. **Structured-output benchmarks:** confirm `benchmark_serving_structured_output.py` +
   `backend_request_func.py` + schema still run on 0.19.1 (they exist at the tag; run one).
4. **Qualification set (run release/benchmarks before merge, compare deltas):**
   - Forge P150: Qwen3-4B, Llama-3.1-8B (representative single-chip).
   - TTNN: one T3K + one galaxy model (e.g., Llama-3.1-70B / Qwen3-32B vLLM).
   - The two new forge-TP p300x2 models (gemma-4-31b-it, Qwen3-32B) — gemma's benchmark now passes the
     tokenizer load with this uplift (local re-verify).
5. **Orthogonal, do NOT conflate:** the Qwen3-32B 6h benchmark timeout is a *runtime* problem (conc-1
   ~16× slow at batch-16), unrelated to this uplift. Track separately.

## Validation done so far
- Throwaway venv `vllm 0.19.1 + transformers 5.5.1`: `AutoTokenizer.from_pretrained("google/gemma-4-31B-it")`
  loads; `vllm bench serve` accepts all run_benchmarks args and reaches the request phase.
- gemma-4-31b-it end-to-end local re-verify (rebuilt benchmark venv) — in progress / see
  `PERF_gemma4_31b_it_forge.md`.

## Risk / rollback
Low and contained (client-only). Rollback = revert the 2 pins. No serving/image/accuracy impact.
