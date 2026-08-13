# Milestone-0 — DeepSeek-V4-Flash-0731 performance targets (Blackhole Galaxy)

**Readiness item:** 6.x / target authoring — owner: Benchmark tooling + Partner
**RFP references:** Appendix B.0 (per-system target expression), B.1/B.2 (sweep grid / graded points), F.1 (scaling-quality rubric). Builds on the device config ([§5.4](m0-blackhole-galaxy-device-config.md)), the scaling-quality rule ([§5.7](m0-scaling-quality-sweep-coverage.md)), and the AIPerf E2E verification ([§7.1](m0-aiperf-e2e-verification.md)).

## Source

The Appendix B AIPerf target sheet (OpenRouter competitive analysis + BH-Galaxy
projection) for `deepseek-ai/DeepSeek-V4-Flash-0731`. Key derived system numbers
(32× P150 Blackhole Galaxy):

| Sheet field | Value |
| ----------- | ----- |
| ASICs / BH-GLX | 32 |
| Aggregate decode throughput (`t/s/BH-GLX`, output) | 6,413 t/s |
| Aggregate prefill throughput | 371,835 t/s |
| Approximate max-num-seqs @ max context | 64 |
| Target Mean ISL | 8,192 |
| Target interactivity (per-user decode) | 100 t/s/u |
| Downrate factors | 0.50 (prefill), 0.25 (interactivity) |

## What was authored

### `model_performance_reference.json`

Key `DeepSeek-V4-Flash-0731` → `blackhole_galaxy`, three graded points at the
sheet's operating concurrency (64), bracketing the 8,192 mean ISL:

| ISL | OSL | conc | `theoretical` ttft_ms | `theoretical` tput_user | `theoretical` tput |
| --- | --- | ---- | --------------------- | ----------------------- | ------------------ |
| 4096 | 128 | 64 | 1410 | 25 | 1603 |
| 8192 | 128 | 64 | 2820 | 25 | 1603 |
| 16384 | 128 | 64 | 5640 | 25 | 1603 |

The JSON's `theoretical` block holds the **published Milestone-0 target** — the downrated
value, not the roofline peak. This spec sets `perf_targets_map: {functional: 1.0}`, so
exactly one graded tier is derived and it equals that value verbatim; the repo default
ladder (`functional 0.10 / complete 0.50 / target 1.0`) is deliberately not used. See
[the target convention](m0-target-convention.md).

### `workflows/model_specs/dev/llm.yaml` (DeepSeek BLACKHOLE_GALAXY spec)

- `max_concurrency: 64` — the sheet's max-num-seqs @ max context (was a placeholder 128).
- `max_tokens_all_users_override: 1056768` (=`64 × (16384 + 128)`) — sizes the device
  KV-token pool so all three graded ISLs reach concurrency 64. Without it the pool
  defaults to `max_context` (1M) and ISL 16384 demotes to concurrency 63, dropping the
  top graded level to 2 input lengths and breaking the §5.7 three-point rule (enforced
  by `get_llm_configs`).

## Derivation of the per-point numbers

**The sheet's figures are roofline peaks. The published target is peak × downrate**, per
[the Milestone-0 target convention](m0-target-convention.md). An earlier revision of this
document stored the peaks directly, which set the bar 2× (TTFT) and 4× (throughput) harder
than intended; the values below are the corrected, downrated ones.

- **Interactivity (`tput_user` = 25 t/s/u):** the sheet's 100 t/s/u peak × the 0.25
  interactivity downrate. Per-user, therefore ISL- and concurrency-independent, so every
  point carries it.
- **Aggregate decode (`tput` = 1,603 t/s):** the sheet's 6,413 t/s `t/s/BH-GLX` peak × the
  same 0.25 downrate. Cross-checks against interactivity: `25 × 64 = 1,600 ≈ 1,603`. It is a
  per-system figure and Blackhole Galaxy
  [expresses targets per system](m0-blackhole-galaxy-device-config.md), so it is used with
  no data-parallel scaling. Decode is ISL-independent, so all three points share it.
- **TTFT (1410 / 2820 / 5640 ms):** a linear-prefill model over the **downrated** prefill
  rate. `371,835 × 0.50 = 185,918` t/s aggregate, so per request at concurrency 64 that is
  `185,918 / 64 = 2,904.96` tok/s and `ttft_ms(ISL) = ISL / 2,904.96 × 1000`. Prefill is
  compute-bound and scales ~linearly with input length, so this is the standard first-order
  model — **not** a measured curve.
- **`tolerance` = 0.10** on every point, per the convention. Pass arithmetic in RFP G.2.4.

## Open decisions (flagged for sweep/target authoring + Partner)

1. **TTFT off the anchor point is modeled, not measured.** Only the 8,192-ISL / conc-64
   TTFT comes from the sheet; the 4,096 and 16,384 values are the linear-prefill
   extrapolation above. Replace with AIPerf-measured TTFT-vs-ISL once real 32-chip Galaxy
   numbers exist (the [AIPerf E2E harness](m0-aiperf-e2e-verification.md) already consumes exactly
   these shapes).
2. **Single graded concurrency level (64).** The sheet characterizes one operating point,
   so only concurrency 64 is graded. If the RFP wants a single-stream (conc 1) graded line
   too, add conc-1 points (each still needs ≥3 ISLs); their TTFT should come from a
   single-stream measurement, not the 64-way prefill share used here.
3. ~~**Tiering scheme.**~~ **Resolved** (llm-gauntlet #78). The per-metric downrates are
   applied when *authoring* the target rather than expressed as tier percentages — a single
   percentage per tier cannot carry two different factors. The spec now grades against one
   tier, `functional: 1.0`, holding the downrated value verbatim, with
   `status: FUNCTIONAL` so that tier is enforced. See
   [the target convention](m0-target-convention.md).
4. **`max_concurrency` / pool are dev-catalog placeholders.** They are grounded in the
   sheet but must be re-confirmed against the Partner's contributed tt-metal impl and the
   measured 32-chip KV pool before promotion to the prod catalog.

## Validation

- `model_performance_reference.json` parses; `get_perf_reference_map` resolves the
  `blackhole_galaxy` key to 3 points, and `get_perf_reference` returns them verbatim
  (per-system, unscaled).
- The dev catalog expands the spec with `max_concurrency=64`, pool `1,056,768`, and the
  3 perf-reference points attached.
- `get_llm_configs(spec, BLACKHOLE_GALAXY)` builds the sweep and the graded set is exactly
  `{(4096,64),(8192,64),(16384,64)}` — the §5.7 three-point guard passes.
- `tests/test_perf_reference_per_system.py`, `tests/llm_module/test_scaling_quality_coverage.py`,
  `tests/test_model_specification.py`, `tests/llm_module/test_benchmark_configs.py` all green.
