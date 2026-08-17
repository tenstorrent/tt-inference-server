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

Key `DeepSeek-V4-Flash-0731` → `blackhole_galaxy`, **22 graded points**: eleven input
lengths swept at both concurrency corners. Input lengths are powers of two from 1K to
512K, plus a context-saturating point.

| ISL | ttft_ms @ conc 1 | ttft_ms @ conc 64 |
| --- | ---------------- | ----------------- |
| 1,024 | 5.5078 | 352.5004 |
| 2,048 | 11.0157 | 705.0009 |
| 4,096 | 22.0313 | 1,410.0017 |
| 8,192 | 44.0626 | 2,820.0035 |
| 16,384 | 88.1252 | 5,640.0069 |
| 32,768 | 176.2505 | 11,280.0139 |
| 65,536 | 352.5009 | 22,560.0277 |
| 131,072 | 705.0019 | 45,120.0555 |
| 262,144 | 1,410.0038 | 90,240.1109 |
| 524,288 | 2,820.0075 | 180,480.2219 |
| 1,048,448 | 5,639.3185 | 360,916.3850 |

Every point carries `tput_user` 25 t/s/u and `tolerance` 0.10. `tput` is `25 ×
concurrency` — 25 at the idle corner, 1,600 at the loaded one.

**Why 1,048,448 and not 2^20.** A full 2^20 input plus any output exceeds `max_context`:
`1,048,576 + 128 > 1,048,576`, and `get_benchmark_max_concurrency` answers that by
silently returning concurrency 1 rather than rejecting the point. The top input length is
therefore `max_context − osl`, which saturates the context window exactly.

**Why `tput` is 1,600 and not the sheet's 1,603.** Measured aggregate decode is defined as
`tput_user × concurrency` (`llm_module.parsers.base.decode_throughput`), so the target has
to be derived the same way or a system hitting interactivity exactly still misses the bar.
6,413 × 0.25 = 1,603 was the sheet's aggregate downrated independently; the 3-token gap is
its rounding.

**Values are stored to 4 decimal places** so the derivation reproduces. At 1 dp the idle
target at ISL 1,024 rounds to 5.5, and ×64 then gives 352.0 against the loaded corner's
352.5 — the convention promises a Partner can recompute any published value, and that
promise fails at 1 dp.

The JSON's `theoretical` block holds the **published Milestone-0 target** — the downrated
value, not the roofline peak. This spec sets `perf_targets_map: {functional: 1.0}`, so
exactly one graded tier is derived and it equals that value verbatim; the repo default
ladder (`functional 0.10 / complete 0.50 / target 1.0`) is deliberately not used. See
[the target convention](m0-target-convention.md).

### `workflows/model_specs/dev/llm.yaml` (DeepSeek BLACKHOLE_GALAXY spec)

- `max_concurrency: 64` — the sheet's max-num-seqs @ max context (was a placeholder 128).
- `max_tokens_all_users_override: 67108864` (=`64 × 1024 × 1024`) — 64 concurrent requests
  at the full 1M context. This is what lets the loaded corner reach concurrency 64 at
  *every* graded input length: `67,108,864 / (1,048,448 + 128) = 64` exactly.

  It was previously `1,056,768` (=`64 × 16,512`), sized only for the three ISLs then
  graded — while the inline comment already read `64 * 1024 * 1024`. Under that value
  every input length above 16K demoted (32K→32, 64K→16, 128K→8, 256K→4, 512K→2), which
  breaks the §5.7 three-point rule *and* leaves seven concurrency levels where Appendix
  B.5's per-point weights are defined over exactly two.

## Derivation of the per-point numbers

**The sheet's figures are roofline peaks. The published target is peak × downrate**, per
[the Milestone-0 target convention](m0-target-convention.md). An earlier revision of this
document stored the peaks directly, which set the bar 2× (TTFT) and 4× (throughput) harder
than intended; the values below are the corrected, downrated ones.

- **Interactivity (`tput_user` = 25 t/s/u):** the sheet's 100 t/s/u peak × the 0.25
  interactivity downrate. Per-user, therefore ISL- and concurrency-independent, so every
  point carries it.
- **Aggregate decode (`tput` = `25 × concurrency`):** 25 t/s at the idle corner, 1,600 t/s
  at the loaded one. Derived from interactivity rather than from the sheet's 6,413 t/s
  `t/s/BH-GLX` peak, because that is how the *measurement* is defined
  (`llm_module.parsers.base.decode_throughput`). It is a per-system figure and Blackhole
  Galaxy [expresses targets per system](m0-blackhole-galaxy-device-config.md), so no
  data-parallel scaling is applied. Decode is ISL-independent, so every point at a corner
  shares it.
- **TTFT:** a linear-prefill model over the **downrated** prefill rate.
  `371,835 × 0.50 = 185,917.5` t/s aggregate, shared by however many requests are in
  flight, so `ttft_ms(ISL) = ISL / (185,917.5 / concurrency) × 1000`:

  | corner | per-request prefill rate |
  | ------ | ------------------------ |
  | concurrency 1 | 185,917.5 tok/s — one request has the whole machine |
  | concurrency 64 | 2,904.96 tok/s |

  The loaded corner is therefore exactly 64× the idle one at every input length. Prefill is
  compute-bound and scales ~linearly with input length, so this is the standard first-order
  model — **not** a measured curve.
- **`tolerance` = 0.10** on every point, per the convention. Pass arithmetic in RFP G.2.4.

## Open decisions (flagged for sweep/target authoring + Partner)

1. **Every TTFT except one is modeled, not measured.** The sheet gives a single operating
   point (8,192 ISL at concurrency 64); all 22 published values come from the linear-prefill
   model above. Replace with AIPerf-measured TTFT-vs-ISL once real 32-chip Galaxy numbers
   exist (the [AIPerf E2E harness](m0-aiperf-e2e-verification.md) already consumes exactly
   these shapes). Expect the long inputs to deviate first: a real attention implementation
   grows faster than linearly, which is precisely what the scaling-quality line scores.
2. ~~**Single graded concurrency level (64).**~~ **Resolved.** Both corners are now graded —
   concurrency 1 and 64 — at all eleven input lengths, which is what Appendix B.5's 25/75
   per-point weights are defined over. The concurrency-1 TTFTs are the same model with the
   machine's full prefill capability given to one request, so they are a projection rather
   than a single-stream measurement; the same caveat as (1) applies.
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
  `blackhole_galaxy` key to 22 points, and `get_perf_reference` returns them verbatim
  (per-system, unscaled).
- The dev catalog expands the spec with `max_concurrency=64`, pool `67,108,864`, and the
  22 perf-reference points attached.
- `get_llm_configs(spec, BLACKHOLE_GALAXY)` with `ONLY_BENCHMARK_TARGETS=1` builds exactly
  those 22 points, every one graded, at exactly two concurrency levels with eleven distinct
  input lengths each — the §5.7 three-point guard passes at both.
- `report_module.scorecard.point_weights` accepts the sweep: weights sum to 1.0000, the
  loaded corner carries 0.75, and the heaviest single point is the longest input under load.
- Longest point fits the context window exactly: `1,048,448 + 128 = 1,048,576`.
- `tests/test_perf_reference_per_system.py`, `tests/llm_module/test_scaling_quality_coverage.py`,
  `tests/test_model_specification.py`, `tests/llm_module/test_benchmark_configs.py` all green.
