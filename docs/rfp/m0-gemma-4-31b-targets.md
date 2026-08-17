# Milestone-0 — gemma-4-31B-it performance targets (Blackhole Galaxy)

**Readiness item:** §5.2 target authoring (llm-gauntlet #66) — owner: Workstream 1
**RFP references:** Appendix B.0 (per-system target expression), B.1/B.2 (sweep grid and
graded points), F.1 (scaling-quality rubric).

Authored under [the Milestone-0 target convention](m0-target-convention.md), which is the
authority for what these numbers mean. Companion to
[the DeepSeek-V4-Flash targets](m0-deepseek-v4-flash-targets.md).

## Source

| Quantity | Value |
| -------- | ----- |
| Aggregate prefill throughput | 123,072 tok/s |
| Interactivity (per-user decode) | 35.0 t/s/u |
| Max concurrent requests at full context | 32 |
| Max context | 262,144 (256K) |
| Numerics | fp8 |

**These are already Milestone-0 targets — the downrate is included.** Unlike the DeepSeek
sheet row, which publishes roofline peaks alongside separate downrate factors, these figures
arrive post-downrate and are stored verbatim. Nothing further is applied to them.

> An earlier working figure of **184,608 tok/s** aggregate prefill circulated (it is still in
> the `llm-gauntlet` simulator profile, which needs correcting). It is 1.5× the real target.
> Anything derived from it — including the simulator's calibrated latency profile and any
> percentile multipliers taken from its output — is wrong by that factor.

## What was authored

Key `gemma-4-31B-it` → `blackhole_galaxy`, **18 graded points**: nine input lengths at both
concurrency corners.

| ISL | ttft_ms @ conc 1 | ttft_ms @ conc 32 |
| --- | ---------------- | ----------------- |
| 1,024 | 8.3203 | 266.2507 |
| 2,048 | 16.6407 | 532.5013 |
| 4,096 | 33.2813 | 1,065.0026 |
| 8,192 | 66.5627 | 2,130.0052 |
| 16,384 | 133.1253 | 4,260.0104 |
| 32,768 | 266.2507 | 8,520.0208 |
| 65,536 | 532.5013 | 17,040.0416 |
| 131,072 | 1,065.0026 | 34,080.0832 |
| 262,016 | 2,128.9652 | 68,126.8851 |

Every point carries `tput_user` 35.0 t/s/u and `tolerance` 0.10. `tput` is
`35 × concurrency` — 35 at the idle corner, 1,120 at the loaded one.

### Derivation

`ttft_ms(ISL) = ISL / (123,072 / concurrency) × 1000`. The machine's prefill capability is
fixed and divided among the requests in flight, so the loaded corner is exactly 32× the idle
one at every input length:

| corner | per-request prefill rate |
| ------ | ------------------------ |
| concurrency 1 | 123,072 tok/s — one request has the whole machine |
| concurrency 32 | 3,846 tok/s |

Prefill is compute-bound and scales roughly linearly with input length, so this is the
standard first-order model — **not** a measured curve. `tput` is derived as
`tput_user × concurrency` because that is how the measurement is defined
(`llm_module.parsers.base.decode_throughput`); deriving it any other way would mean a system
hitting interactivity exactly still misses the bar. Values are stored to 4 decimal places so
a Partner can reproduce the derivation, as the convention promises.

**Why the top input length is 262,016 and not 262,144.** A full 2^18 input plus any output
exceeds `max_context`, and `get_benchmark_max_concurrency` answers that by silently returning
concurrency 1 rather than rejecting the point — so the mistake surfaces as a stray concurrency
level, not an error. The top input is `max_context − osl`, which saturates the window exactly.

## `workflows/model_specs/dev/llm.yaml`

- `max_concurrency: 32` — expected max-num-seqs at full context; was a placeholder 128.
- `max_tokens_all_users_override: 8388608` (= `32 × 256 × 1024`) — 32 concurrent requests at
  the full 256K context, so the loaded corner reaches concurrency 32 at *every* graded input
  length: `8,388,608 / (262,016 + 128) = 32` exactly.

  It previously read `32 * 256 * 1024` unquoted, which **YAML does not evaluate**: the value
  loaded as the string `'32 * 256 * 1024'` and `get_benchmark_max_concurrency` raised
  `TypeError: unsupported operand type(s) for //: 'str' and 'int'`. Nothing caught it because
  the model had no `blackhole_galaxy` targets, so the sweep never reached that code path. The
  arithmetic in the comment was right; only the literal was wrong.
- `perf_targets_map: {functional: 1.0}` and `status: FUNCTIONAL`, per the convention.

## The reference-file key

Targets are filed under **`gemma-4-31B-it`** — upper-case `B` — because that is what the
Milestone-0 spec derives from `google/gemma-4-31B-it`. The pre-existing `gemma-4-31b-it`
(lower-case) belongs to the **Forge** spec, which derives that spelling from
`google/gemma-4-31b-it` to match `tt-media-server`'s `ModelNames.GEMMA_4_31B_IT`, and owns the
`p300x2` entry there.

Both keys therefore exist and neither is renamed. Renaming either would move targets off the
spec that reads them — which is why tenstorrent#4884 only added a warning rather than
"fixing" the spelling.

## Open items

1. **Every TTFT is modelled, not measured.** Expect the long inputs to deviate first: a real
   attention implementation grows faster than linearly, which is exactly what the
   scaling-quality line scores. Replace with AIPerf-measured TTFT-vs-ISL once 32-chip Galaxy
   numbers exist.
2. **`max_concurrency` and the KV pool are expectations, not measurements.** Both must be
   re-confirmed against the measured 32-chip pool. If the real pool is smaller, the top graded
   input lengths must come down with it, or the loaded corner fragments into several
   concurrency levels and breaks both the three-point rule and the Appendix B.5 weights.
3. **The `llm-gauntlet` simulator profile still carries 184,608 tok/s** and must be corrected
   to 123,072 before any calibration is taken from its output.

## Validation

- `get_perf_reference_map("gemma-4-31B-it", {"functional": 1.0})` resolves 18 points; the
  case-mismatch warning for this spec is gone.
- `get_llm_configs` with `ONLY_BENCHMARK_TARGETS=1` builds exactly those 18 points, every one
  graded, at two concurrency levels with nine distinct input lengths each — the §5.7
  three-point guard passes at both.
- `report_module.scorecard.point_weights` accepts the sweep: weights sum to 1.0000 and the
  loaded corner carries 0.75.
- Longest point fits the context window exactly: `262,016 + 128 = 262,144`.
- Rough sweep runtime ≈ 31 min (19.8 idle + 11.3 loaded).
