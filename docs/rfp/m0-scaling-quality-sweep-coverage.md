# Milestone-0 — Scaling-quality sweep coverage (three-point rule)

**Readiness item:** 5.7 — Blocker — owner: Performance
**RFP references:** Appendix B.1 (sweep grid), Appendix B.2 (graded points), Appendix F.1 (scaling-quality rubric line), readiness §5.7. Interacts with the device configuration ([§5.4](m0-blackhole-galaxy-device-config.md)).

The scaling-quality rubric line fits **time-to-first-token against input length,
separately at each graded concurrency level**. A regression needs at least three
points, so the RFP requires a hard constraint on how the graded sweep is authored:

> **Every graded concurrency level must carry at least three distinct input lengths.**

A sweep with five input lengths at concurrency 1 but only two at maximum
concurrency makes the high-concurrency fit meaningless — and that level carries
75 % of the line. This is a **Blocker**: it must be settled before the sweep and
targets are authored (sweep/target work), because it constrains *which* points can be graded.

---

## Summary

| # | Item | Decision |
| - | ---- | -------- |
| 1 | The rule | Each graded concurrency level needs ≥ **3** distinct input lengths (`SCALING_QUALITY_MIN_INPUT_LENGTHS`). |
| 2 | Enforcement | Machine-checked in `get_llm_configs`: a scaling-quality-graded device whose **post-cap** graded set violates the rule fails fast, before a benchmark run is wasted. |
| 3 | Feasibility | With the token pool defaulted to `max_context` (262144), **only ISLs 128 and 1024 reach concurrency 128** — 2 points, a violation. The top gradeable concurrency at that pool is **120**. |
| 4 | Fix | Size the device KV-token pool so ≥3 input lengths reach the top graded concurrency: set `max_tokens_all_users_override` ≥ **278528** on the Blackhole Galaxy specs, or rescope the graded concurrency levels. |

---

## 1. Why the naive sweep fails at the top concurrency

`reference_config/benchmarking/benchmark_config.py` expands each input/output pair
to two concurrency points: `1` and the per-shape **allowed max**
(`get_benchmark_max_concurrency = min(model_max_concurrency, max_tokens_all_users // (isl+osl))`).
The allowed max **shrinks as input length grows**, so the single "maximum
concurrency" value is only reached by the smallest input lengths. For the
Milestone-0 gemma-4-31B-it config (`max_context = 262144`, `max_concurrency = 128`)
with the token pool defaulted to `max_context = 262144`, the per-ISL ceilings are:

| ISL (osl=128) | 128 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
| ------------- | --- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| allowed max concurrency | 128 | 128 | 120 | 62 | 31 | 15 | 7 | 3 | 1 |

At concurrency **128** only **two** input lengths (128, 1024) survive — exactly the
§5.7 failure. The highest concurrency three distinct input lengths can share is
**120** (the third-largest ceiling). Capping is also silent: an authored graded
point at `(isl=2048, conc=128)` is capped down to `conc=120`, quietly dropping it
out of the concurrency-128 group.

## 2. Enforcement (this branch)

- `DeviceTypes.grades_scaling_quality()` (`workflows/workflow_types.py`) → `True`
  for `BLACKHOLE_GALAXY` (the Milestone-0 target).
- `get_llm_configs` (`llm_module/benchmark_configs.py`) validates the **post-cap
  graded set** (the points that actually carry targets and will be graded) for
  such devices and raises with an actionable message when any graded concurrency
  level has fewer than three distinct input lengths. Validating post-cap means
  context/token-budget capping that moves a point to a lower concurrency is
  caught here rather than surfacing as an ungradeable run downstream.
- Pure, reusable helpers in `reference_config/benchmarking/benchmark_config.py`:
  - `scaling_quality_coverage_violations(graded_points)` — the validator.
  - `input_lengths_reaching_concurrency(...)` — which input lengths physically
    reach a concurrency under a token budget.
  - `max_gradeable_concurrency(...)` — the highest concurrency ≥3 input lengths
    can share (120 at the default pool).
  - `min_token_pool_for_concurrency(target)` — the smallest `max_tokens_all_users`
    that lets ≥3 input lengths reach `target` (278528 for concurrency 128).
- Guarded by `tests/llm_module/test_scaling_quality_coverage.py`.

The guard is **vacuous for the current catalog** (prod has no `BLACKHOLE_GALAXY`
specs, and no Milestone-0 model has `blackhole_galaxy` targets yet), so it changes
no existing run. It activates the moment the graded sweep is authored — which is
precisely when the rule must hold.

## 3. Interaction with the device configuration (§5.4)

The `max_context = 262144` / `max_concurrency = 128` decisions from
[§5.4](m0-blackhole-galaxy-device-config.md) do **not** by themselves guarantee the
three-point rule at the top concurrency, because the token pool defaults to
`max_context`. A 32-chip Blackhole Galaxy has a much larger aggregate KV-token
pool than a single replica's context; the correct fix is to declare it via
`max_tokens_all_users_override` (a field that already decouples the benchmark token
budget from `max_context` without changing it).

Each Milestone-0 `BLACKHOLE_GALAXY` spec now carries a `[TBD — Performance]`
placeholder for this override with the computed floor:

```yaml
# max_tokens_all_users_override: 278528   # >= 128 * (2048 + 128)
```

Set it to the **measured** 32-chip pool (which will be well above the floor). The
floor `278528 = 128 × (2048 + 128)` is the minimum that lets input lengths
{128, 1024, 2048} all reach concurrency 128. If the measured pool cannot support
three input lengths at the intended top concurrency, **rescope** the graded
concurrency levels instead (e.g. grade at concurrency 1 and at
`max_gradeable_concurrency(...)`), so both graded levels satisfy the rule.

## Downstream

- **Sweep + targets authoring**: author the `blackhole_galaxy` graded set with
  ≥3 distinct input lengths at each graded concurrency level; run against the
  validator (now enforced automatically by `get_llm_configs`). Set
  `max_tokens_all_users_override` from the AIPerf measurement, or rescope.
- Feeds Appendix B.1/B.2/F.1 of the partner RFP: the graded concurrency levels and
  their input-length grids are constrained by this rule.
