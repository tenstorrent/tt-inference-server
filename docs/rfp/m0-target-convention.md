# Milestone-0 — how a performance target is authored and enforced

**Readiness item:** §9.2 (llm-gauntlet #78) — owner: Acceptance
**RFP references:** requirements G.2.4 (pass arithmetic), I.2 (the automated verdict),
Appendix B.2 (graded points and target values), B.5 (rubric values).

The single authority for authoring a Milestone-0 target. Read this before adding entries to
`reference_config/benchmarking/benchmark_targets/model_performance_reference.json` for a
Blackhole Galaxy model (llm-gauntlet #66).

It exists because the same question was answered two different ways once already: the first
DeepSeek-V4-Flash entries stored the roofline sheet's figures directly, which set the bar 2×
(TTFT) and 4× (throughput) harder than intended. What the stored number *means* has to be
settled once, in writing, or every model gets its own interpretation.

---

## 1. `theoretical` holds the published target, not the peak

The roofline sheet gives **peaks**. Milestone-0 publishes a **target**, and the target is the
peak with a downrate factor applied:

```
target = roofline peak × downrate factor
```

| Metric family | Downrate | Applies to |
| ------------- | -------- | ---------- |
| Prefill (and anything derived from it, i.e. TTFT) | **0.50** | `ttft_ms` |
| Interactivity and aggregate decode | **0.25** | `tput_user`, `tput` |

**The peak never enters the JSON.** Only the downrated target does, under the `theoretical`
key. The name is historical — it is the key the loader looks for, not a claim that the value
is a theoretical maximum.

Doing the downrating at authoring time, rather than expressing it as tier percentages, is
forced: the two factors differ by metric, and `perf_targets_map` carries **one percentage per
tier applied to all metrics**. There is no way to say "×0.50 for latency, ×0.25 for
throughput" in that structure.

### Worked example — DeepSeek-V4-Flash-0731

Sheet: aggregate prefill 371,835 t/s, interactivity 100 t/s/u, aggregate decode 6,413 t/s,
max-num-seqs 64.

```
prefill:  371,835 × 0.50 = 185,918 t/s aggregate
                         ÷ 64 concurrent = 2,904.96 t/s per request
          ttft_ms(ISL)   = ISL / 2,904.96 × 1000

decode:   tput_user = 100   × 0.25 =    25 t/s/u
          tput      = 6,413 × 0.25 = 1,603 t/s
```

| ISL | `ttft_ms` | `tput_user` | `tput` |
| --- | --------- | ----------- | ------ |
| 4 096 | 1410 | 25 | 1603 |
| 8 192 | 2820 | 25 | 1603 |
| 16 384 | 5640 | 25 | 1603 |

Sanity check worth repeating for every model: `tput_user × concurrency ≈ tput`
(`25 × 64 = 1,600 ≈ 1,603`). If those two disagree, one of them was downrated and the other
was not.

---

## 2. One graded tier, named `functional`

Each Milestone-0 `BLACKHOLE_GALAXY` device row sets:

```yaml
perf_targets_map:
  functional: 1.0
```

and the template sets `status: FUNCTIONAL`.

That produces exactly one tier, holding the published value verbatim. The default ladder
(`functional 0.10 / complete 0.50 / target 1.0`) is not used: the RFP states one number per
point (Appendix B.2), and three bars in code would send a Partner looking for two they cannot
find in the document.

### Why the one tier is called `functional` and not `target`

Because `ModelStatusTypes.required_target_tiers` matches on tier **name**:

| Status | Enforces |
| ------ | -------- |
| `EXPERIMENTAL` | *nothing* |
| `FUNCTIONAL` | `functional` |
| `COMPLETE` | `functional`, `complete` |
| `TOP_PERF` | `functional`, `complete`, `target` |

A single tier named `target` under `status: FUNCTIONAL` would be computed, reported, and then
**ignored** — the gate would look configured and enforce nothing. The only status that
enforces a tier named `target` is `TOP_PERF`, which asserts near-peak optimisation. Milestone-0
is a first-of-kind bring-up on a compute class nobody has served these models on; that is not
a claim we make.

So the tier is named for the status that enforces it. **The two must always be changed
together** — there is a test asserting
`perf_targets_map.keys() ⊆ status.required_target_tiers` precisely because renaming one alone
fails silently and in the safe-looking direction.

> Do not "tidy" this by renaming the tier to `target`. It reads like a naming inconsistency
> and is load-bearing.

### `status: EXPERIMENTAL` disables more than performance

`evals_enforced` reuses the same signal (`bool(required_target_tiers)`), so an `EXPERIMENTAL`
model has **accuracy failures waived too**. Before #78 all three Milestone-0 models sat at
`EXPERIMENTAL`, meaning a submission could fail every performance target and every eval and
still be accepted. `FUNCTIONAL` restores both.

---

## 3. Tolerance is per point and explicit

```json
{
  "isl": 8192, "osl": 128, "max_concurrency": 64, "num_prompts": 128,
  "targets": { "theoretical": { "ttft_ms": 2820, "tput_user": 25, "tput": 1603 } },
  "tolerance": 0.10
}
```

`tolerance` sits **beside** `targets`, not inside `theoretical`: it is a policy knob, not a
measured metric, and Appendix B.2 lists it as its own column. It defaults to `0.0` when
absent, so entries that do not opt in keep the previous behaviour of requiring the target to
be beaten outright.

Milestone-0 publishes **0.10**. Applied by `llm_module/target_checks.py` exactly as RFP G.2.4
states:

```
ratio = measured / target
TTFT (lower is better)              passes when ratio < 1 + tolerance
tput_user, tput (higher is better)  passes when ratio > 1 - tolerance
```

Passing on tolerance is the weakest qualifying result: the rubric's qualifying value is the
target itself, so such a submission scores **zero** on that line (RFP K.2). The allowance
costs nothing in ranking and absorbs run-to-run variance.

---

## 4. Constraints an authored sweep must satisfy

| Rule | Source |
| ---- | ------ |
| Targets are **per-system**, never per-replica — a Partner's data-parallel choice must not move the bar | [device config](m0-blackhole-galaxy-device-config.md) (§5.4) |
| **≥ 3 distinct input lengths at every graded concurrency**, or the scaling-quality fit is meaningless and `get_llm_configs` fails fast | [scaling-quality coverage](m0-scaling-quality-sweep-coverage.md) (§5.7) |
| **Exactly two concurrency corners**, idle and loaded. Appendix B.5's per-point weights are defined as 25 % / 75 % over two levels, and `report_module.scorecard.point_weights` raises rather than invent a split for three | RFP B.5 |
| A point with no target is reported as ungradable, never as a pass | RFP G.2.5 |
| Exactly one point per model is the headline point; no waiver may be applied to it | RFP B.2, M.5 |

### The two rules above interact, and the KV pool decides whether they can both hold

`get_benchmark_max_concurrency` demotes a point's concurrency when the device KV-token pool
cannot hold that many requests of that shape:

```
concurrency = min(max_tokens_all_users // (isl + osl), model_max_concurrency)
```

So a pool sized only for the shortest inputs quietly splits the loaded corner into a
different level per input length — one input length each, which breaks the three-point rule
*and* produces more than two levels. DeepSeek-V4-Flash hit exactly this: at a pool of
1,056,768 the sweep demoted 32K→32, 64K→16, 128K→8, 256K→4, 512K→2.

**Size the pool for `model_max_concurrency` requests at the longest graded input**, and
check the whole sweep resolves to two levels before authoring targets:

```bash
MODEL_SPECS_ENV=dev ONLY_BENCHMARK_TARGETS=1 python -c "
from collections import defaultdict
from llm_module.benchmark_configs import get_llm_configs
from report_module.scorecard import point_weights
from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes
s = next(s for s in MODEL_SPECS.values()
         if s.device_model_spec.device == DeviceTypes.BLACKHOLE_GALAXY
         and 'YOUR-MODEL' in s.hf_model_repo)
c = get_llm_configs(s, DeviceTypes.BLACKHOLE_GALAXY)
by = defaultdict(list)
for x in c: by[x.max_concurrency].append(x.isl)
print({k: len(v) for k, v in sorted(by.items())})
print(sum(point_weights([(x.max_concurrency, x.isl) for x in c]).values()))
"
```

Two levels, ≥3 input lengths each, weights summing to 1.0.

### A full-context input cannot be swept

`isl + osl` must be ≤ `max_context`. A power-of-two input equal to the context window leaves
no room for output, and `get_benchmark_max_concurrency` responds by returning concurrency 1
rather than rejecting the point — so the mistake surfaces as a mysterious extra concurrency
level, not as an error. The largest usable input is `max_context − osl`.

---

## 5. Authoring checklist

1. Take the peaks from the Appendix B sheet.
2. Apply the downrates: ×0.50 prefill/TTFT, ×0.25 interactivity/decode.
3. Check `tput_user × concurrency ≈ tput`.
4. Write the downrated values under `theoretical`; add `"tolerance": 0.10` beside `targets`.
5. Ensure ≥3 distinct ISLs at each graded concurrency.
6. On the device row: `perf_targets_map: {functional: 1.0}`. On the template:
   `status: FUNCTIONAL`.
7. Resolve the spec and confirm one tier, the right values, and `tolerance == 0.1`:

   ```bash
   MODEL_SPECS_ENV=dev python -c "
   from workflows.model_spec import MODEL_SPECS
   from workflows.workflow_types import DeviceTypes
   s = next(s for s in MODEL_SPECS.values()
            if s.device_model_spec.device == DeviceTypes.BLACKHOLE_GALAXY
            and 'YOUR-MODEL' in s.hf_model_repo)
   print(s.status.name, s.status.required_target_tiers)
   for t in s.device_model_spec.perf_reference:
       print(t.isl, t.max_concurrency, t.targets)
   "
   ```

8. Confirm the gate bites: a point 20 % over target must make
   `acceptance_criteria_check` return **not accepted**.

---

## 6. Known trap when adding a model

**The reference-file key is case-sensitive and derived from the spec's weights.**
`gemma-4-31B-it` is filed in the reference file as `gemma-4-31b-it` (lowercase `b`, matching
the Forge dev spec and `tt-media-server`'s `ModelNames` constant), while the prod spec derives
`gemma-4-31B-it`. Targets filed under the wrong spelling resolve to nothing. There is now a
warning (tenstorrent#4884) but the result is still no targets. Settle the key before
authoring.

`Mistral-Small-3.1-24B-Instruct-2503` also carries a stale `blackhole_galaxy` entry copied
verbatim from its `p300x2` row. It is not a Milestone-0 model and that entry is not a starting
point for one.
