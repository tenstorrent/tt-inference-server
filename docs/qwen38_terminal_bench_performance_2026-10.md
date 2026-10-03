# Qwen3.8 Terminal-Bench performance investigation and validated configuration

## Status and scope

This report records the October 1–2, 2026 investigation of Qwen3.8-27B
Terminal-Bench 2.1 performance on one P300X2. It covers the original five-task
baseline, three optimized five-task confirmations, the ten-task scale-up,
parallelization experiments, output-length recovery, persistent TT-Metal cache
validation, rejected experiments, infrastructure-invalid attempts, token and
time cost, and the final recommended configuration.

The validated implementation is on:

- branch: `mvasiljevic/qwen38-terminal-perf-production-validated-20261002`
- commit: `fd4365d674559551b50f95848e812ae8295f41e2`
- model: `Qwen/Qwen3.8-27B`
- implementation: `qwen38-27b-qb2`
- device: P300X2, four Blackhole chips
- image: `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.23.0-6a30791d865b8aaa9864f69fe4cdddf88857c38f-1d87a00-110289048924`
- tt-metal/source pin encoded by the image: `6a30791d865b8aaa9864f69fe4cdddf88857c38f`
- vLLM plugin pin encoded by the image: `1d87a00e7d91ec246582d07865ec4f8b0a8fb25c`
- Harbor pin: `1da0bfd8c71cadbff17413fac984b8e391d2afc2`
- fixed QEMU task commit: `a355fc6aaeaf62ba94b6cab023e179c7e440c651`

The branch is an evaluation/performance candidate, not a general release merge
as-is. Its Qwen3.8 eval configuration intentionally isolates Terminal-Bench and
removes the GPQA and SWE tasks from that model entry. A production PR should
split reusable harness/serving changes from cohort-specific configuration.

### Release validation application (October 3, 2026)

The derived branch
`mvasiljevic/qwen38-release-optimized-evals-20261003` restores the release
evaluation suites and applies the safe parts of the Terminal optimization to
them. The executable configuration is pinned at commit
`1207774e430688d021a5d7b5f89419e99c5bc0b0` and contains:

- 10 GPQA questions at concurrency five, temperature 1.0, medium reasoning,
  and a 16K output cap;
- the validated 10-task Terminal-Bench dynamic queue at concurrency five;
- 5 SWE-bench Verified tasks at concurrency five, temperature 1.0, medium
  reasoning, a 16K output cap, and server-metrics collection;
- the same exact model image, tt-metal/vLLM pins, Harbor compact-recovery pin,
  persistent TT-Metal cache, fixed QEMU task, and corrected aggregation path.

The release configuration and routing passed 227 focused tests. Release CI
[run 37125582459](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37125582459)
was dispatched in CI mode at 2026-10-03 13:15 UTC. It runs the complete
benchmark matrix plus the bounded 10 GPQA / 10 Terminal / 5 SWE cohorts. The
dispatch uses the exact executable commit above, rather than a moving branch;
its final benchmark, accuracy, time, and token results should be added here
when the workflow completes.

## Executive result

The device was not the primary cause of the six-hour Terminal-Bench runs.
Qwen3.8 sustains approximately 37–40 output tokens/s at concurrency one and
approximately 110–124 aggregate output tokens/s at a fully occupied concurrency
of five. The dominant costs were high-temperature trajectory variance, very
long generated responses, repeated full-history prefill, tool execution, and
the single-task tail after other tasks completed.

The accepted changes preserved high-temperature sampling:

- temperature `1.0`
- top-p `0.95`
- top-k `20`

They changed default reasoning from `xhigh` to `medium`, reduced the normal
per-turn output limit from 80K to 16K tokens, retained up to 100 productive
turns, dynamically backfilled five execution slots, persisted the TT-Metal
program cache, added request/server telemetry, fixed report aggregation, used
the corrected QEMU verifier, reused an exact prebuilt image, and added bounded
compact recovery for output-length truncation.

Three comparable optimized five-task runs all scored 5/5. Their Harbor wall
times were 59m57s, 3h27m51s, and 4h14m16s. The median was 3h27m51s, 42.3% below
the six-hour baseline. Mean prompt-token volume fell approximately 83.0%, mean
generated-token volume fell approximately 66.1%, and request count fell about
59%.

The first dynamic ten-task run scored 10/10 in 1h27m43s. The final combined
production-branch validation scored 9/10 in 5h57m44s: FEAL passed after a
pathological truncation sequence, while `password-recovery` failed semantically.
The final run confirms that the mechanisms work, but also confirms that a
high-temperature run has no narrow wall-time guarantee.

## Measurement boundaries

The report uses four different time boundaries and names them explicitly:

- **Harbor wall**: elapsed time from Harbor trial scheduling until all selected
  trials finish. This is the best measure of evaluation makespan.
- **Run-tests job**: server setup, Harbor, reporting, and cleanup on the device
  runner.
- **Workflow end-to-end**: image build and runner wait in addition to the test
  job. This can greatly exceed model/eval time.
- **Device-hours**: sum of Harbor wall time across simultaneously allocated
  devices. This matters when comparing one dynamic queue with static sharding.

Token counts are cumulative API/server counts, not unique conversation tokens.
Every agent turn resubmits a growing history because prefix caching is disabled.
For runs with truncated responses, server-generated totals exceed recorded
trajectory totals unless the length-recovery accounting patch is present.

## Original five-task baseline

The principal baseline is
[run 36837684278](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36837684278).

| Quantity | Baseline |
|---|---:|
| Raw score | 3/5 |
| Harbor wall | 6h00m |
| Run-tests job | 6h11m08s |
| Workflow end-to-end | 8h42m06s |
| Image build | 1h02m58s |
| Post-build/device-runner gap | about 1h27m43s |
| Model requests | 320 |
| Cumulative prompt tokens | about 15.27M |
| Server-generated tokens | about 1.011M |
| Server busy interval | 21,420 of 21,640 seconds |
| Timed-out trials | 2 |

This baseline is different from the earlier direct Shield five-task run
`36834331904`, which completed in 3h26m and scored 3/5. The QB2 baseline above
is the controlled six-hour case used for optimization comparisons because it
contains the extreme trajectory/token behavior that motivated the work.

The baseline result was not evidence of an idle or stalled device. Pure-decode
aggregate throughput rose from about 33 tok/s at C1 to about 109.5 tok/s at C5,
and the server was busy for almost the entire Harbor interval. The evaluation
was slow because the agent generated and re-prefilled a very large trajectory.

## Why QEMU failed before and why it can still fail

The original `qemu-startup` verifier installed dependencies from obsolete
Debian Bullseye repositories. After the security index expired, package URLs
returned 404 and left `curl`/`uvx` unavailable. The model could reach the Alpine
login prompt and still receive reward zero before pytest ran.

The fixed task commit
`a355fc6aaeaf62ba94b6cab023e179c7e440c651` removes that infrastructure false
negative by using preinstalled tools and a pinned verifier setup.

That fix does not guarantee task correctness. In
[run 36985204709](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36985204709),
the fixed verifier connected successfully, but the high-temperature agent had
deliberately configured root password `alpine`; the verifier correctly expected
passwordless root and reported `Login incorrect`. That was a genuine semantic
failure, not recurrence of the obsolete-APT bug.

The report parser also previously selected only Harbor's one-task `adhoc` group
for the Git-injected QEMU task. A raw 3/5 could therefore be displayed as 1/1,
100%. The parser now computes a trial-count-weighted aggregate across every
adhoc and registry group, including pass@k and resolved counts.

## Accepted implementation changes

### Agent budget and reasoning

The tokenizer template defaulted Qwen3.8 to `reasoning_effort=xhigh`, asking for
exhaustive reasoning on every turn. The accepted configuration keeps thinking
enabled and temperature high but uses:

```text
temperature = 1.0
top_p = 0.95
top_k = 20
reasoning_effort = medium
max_output_tokens = 16,384
max_turns = 100
agent_timeout = 6 hours per task
```

The output cap addresses single replies that can otherwise consume tens of
thousands of tokens. The turn limit remains 100 because 76 was too tight:

- CompCert once passed on turn 76.
- In controlled shard A, a different CompCert trajectory reached turn 76 while
  still actively repairing/building and failed only because the loop stopped.
- In a later ten-task run, Caffe reached turn 76 while a productive 500-step
  training job was only around iteration 200.

The evidence supports a conservative no-progress detector in the future, not a
lower global turn cap.

### Dynamic C5 scheduling

One Harbor queue contains all tasks with `n_concurrent_trials=5`. Harbor starts
five tasks and backfills a slot immediately when one finishes. For ten tasks,
this avoids a rigid second wave and prevents the shorter half of the workload
from waiting for the slowest task in wave one.

The ten-task cohort was:

1. `break-filter-js-from-html`
2. `cobol-modernization`
3. `compile-compcert`
4. `feal-differential-cryptanalysis`
5. `qemu-startup`
6. `caffe-cifar-10`
7. `password-recovery`
8. `portfolio-optimization`
9. `hf-model-inference`
10. `financial-document-processor`

The added tasks cover ML build/training, password forensics, native
optimization, model service setup, and document processing.

### Exact image reuse

Changes to cohort selection, agent limits, prompt settings, parsing, and Harbor
do not require rebuilding the model-server image. The campaign initially paid
about 63 minutes to build an image, then reused the exact image above for all
comparable runs. This removes build variance and approximately one hour of CI
latency per harness-only experiment.

### Persistent TT-Metal program cache

The launcher previously mounted `TT_CACHE_PATH`, but TT-Metal still wrote
compiled programs to ephemeral `~/.cache/tt-metal-cache`. The branch now sets:

```text
TT_METAL_CACHE=<persistent model cache>/tt_metal_cache
```

for Docker, local launch, and server fallback, while respecting an explicit
override.

Observed restart comparison on a previously populated physical runner:

| State | Server startup | BRISC builds |
|---|---:|---:|
| Cold cache | about 7m22s | 1,224 |
| Warm persistent cache | about 5m06s | 78 |
| Improvement | about 2m16s, 30.8% | 93.6% fewer builds |

This optimization is runner/volume-local. The first run on a new runner remains
cold and populates that runner's cache.

### Server telemetry

While Harbor runs, a low-overhead sampler records selected vLLM Prometheus
metrics every 15 seconds to `server_metrics.jsonl`. It includes:

- running, waiting, and swapped requests;
- preemptions and KV-cache utilization;
- prefix-cache queries/hits;
- prompt and generation tokens;
- request success and requested/actual token counts;
- TTFT, TPOT, inter-token latency, and end-to-end latency;
- queue, prefill, decode, and inference time;
- occupancy and iteration token counters.

Metric collection is diagnostic-only: sampling failures are recorded and never
invalidate an eval.

### Compact recovery from output truncation

Before this change, an invalid response ending at the 16K limit was discarded
and retried with another full 16K allowance. Some tasks repeatedly consumed the
entire allowance.

The carried Harbor patch now:

1. records truncated usage accurately;
2. salvages complete JSON/XML responses even when `finish_reason=length`;
3. otherwise retries with an explicit compact prompt and a 4K output cap;
4. retains temperature 1, top-p 0.95, top-k 20, and medium reasoning;
5. limits compact recursion to eight attempts.

The final validation's FEAL trajectory had 48 length finishes: 45 compact 4K
retries and three recursion-guard activations. Compact recovery avoided exactly
516,096 additional output tokens relative to allowing those 42 repeated compact
retries to consume 16K each. At the observed C1 tail rate this is approximately
3h49 of decode time and likely converted an otherwise timed-out FEAL trial into
a pass.

## Five-task confirmation

The three comparable optimized runs all used one P300X2, dynamic C5,
temperature 1.0, top-p 0.95, top-k 20, medium reasoning, a 16K normal turn cap,
the fixed QEMU task, and the same known-good image/source pins.

| Run | Score | Harbor wall | Job wall | Requests | Server prompt tokens | Server output tokens | Dominant tail |
|---|---:|---:|---:|---:|---:|---:|---|
| [36943022378](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36943022378) | 5/5 | 59m57s | 1h08m53s | 96 | about 1.267M | 163,326 | break-filter, 59m57s |
| [36961379927](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36961379927) | 5/5 | 3h27m51s | 3h36m42s | 143 | about 2.634M | about 379.8K | FEAL, 3h27m51s |
| [36941756400](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36941756400) | 5/5 | 4h14m16s | 4h24m49s | 152 | about 3.894M | about 485.9K | FEAL, 4h14m16s |

Summary against the six-hour baseline:

| Quantity | Baseline | Optimized three-run result | Change |
|---|---:|---:|---:|
| Score | 3/5 | 5/5 in all three runs | +2 tasks in these samples |
| Harbor wall | 6h00m | median 3h27m51s | −2h32m09s, −42.3% |
| Harbor wall | 6h00m | mean 2h54m01s | −3h05m59s, −51.7% |
| Prompt tokens | about 15.27M | mean about 2.598M | −83.0% |
| Output tokens | about 1.011M | mean about 343K | −66.1% |
| Requests | 320 | mean about 130 | about −59% |

The range from one hour to more than four hours is material. These runs prove a
lower median and much lower token volume, not deterministic runtime at
temperature 1.

## Ten-task scale-up

### Successful dynamic queue

[Run 36964059545](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36964059545)
scored 10/10 with zero errors.

| Quantity | Result |
|---|---:|
| Harbor wall | 1h27m43s |
| Job wall including cold startup/setup | 1h38m15s |
| Agent turns | 226 |
| Agent prompt/output | 3.706M / 237,465 |
| Server requests | 227 |
| Server prompt/output | 3.708M / 253,849 |
| Discarded full-length responses | 1 × 16,384 tokens |
| Mean TTFT | 10.35s |
| Token-weighted TSU | 64.62ms/token, 15.47 tok/s/user |
| Full-wall aggregate output | 48.2 tok/s |
| Mean running occupancy | 3.18 |
| Busy samples | 97.7% |
| Total queue time | 1.292s, 5.7ms/request |
| Preemptions | 0 |

Per-task completion times:

| Task | Time | Result |
|---|---:|---:|
| `qemu-startup` | 8m04s | Pass |
| `portfolio-optimization` | 9m43s | Pass |
| `hf-model-inference` | 11m54s | Pass |
| `financial-document-processor` | 25m13s | Pass |
| `password-recovery` | 34m18s | Pass |
| `break-filter-js-from-html` | 41m30s | Pass |
| `cobol-modernization` | 45m59s | Pass |
| `caffe-cifar-10` | 47m36s | Pass |
| `feal-differential-cryptanalysis` | 1h17m20s | Pass |
| `compile-compcert` | 1h26m53s | Pass |

Nine tasks were complete by about 1h27, leaving less than a one-minute final
tail. Dynamic backfill maintained useful occupancy despite unequal task lengths.

### Final combined production-branch validation

[Run 37005655537](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37005655537)
used the final `max_turns=100` plus bounded compact-recovery configuration.

| Quantity | Result |
|---|---:|
| Score | 9/10 |
| Harbor wall | 5h57m44s |
| Run-tests job | 6h04m18s |
| Server requests | 301 |
| Server prompt tokens | about 6.977M |
| Server output tokens | 839,053 |
| Mean TTFT | 6.95s |
| Token-weighted TSU | 41.87ms/token, 23.88 tok/s/user |
| Request-average TPOT | 47.90ms/token, 20.88 tok/s/user |
| Full-wall aggregate output | 39.09 tok/s |
| Mean occupancy | 1.65 |
| Queue time | 4.37ms/request |
| Preemptions | 0 |

QEMU and eight registry tasks passed. `password-recovery` failed semantically.
FEAL passed at 5h51m41s and dominated the wall time. This run validates the
combined production mechanisms but demonstrates the residual high-temperature
tail risk.

## Parallelization experiment

The controlled comparison used the same image, model pins, temperature, agent
settings, and total ten-task cohort.

### One dynamic queue on one device

- one P300X2;
- C5 with immediate backfill;
- 10/10 in 1h27m43s;
- 1.46 device-hours.

### Static 5+5 split on two devices

- shard A: CompCert, COBOL, password, portfolio, QEMU;
- shard B: FEAL, break-filter, Caffe, financial-document, hf-model;
- both shards C5, started concurrently when runners were available.

Results:

| Run | Score | Harbor wall | Server prompt/output | Observation |
|---|---:|---:|---:|---|
| [Shard A 36973628141](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36973628141) | 4/5 | 1h12m22s | 3.183M / 121.7K | CompCert cut off at turn 76 |
| [Shard B 36973627932](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36973627932) | 5/5 | 3h53m30s | 1.925M / 460.9K | FEAL occupied the long tail |
| [Corrected A 36985204709](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36985204709) | 4/5 | 54m41s | 2.033M / 133.2K | CompCert passed; QEMU semantic failure |

For the original simultaneous shards, the effective makespan was 3h53m30s and
device consumption was 5.10 device-hours. That is 2.66× slower in makespan and
3.49× more device time than the favorable one-device dynamic queue. Static
sharding could not mitigate the sampled FEAL serial tail; it merely stranded the
other device.

Recommendation: use one dynamic C5 queue per P300X2. If multiple devices are
available, use a shared dynamic work queue or work stealing rather than fixed
5+5 partitions. A single stochastic A/B is not proof that two devices are
intrinsically slower, but it is direct evidence that static partitioning is
fragile and can cost more device time without improving makespan.

## Device performance and overhead diagnosis

Representative instrumented results:

| Run | Mean TTFT | Per-user TPOT/TSU | Aggregate output | Mean occupancy | Queueing |
|---|---:|---:|---:|---:|---:|
| Five-task fast | 12.66s | 52.2ms, 19.15 tok/s | 45.4 tok/s | high early C5 | 0.003s total |
| Five-task 4h14 | 12.21s | token-weighted 38.18ms, 26.19 tok/s | tail dominated | 1.22 | 8.6ms total |
| Five-task 3h28 | 10.94s | token-weighted 38.99ms, 25.64 tok/s | 37.56 tok/s | 1.20 | 6.8ms total |
| Ten-task 10/10 | 10.35s | token-weighted 64.62ms, 15.47 tok/s | 48.2 tok/s | 3.18 | 5.7ms/request |
| Final 9/10 | 6.95s | token-weighted 41.87ms, 23.88 tok/s | 39.09 tok/s | 1.65 | 4.37ms/request |

Conclusions:

- C1 decode is approximately 37–40 tok/s, close to the measured device ceiling.
- A fully occupied C5 can reach about 110–124 aggregate tok/s.
- Per-user speed drops under concurrency, as expected, but total throughput and
  makespan improve while several tasks remain runnable.
- Queueing is negligible and there were no preemptions.
- Real full-wall aggregate throughput is lower because agents execute tools,
  submit large prefills, finish at unequal times, and sometimes leave only one
  long task.
- The remaining large cost is model output plus repeated full-history prefill,
  not an unexplained TT device stall.

Synthetic C1 benchmark context for the same implementation family:

| Input length | TTFT | User output rate |
|---:|---:|---:|
| 128 | 69ms | 40.48 tok/s |
| 4K | 633ms | 39.78 tok/s |
| 16K | 2.605s | 39.01 tok/s |
| 32K | 5.505s | 37.99 tok/s |
| 65K | 12.255s | 36.18 tok/s |
| 131K | 29.565s | 33.01 tok/s |

The agentic mean TTFT of roughly 7–13 seconds is compatible with large, growing,
uncached prompts and changing prefill shapes. It should not be compared only to
the 128-token synthetic point.

## Length-recovery validation and rejected alternatives

### First ten-task recovery validation

[Run 36982427496](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36982427496)
used the recovery patch but accidentally retained `max_turns=76`.

- raw result: 8/10;
- Harbor wall: 3h35m11s;
- exact prompt/output accounting: 7,597,395 / 581,410;
- 291 server requests: 286 normal episodes plus five compact recoveries;
- five `finish_reason=length` events and five 4K recovery requests;
- Caffe failed because turn 76 interrupted productive training;
- password-recovery produced a genuine incorrect reconstruction.

The patch worked: one recovery chain used 16,384 + three 4,096-token retries +
2,453 tokens, instead of giving each retry another 16K allowance. It avoided
36,864 decode tokens in that chain.

### Unsupported `minimal` recovery effort

[Run 37053798687](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37053798687)
attempted `reasoning_effort=minimal` only on compact recovery. The server accepts
only `xhigh`, `medium`, and `low`, so the request returned `BadRequest`. This run
is infrastructure/config invalid and must not be scored as a model failure.

### Supported `low` recovery effort

[Run 37063640772](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37063640772)
used supported `low` effort for recovery while retaining high-temperature
sampling. FEAL completed in 17m28s but scored 0/1 after choosing the invalid
shortcut `return feal.key[5]`. It still produced a 16K + three 4K truncation
chain, so lower recovery effort did not remove the truncation pathology and may
have reduced solution quality. This change was rejected.

The production branch therefore keeps `medium` reasoning on every request and
uses the validated eight-attempt compact guard.

## Experiment ledger, including invalid and canceled attempts

All links below are retained so failed setup iterations are not rediscovered or
misinterpreted later.

| Run | Outcome | Reason / evidence |
|---|---|---|
| [36837684278](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36837684278) | Valid baseline, 3/5 | Six-hour Harbor run; 15.27M prompt/1.011M output |
| [36919397799](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36919397799) | Invalid before device | Checkout did not accept a shortened inference-server ref |
| [36919689211](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36919689211) | Invalid before device | Incorrect catalog implementation spelling |
| [36919903678](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36919903678) | Invalid, zero eval requests | Stale TTIS 0.21 image lacked `TTQwen38ForCausalLM`; two health timeouts |
| [36920663066](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36920663066) | Canceled | Duplicate cap-only image build; harness change did not need a new image |
| [36921624184](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36921624184) | Canceled | Duplicate medium-reasoning image build |
| [36927489337](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36927489337) | Invalid, zero eval requests | Reused the same stale 0.21 image |
| [36928146793](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36928146793) | Invalid, zero eval requests | Same stale-image failure in the cap-only arm |
| [36937908092](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36937908092) | Canceled | Queued replicate canceled after stale-image diagnosis |
| [36943022378](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36943022378) | Valid, 5/5 | Fast five-task confirmation |
| [36961379927](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36961379927) | Valid, 5/5 | Five-task confirmation with 3h28 FEAL tail |
| [36941756400](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36941756400) | Valid, 5/5 | Five-task confirmation with 4h14 FEAL tail |
| [36964059545](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36964059545) | Valid, 10/10 | Dynamic C5 ten-task scale-up in 1h27m43s |
| [36973628141](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36973628141) | Valid execution, 4/5 | Static shard A; CompCert cut off by 76 turns |
| [36973627932](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36973627932) | Valid, 5/5 | Static shard B; 3h53 FEAL tail |
| [36985204709](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36985204709) | Valid execution, 4/5 | Max-turn correction; QEMU semantic password failure |
| [36982427496](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36982427496) | Valid execution, 8/10 | Recovery patch worked; inherited 76-turn cap caused Caffe failure |
| [37005655537](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37005655537) | Final combined validation, 9/10 | Max 100 + bounded recovery; FEAL passed, password failed semantically |
| [37053798687](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37053798687) | Invalid configuration | Unsupported `reasoning_effort=minimal` |
| [37063640772](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37063640772) | Valid targeted task, 0/1 | Supported `low` recovery effort degraded FEAL solution; rejected |

## Time and token cost of the improvement campaign

### Savings achieved on the five-task cohort

Compared with the six-hour baseline, the three optimized confirmations used:

- median Harbor time: 3h27m51s, saving 2h32m09s per five-task run;
- mean Harbor time: 2h54m01s, saving 3h05m59s per five-task run;
- mean prompt tokens: about 2.598M instead of 15.27M, saving about
  12.672M cumulative prompt tokens per run;
- mean generated tokens: about 343K instead of 1.011M, saving about 668K
  generated tokens per run;
- mean requests: about 130 instead of 320, saving about 190 requests per run.

These are sample means/medians at temperature 1, not deterministic service-level
guarantees.

### Measurement cost

For the eleven valid device evals with known Harbor time in this report—the
baseline, three five-task confirmations, dynamic ten-task run, two original
shards, corrected shard A, first length-recovery run, final combined run, and
targeted supported-low FEAL run—the summed Harbor allocation was approximately
**32.0 device-hours**.

Exact or reported server token totals are available for ten of those runs. They
sum to approximately:

- **48.49M cumulative prompt tokens**;
- **4.43M generated tokens**.

The targeted supported-low FEAL run is included in the 32.0 hours but excluded
from those token sums because its full server token totals were not retained in
the investigation notes. The totals also exclude infrastructure-invalid runs
that produced zero eval requests, canceled builds, the earlier isolated QEMU
A/B, and engineering/test execution outside Harbor. They should therefore be
read as a documented lower bound on campaign cost, not an accounting/billing
total.

The campaign spanned from the first baseline workflow at 2026-10-01 08:39 UTC
through the last targeted validation at 2026-10-02 21:19 UTC, about 36h40m of
calendar time. Many device runs overlapped, so calendar time, summed Harbor
device-hours, and engineer/agent time are not interchangeable.

## Code and test coverage

The branch adds or changes:

- `llm_module/agentic/server_metrics.py`: 15-second Prometheus sampler;
- `llm_module/agentic/harbor.py`: task overrides and metrics lifecycle;
- `llm_module/parsers/agentic.py`: weighted multi-group Harbor aggregation;
- `reference_config/evals/eval_config.py`: ten-task cohort and agent settings;
- `vllm-tt-metal/src/run_vllm_api_server.py`: persistent TT-Metal cache fallback;
- Docker/local launcher cache propagation;
- `workflows/workflow_venvs.py`: exact Harbor pin and deterministic patching;
- `workflows/patches/harbor/0001-recover-compactly-from-length-truncation.patch`:
  bounded compact recovery and exact usage accounting;
- focused tests for metrics, parser aggregation, task overrides, patch
  application, cache propagation, agentic dispatch, and Qwen3.8 configuration.

Staged validation during development included 64–149 focused TTIS tests,
40 focused Harbor tests plus patch-application tests, and 11 final compact
recovery/guard tests. CI evaluation evidence is authoritative for end-to-end
behavior; the run ledger above distinguishes valid model outcomes from
infrastructure/configuration failures.

## Recommended production configuration

Use:

- one dynamic C5 Harbor queue per P300X2;
- temperature 1.0, top-p 0.95, top-k 20;
- `reasoning_effort=medium`;
- 16K normal output limit;
- `max_turns=100`;
- six-hour per-task timeout;
- compact 4K recovery with guard 8;
- exact pinned Harbor and QEMU task revisions;
- persistent `TT_METAL_CACHE` under the model cache;
- prebuilt exact model-server images for harness-only changes;
- 15-second server telemetry;
- raw weighted aggregation across all Harbor groups.

Do not use:

- `max_turns=76` as a global limit;
- fixed 5+5 static task shards without work stealing;
- recovery-only `reasoning_effort=low`;
- unsupported `reasoning_effort=minimal`;
- the stale TTIS 0.21 Qwen3.8 image;
- the old live-Bullseye-APT QEMU verifier;
- the first Harbor group as the whole evaluation score.

## Remaining work and risks

1. **High-temperature long tails remain.** FEAL varied from under one hour to
   nearly six hours and can dominate the complete cohort.
2. **Semantic failures remain possible.** QEMU can be configured with the wrong
   login semantics; password-recovery failed independently in multiple runs.
3. **A no-progress detector is preferable to a lower turn cap.** It must avoid
   terminating productive compile/training/repair loops.
4. **Repeated full-history prefill remains structural overhead.** Prefix cache
   hit rate is zero in these runs.
5. **Generic prefix caching is unsafe for Qwen3.8.** Its hybrid GDN layers need
   recurrent state and convolution state in addition to KV state. A correct
   continuation cache must snapshot GDN recurrent state, convolution state, KV
   state, and session position together and pass numerical validation.
6. **Arbitrary prefill-tail shapes still affect TTFT.** Correct masked shape
   bucketing and stable trace reuse remain possible follow-up work, but padding
   must not advance or corrupt GDN state.
7. **Multi-device scaling should use dynamic shared scheduling.** Static shards
   are too sensitive to sampled task tails.
8. **The branch must be separated before merge.** Reusable harness/telemetry,
   Harbor recovery, cache persistence, parser correctness, and experiment-only
   Qwen cohort changes should be reviewed independently.

## Final conclusion

The optimization is real and CI-confirmed, but it is probabilistic rather than
a fixed speedup. On the five-task cohort it reduced median Harbor wall time by
42%, cumulative prompt tokens by 83%, and output tokens by 66%, while all three
optimized confirmations scored 5/5. A dynamic ten-task run achieved 10/10 in
1h27m43s. The final combined production-branch validation confirmed cache
persistence, metrics, max-turn headroom, correct aggregation, QEMU override, and
bounded length recovery, but took 5h57m44s and scored 9/10 because high-temperature
trajectory variance and semantic task failures remain.

The correct operational interpretation is: device decode is near the current
implementation ceiling; dynamic C5 is the best tested scheduler; most avoidable
cost came from excessive reasoning/output, repeated context ingestion, cold
program compilation, and discarded length-capped responses. The recommended
branch removes or bounds those costs without reducing temperature.
