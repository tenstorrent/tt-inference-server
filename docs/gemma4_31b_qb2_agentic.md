# Gemma 4 31B QB2 agentic evaluation and serving experiments

Updated 2026-10-09 UTC. This file tracks the dedicated `gemma4-31b-qb2`
implementation on the four-chip P300X2 QuietBox. Results from different model
commits, task lists, or agent policies are kept separate.

## Existing evidence

- The [September 21 baseline](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/35667712636)
  used Metal `dcfa5e2da087432c337a70ac92f8eeeb3bb97f55` and the fixed
  first five Terminal-Bench 2.0 and SWE-bench Verified cases. It scored **1/5**
  on each. Terminal ran 7.84 hours (5,646.8 seconds per case), with 4.61
  million input and 557 thousand
  output tokens across the five cases. SWE ran 2.00 hours. These are
  historical outcomes, not scores for current Metal main. That server logged
  whole-prompt prefill (`enable_chunked_prefill=False`, batched-token budget
  262144); the current main catalog enables chunked prefill at 8192. This
  serving-policy change is another reason to obtain a new C1 baseline.
- The five Terminal case clocks summed to 28,230 seconds. Their recorded model
  API requests summed to 16,248 seconds (**57.6%**); the 11,982-second
  remainder includes shell work, tools, setup and verification. CompCert and
  HTML filtering accounted for much of that remainder. Overlapping another
  agent's inference with these activities is a concrete reason to test C2,
  even if per-user token speed falls slightly.
- For 127 Terminal requests whose trajectory token counts align one-to-one
  with API timings, a linear fit gives **5.30 seconds/request + 0.147 seconds
  per 1,000 prompt tokens + 26.28 seconds per 1,000 output tokens**
  (R²=0.996). CompCert's 53 API timings versus 50 trajectory steps are
  excluded rather than guessed into alignment. The 127 matched requests used
  2.83 million prompt and 438 thousand output tokens, with 12,598 API seconds.
  This observational fit suggests decode and output length dominate those API
  calls. It is not a device prefill/decode profiler and is specific to the
  older Metal run. The separate September 24 cases give 26.69 seconds/1,000
  output tokens on 37 matched requests, reinforcing that direction.
- CompCert's first two trajectory response gaps were 1,363 and 2,735 seconds,
  while their first two recorded API calls were only 47 and 59 seconds. Those
  two intervals alone contain about 3,992 seconds outside the API. The trace
  does not separate shell execution from environment/setup and agent overhead,
  but it explains why parallel agents could hide a substantial part of this
  case's elapsed time.
- That Terminal server logged 409 model-trace warm/capture cycles. Matched
  warm-to-capture intervals sum to 119.8 seconds (median 0.157 seconds, p95
  0.440 seconds), so trace recapture alone is a small part of its 7.84-hour
  Terminal evaluation. The first warmup was much slower and cold startup must
  be kept separate. Its 3,545 roughly ten-second KV samples reached 28.7%
  maximum and 19.7% p95; no sample showed a waiting request. These samples
  cannot establish a safe smaller pool under higher concurrency. Across the
  combined Terminal and SWE server lifetime, 75.9% of the ten-second samples
  had positive generation throughput, 9.4% positive prompt throughput, and
  every prompt-positive sample was also generation-positive. Median positive
  generation throughput was 37.4 tokens/s. These are activity indicators over
  sampling windows, not exclusive prefill/decode device-time fractions.
- The separate [September 24 exploratory cases](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/35986956766)
  scored 3/5 Terminal and 1/5 SWE on **different task IDs**. Terminal used
  72,817 output tokens and 3.62 hours; peak sampled KV was 13.4%. The differing
  tasks and trajectories prevent reading its shorter wall time as a serving
  speed improvement over the September 21 fixed-case run.
- The dedicated model's [README](https://github.com/tenstorrent/tt-metal/blob/2c1e1ebdd638886821f35113a5fd0d6335d71608/models/demos/gemma4_31b_qb2/README.md)
  supports batch 1–32. Its measured 128/128 case completed 0.357 requests/s at
  batch 1 and 5.847 requests/s with 32 concurrent requests. At 1024/128,
  batch-32 per-user decode dropped to 21.00 tokens/s from batch-1's 44.27.
  Agentic prompts are much longer, so those burst measurements do not choose
  the best agent concurrency. This dedicated implementation explicitly rejects
  prefix caching, so the 4.61 million repeated agent input tokens cannot be
  accelerated by enabling a server flag alone. The main catalog's separate
  `tt_transformers` implementation has a different cache path and should be
  measured as its own implementation if C2/C4 leaves prefill dominant; the
  historical request fit currently points more strongly to decode. Its
  [Gemma4 README](https://github.com/tenstorrent/tt-metal/blob/2c1e1ebdd638886821f35113a5fd0d6335d71608/models/demos/gemma4/README.md)
  reports 22.68 tokens/s for a warm QB2 31B batch-1 metal demo at 4K input.
  That is a different measurement path from the dedicated server's 44.27
  tokens/s at 1K input, so it is only a reason to prioritize the dedicated
  C2/C4 measurements, not a controlled implementation comparison.
- Granite's [fixed-five timing analysis](https://github.com/tenstorrent/tt-metal/blob/533b81b7bbd/models/autoports/ibm_granite_granite_4_2_30b/doc/agentic_evals/TERMINAL_BENCH_TIME_BREAKDOWN.md)
  measured 79.3% of accumulated agent execution inside model API calls at C2;
  other agent work overlaps another case's inference. Its thinking-off pilot
  produced fewer output tokens and shorter suite time on one stochastic
  trajectory at unchanged task/scorer, while expanded KV reduced sampled
  waiting at C10. Neither effect transfers automatically to Gemma.

| September 21 Terminal case | Reward | Wall min | Model API min | Output tokens |
|---|---:|---:|---:|---:|
| HTML filter | 0 | 171.4 | 102.4 | 212,971 |
| COBOL modernization | 1 | 9.0 | 7.1 | 15,178 |
| CompCert | 0 | 162.1 | 60.8 | 118,969 |
| FEAL | 0 | 63.7 | 54.3 | 120,476 |
| QEMU startup | 0 | 64.3 | 46.2 | 89,380 |

The same run's SWE cases resolved Django 11299 (1/5 overall). Astropy 14096,
Matplotlib 25332, Sympy 13551 and scikit-learn 14629 did not resolve. The
Mini-SWE-Agent trajectories recorded 32–50 API calls per case and a maximum
individual prompt of 26,078 tokens. Extracted output totals ranged from 7,621
to 11,713 tokens per case. Their archived summary lacks per-case wall timings.

## Current-main baseline setup

- Metal main pinned at `2c1e1ebdd638886821f35113a5fd0d6335d71608`.
- The local workstation has a QB2, but no Gemma 4 31B checkpoint in its
  Hugging Face cache and no active Hugging Face login. Host-only budget and
  config checks ran locally; model/device measurements use the CI runners,
  whose checkpoint mounts are already provisioned.
- Dispatch through `tt-agentic-bringup-qb2` wrapper ref
  `mvasiljevic/granite-kv-overlay-ci`, whose manual workflow exposes the
  `agentic` workflow and partition selector. It calls Shield's reusable
  `on-dispatch.yml` at `47e4089ff4833fb5f44a531470f8124217cad388`;
  the optional Granite overlay job is skipped for every Gemma run.
- That Shield pin's test-job allowlist omitted `tb2.0` even though
  inference-server `run.py` accepts it. The
  [one-line Shield branch](https://github.com/tenstorrent/tt-shield/tree/mvasiljevic/gemma-tb2-partition)
  at `71ec7d17192ece0c8b2a864aff2bca3208bd1a98` adds the TB2.0 alias.
  The separate [Gemma QB2 wrapper](https://github.com/tenstorrent/tt-agentic-bringup-qb2/tree/mvasiljevic/gemma-tb2-ci)
  at `0352933a1e7b1f9b6b7d8a55924f7b17df2bb972` pins that Shield commit;
  use this wrapper ref for Terminal 2.0 dispatches. Its later
  [read-only occupied-runner hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963641492)
  landed on `120-qb2-p04t07`, the host that repeatedly failed the ownership
  check due to a stopped container. The hold leaves that container and
  hardware untouched while clean hosts accept the queued Gemma jobs; cancel
  it after those jobs are assigned.
- The inference-server [baseline branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-baseline)
  adds tool-call parsing to the dedicated dev catalog entry and restores the
  original five fixed Terminal-Bench 2.0 and SWE-bench Verified cases. It keeps
  server capacity 1, one concurrent agent, thinking enabled, temperature 1.0,
  top-p 0.95, top-k 20 and the earlier bounded output settings. It retains the
  recorded full-set H100 targets (Terminal 44.94%, SWE 64.80%) for context;
  five-case percentages are discrete diagnostics and the smaller request
  budgets prevent direct full-set H100 comparison. Two published-score fields
  copied from a Qwen entry were cleared; the Gemma H100 reference targets were
  retained.
- The first [Terminal 2.1 dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37954905591)
  and [SWE dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37954917274)
  stopped before running tests because the older TTI tool branch lacked the
  dedicated catalog implementation. The next
  [dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37955025229)
  used a short commit ref the checkout could not fetch. Another
  [dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37955275595)
  used the implementation ID instead of the required implementation **name**.
  These have no model scores.
- The first [Terminal 2.0 dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37955350355)
  built and published Metal-main image
  `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.24.0-2c1e1ebdd638886821f35113a5fd0d6335d71608-c62035d-113904451192`.
  Its test job stopped before model startup because the old Shield pin
  rejected `tb2.0`; it has no score. Shield
  invokes `run.py --dev-mode`, which passes the checked-out branch's model spec
  into Docker and mounts its source/config directories; this permits C1/C2/C4
  catalog experiments on the same Metal-main image without silently reusing
  the image's baked C1 catalog. The Metal KV-hook candidate needs a new image.

The same published image is reused by [SWE C1](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962784599)
(inference-server `fb5b77a4e26af79c455af2893babf1a9928bfc08`),
[SWE C2](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962800664)
(`ebb2cf2870b86df3edd5b6636a1503f1e9426c1d`), and
[Terminal C4 attempt](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962796381)
(`7e6ac75afa416ae28274d8a663809a039ddc3c44`) also stopped at the old
TB2.0 allowlist. The first
[Terminal C2 dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962790734)
stopped before model startup: its runner contained a stopped, unowned Docker
container. No container was removed. The
[Terminal C2 retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963042451)
was cancelled after the Shield TB2.0 allowlist problem was identified. The
[corrected Terminal C1](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963232263),
[C2](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963238207),
and [C4](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963243142)
use the new wrapper, identical Metal-main image and fixed five cases. Their
inference-server SHAs are respectively `fb5b77a4e26af79c455af2893babf1a9928bfc08`,
`ebb2cf2870b86df3edd5b6636a1503f1e9426c1d`, and
`7e6ac75afa416ae28274d8a663809a039ddc3c44`.
The first corrected C2 and C4 runs also landed on the occupied host and
stopped before model work. Their
[C2 retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963842946)
and [C4 retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963848727)
are queued behind active clean-host jobs, with the occupied host held aside.
The [C2 throughput sweep](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962976771)
and [C4 throughput sweep](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962982690)
reuse that image as separate benchmark jobs. Compare matching input/output
lengths and both per-user latency and aggregate throughput; short synthetic
bursts alone do not establish the fastest agentic suite wall time. The first
C4 benchmark attempt stopped before testing on the same occupied host as the
failed C2 Terminal attempt. The
[C4 benchmark retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963865248)
is queued for a clean runner.

## Candidates and acceptance

| Variant | Decode rows | Agent trials | Context | Thinking | Metal |
|---|---:|---:|---:|---|---|
| Current-main baseline | 1 | 1 | 256K | On | `2c1e1ebd` |
| C2 | 2 | 2 | 256K | On | `2c1e1ebd` |
| C4 | 4 | 4 | 256K | On | `2c1e1ebd` |
| C2/192K KV | 2 | 2 | 192K | On | `6b70d126` |
| C2 thinking-off pilot | 2 | 2 | 256K | Off | `2c1e1ebd` |

1. [C2 catalog branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2):
   two physical decode rows and two concurrent agents, with unchanged fixed
   tasks, request policy, timeout, scorer and 256K context. This is the first
   batching candidate; test the server startup, live API/tool calls, KV
   occupancy, task completion and wall time before increasing agent count.
   A [C4 branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c4)
   is prepared for an exploratory performance comparison; do not use its
   four-agent eval result as a replacement for the C1 baseline without a
   complete fixed-case denominator and resource/score review. The four largest
   archived Terminal request lengths from different cases sum to about 206K,
   below C4's 256K full-attention logical pool, but four requests at their
   configured 80K ceilings would exceed it. Watch live waiting/preemption
   and agent deadlines rather than inferring safety from the old trajectories.
2. [Context-sized KV hook](https://github.com/tenstorrent/tt-metal/tree/mvasiljevic/gemma4-31b-agentic-kv)
   and [C2/192K catalog](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-kv192):
   keep the physical batch at two and size the base full-attention pool from
   `max_model_len=196608`. The previous full-context 262144 setting returns
   the same pool as main. The hook requests 355,584 tokens at C2/256K and
   290,048 at C2/192K, including sliding-window and in-flight headroom, before
   the plugin's additional per-request output pages. The 64K/16K Terminal
   request budgets fit below 192K. The inference-server benchmark selector
   still caps **combined logical** in-flight tokens at the 192K context by
   default; the hook's extra 93K covers five sliding groups and prefill
   headroom, not another 93K of full-attention logical context. Thus a C2
   benchmark point requires each simultaneous request to be around 96K or
   shorter, and the two 80K Terminal request ceilings fit. The archived SWE
   cases peaked at 26,078 prompt tokens, leaving room for those trajectories,
   though current-main stochastic paths can grow differently. The largest
   archived Terminal request lengths (prompt + completion) were 70,989 for
   CompCert and 62,530 for QEMU. Their sum, 133,519, already exceeds a 128K
   shared full-attention pool, so 192K is the smallest prepared 64K-step context
   with headroom for that observed pair; it is not proven optimal until live
   concurrent KV telemetry confirms it.
   The candidate requires device proof: startup/allocation, high page IDs,
   long prefill/decode, 1- and 2-request quality checks, and an agentic run
   without KV preemption before it can be selected. One host-side budget test
   and Ruff/pre-commit pass; no device claim is made yet. Its
   [benchmark dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37956805379)
   built a separate image from Metal `6b70d1267a5a16410d5aa5c7ec8e851b7ad47f94`
   and inference-server `84c126298977df6f3d90ff2ea1313c7a255ec4c8`:
   `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.24.0-6b70d1267a5a16410d5aa5c7ec8e851b7ad47f94-c62035d-113909389966`.
   Hardware benchmark results are still pending. The
   [fixed-five Terminal C2/192K job](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37964456985)
   is queued on that image with inference-server
   `6682a78c8a1009934838f58b3070927973ec7f4f`; cancel it before model
   work if the preceding KV benchmark fails startup or correctness.
3. [C2 thinking-off pilot](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-thinkoff):
   changes only the server's `enable_thinking` default. Granite's fixed-case
   pilot reduced output volume markedly, and Gemma's archived latency fit
   makes output reduction promising. Run the same five cases and compare
   rewards, token counts, per-case clocks and API calls; treat it as a distinct
   quality policy, not a reference-equivalent speed result. Do not select it
   solely on a faster wall clock or a five-case score tie. Its
   [fixed-five dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37965857157)
   is queued behind the thinking-on comparison and uses inference-server
   `d55d5fb55308109b303a85d14ea2832c679b9888` on the Metal-main image.

Use the same five case IDs and official verifier rewards for all timing
comparisons. Record completed/errored/cancelled counts, per-case wall and model
API time, token counts, request-running/waiting samples, KV occupancy,
preemptions, trace recapture count and server identity. Increase to ten cases
only after the five-case results preserve at least the baseline's solved cases
and show a useful wall-time gain. The full-set H100 targets remain recorded;
do not turn a missing reward or smaller denominator into a passing score. At
temperature 1.0, a changed solved-case set on one trajectory is a reason to
repeat the same pinned five-case configuration before treating it as a quality
regression or selecting a new serving policy.

The predeclared second five from September's separate exploratory run are
Terminal `cancel-async-tasks`, `git-multibranch`, `password-recovery`,
`regex-log`, `sqlite-db-truncate`; SWE `django__django-11820`,
`psf__requests-2317`, `pylint-dev__pylint-4970`, `sphinx-doc__sphinx-8265`,
`sympy__sympy-16597`. That Terminal group previously scored 3/5, so a
ten-case aggregate containing it may be easier than the original five. Keep
the fixed-five result visible and avoid comparing a ten-case percentage to a
five-case or full-set target. The
[C2 ten-case branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-ten)
appends these exact IDs after the first five and is ready for dispatch when
the current-main five-case comparison supports expansion.

A [C2 one-hour Terminal pilot branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-60m)
is prepared but has not been dispatched. It changes only the per-case agent
deadline from three hours to one hour. In the archived five-case run, COBOL
passed in nine minutes while the other four failed after 64–171 minutes, so
the shorter limit could remove long unsuccessful tails. Current main may take
different paths and solve cases later; keep the full-deadline run as the
quality control and compare all five rewards before selecting this budget.
Schedule the pilot when clean runner capacity returns.

For a downloaded Actions artifact directory, run
`python3 scripts/gemma4_agentic_summary.py ARTIFACT_DIR --output summary.json`.
The script emits only case IDs, numeric timings/tokens/rewards and server
counts/throughput samples; it omits prompts, patches, shell transcripts and
model responses. Its
Terminal request fit reads token counts from trajectories but drops the
messages; any case with unequal trajectory-step and API-timing counts is
reported as unmatched and excluded from that fit.

The current model uses one physical decode batch equal to the server's
`max_num_seqs`, warmed at startup. It does not switch physical batch sizes
while serving; one C2 server can handle one or two active requests on its
already-warmed physical batch. Changing the physical batch between C1, C2 and
C4 requires a server restart, so startup and warmup are recorded separately
from steady-state throughput. A new prefill signature can retire and recapture
that trace;
the old run's measured recapture time makes this a lower priority than
decoding, task parallelism and KV admission. Any future dynamic-bucket change
must retain warmed traces for each bucket and verify cache/table identity
before replay.

## First current-main throughput sweep (C2, 9 October)

The [C2 benchmark run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962976771)
used Metal main `2c1e1ebdd638886821f35113a5fd0d6335d71608`, the
dedicated `gemma4-31b-qb2` implementation, inference-server
`ebb2cf2870b86df3edd5b6636a1503f1e9426c1d`, two warmed decode rows,
and 256K context. All 23 synthetic points that were attempted completed
without request errors. The run is marked failed by the release threshold at
128/128 C2: measured output 72.56 tokens/s against a 217.65 tokens/s
`complete` threshold. The threshold is a release acceptance gate, not the
observed speed of this implementation. Use the raw points for this batch
comparison; do not report the failed run as a passed release benchmark.

| Input/output tokens | C1 output tok/s | C2 output tok/s | C2/C1 | C1/C2 mean TTFT (ms) |
| --- | ---: | ---: | ---: | ---: |
| 128/128 | 36.98 | 72.56 | 1.96 | 72 / 133 |
| 128/1024 | 36.74 | 72.82 | 1.98 | 72 / 134 |
| 2,048/128 | 33.50 | 61.57 | 1.84 | 303 / 591 |
| 8,192/128 | 26.75 | 42.12 | 1.57 | 1,244 / 2,471 |
| 8,192/1024 | 34.35 | 64.93 | 1.89 | 1,256 / 2,474 |
| 10,000/1024 | 33.87 | 45.19 | 1.33 | 1,627 / 12,173 |
| 16,384/128 | 20.50 | 28.44 | 1.39 | 2,670 / 4,009 |
| 32,768/128 | 13.38 | 16.23 | 1.21 | 5,928 / 8,990 |
| 65,536/128 | 6.99 | 7.70 | 1.10 | 14,537 / 21,922 |

The table compares serial C1/C2 test points within the **same C2 server**,
so C1 here means one active request on its warmed two-row trace, not a
separate one-row server. Decode per-user TPOT at 128/128 was 26.7 ms for
both; the nearly doubled short-context aggregate output came from serving
two users together. Long-prompt C2 advantage shrank, and the 10K/1024 C2
point was anomalously slow to first token. Its four individual TTFTs were
13.8, 29.0, 2.6 and 3.3 seconds; the first two requests dominate the mean.
The server log shows two active requests, no waiting requests, low KV
occupancy, and about 20 seconds of near-zero generation around that point;
two brief trace captures cannot by themselves explain it. This could be a
transient first-pair/prefill scheduling effect. Repeat before treating 45.19
tokens/s as representative steady-state C2 throughput at this shape.
At 65K/128 C2 the server logged
one waiting request in some samples despite reported KV usage below 60%,
consistent with admission or logical-token constraints being relevant as
well as physical page occupancy. This is not yet proof of the exact cause;
review the C4 and 192K sweeps and agentic timing before selecting concurrency.

The complete raw benchmark JSON and server log are downloaded locally under
`/home/mvasiljev/build/gemma-c2-benchmark-main/`. To extract numeric points,
individual TTFTs and peak server KV/waiting observations from this or a
subsequent benchmark artifact, run
`python3 scripts/gemma4_benchmark_summary.py ARTIFACT_DIR --output summary.json`.
This parser omits generated text. For the C2 run it found 23 completed points,
zero failed requests, peak two running/one waiting request, and peak 59% KV
usage. The C4 and C2/192K
sweeps are queued/running respectively. None of these synthetic points
measures official Terminal or SWE rewards.

## First current-main SWE result (C2, 9 October)

The [C2 fixed-five SWE Verified run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962800664)
completed all five cases without case errors on the Metal-main image and
inference-server `ebb2cf2870b86df3edd5b6636a1503f1e9426c1d`. Its
evaluation wall was **2,006 seconds (33.4 minutes)** and its five case clocks
sum to 3,623 seconds, or 1.81 active cases on average. It resolved **1/5**,
the same count as the September fixed-five run, but the solved case changed
from Django to Matplotlib. The stochastic path means this is not proof of
score equivalence; compare the pending current-main C1 control and repeat
before attributing a solved-case change to concurrency.

| Fixed SWE case | C2 reward | Case wall (s) | Input / output tokens |
| --- | ---: | ---: | ---: |
| astropy 14096 | 0 | 807 | 283K / 9.0K |
| django 11299 | 0 | 695 | 765K / 9.0K |
| matplotlib 25332 | 1 | 922 | 641K / 9.6K |
| scikit-learn 14629 | 0 | 584 | 1,084K / 9.0K |
| sympy 13551 | 0 | 616 | 585K / 8.5K |

The 201 ten-second server samples reached two running requests, 35.7% KV
usage (p95 33.0%), and one sample with a waiting request. There were 242
trace warm/capture pairs totaling 72.7 seconds from warm log to capture log;
this is around 3.6% of evaluation wall, though it is not a device-time
measurement. Prompt throughput was positive in 80.1% of samples and
generation in 92.5%; those sampled intervals overlap. The raw artifacts
and numeric summary are under `/home/mvasiljev/build/gemma-swe-c2-main/` and
`/home/mvasiljev/build/gemma-swe-c2-main-summary.json`.

CI marked this run **FAIL** only because its generic acceptance criterion
compares the five-case 20% score with the full 500-case H100 reference of
64.8%. Preserve that full-set reference target; do not interpret this
five-case acceptance gate as a model error or claim that 20% meets 64.8%.
