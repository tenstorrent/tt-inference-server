# Gemma 4 31B QB2 agentic evaluation and serving experiments

Updated 2026-10-09 UTC. This file tracks the dedicated `gemma4-31b-qb2`
implementation on the four-chip P300X2 QuietBox. Results from different model
commits, task lists, or agent policies are kept separate.

## Current readout

| Current-main measurement | Result | Limit |
| --- | --- | --- |
| SWE fixed five, C1 | 1/5 solved, 54.9 min evaluation | Full-set H100 target is 64.8%, not comparable to five cases |
| SWE fixed five, C2 | 1/5 solved, 33.4 min evaluation | Different solved case and 34% fewer output tokens than C1 |
| SWE fixed five, C4 | 2/5 solved, 29.6 min evaluation | Retained both previously solved cases; 800 MHz AICLK warning on this runner |
| C2 synthetic 128/128 | 72.56 output tok/s for two users | CI release throughput gate fails at 217.65 |
| C4 synthetic 128/128 | 131.48 output tok/s for four users | Long-prompt gain is much smaller; gate still fails |
| C2/192K versus 256K | Median absolute rate change 0.12% across 23 shapes | Terminal reward and live KV check pending |
| Terminal fixed five, C2 | 1/5 solved (COBOL), 175.9 min evaluation | C1, C4 and C2/192K pending; paths and token volume differ |

All current-main rows above pin Metal
`2c1e1ebdd638886821f35113a5fd0d6335d71608`, except the 192K
KV-hook row, which uses a separate image built from that commit plus the
isolated sizing change. Details, artifacts and acceptance explanations follow.

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
  waiting at C10. Granite also warms physical decode buckets 1, 8 and 16
  before serving, then switches by live decode count without observed case-time
  compilation; this is the relevant trace-switching precedent. Gemma's
  adapter currently fixes one physical batch at server startup, so the C4
  single-active penalty and the optional C8 sweep determine whether multiple
  warmed physical buckets would buy enough to justify adapting that design.
  Neither Granite timing effect transfers automatically to Gemma.

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
  Main later advanced to `6ea152854fe896a9980e02c51af3cd786a0c6599`
  during these runs. The intervening seven commits changed generic dispatch,
  kernel-build and test code, with no file changes under the dedicated
  `models/demos/gemma4_31b_qb2` implementation. Keep this experiment pinned
  to its start-of-run commit; a follow-up on later main would be a separate
  server-image comparison.
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
  it after those jobs are assigned. A
  [second queued hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37973784755)
  can take the same occupied host when the first hold expires; move queued
  model jobs behind it before that handoff so the dirty runner does not take
  one. Cancel the hold once all model jobs have clean runner assignments.
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
are running on clean hosts, with the occupied host held aside.
The [C2 throughput sweep](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962976771)
and [C4 throughput sweep](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962982690)
reuse that image as separate benchmark jobs. Compare matching input/output
lengths and both per-user latency and aggregate throughput; short synthetic
bursts alone do not establish the fastest agentic suite wall time. The first
C4 benchmark attempt stopped before testing on the same occupied host as the
failed C2 Terminal attempt. The
[C4 benchmark retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963865248)
completed all 23 request shapes on a clean runner; its numeric analysis
appears below.

## Candidates and acceptance

| Variant | Decode rows | Agent trials | Context | Thinking | Metal |
|---|---:|---:|---:|---|---|
| Current-main baseline | 1 | 1 | 256K | On | `2c1e1ebd` |
| C2 | 2 | 2 | 256K | On | `2c1e1ebd` |
| C4 | 4 | 4 | 256K | On | `2c1e1ebd` |
| C8 scaling sweep | 8 | 8 | 256K | On | `2c1e1ebd` |
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
   The candidate requires startup/allocation, high page IDs,
   long prefill/decode, 1- and 2-request quality checks, and an agentic run
   without harmful KV preemption before it can be selected. A host-side
   budget test and Ruff/pre-commit pass. Its
   [benchmark dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37956805379)
   built a separate image from Metal `6b70d1267a5a16410d5aa5c7ec8e851b7ad47f94`
   and inference-server `84c126298977df6f3d90ff2ea1313c7a255ec4c8`:
   `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.24.0-6b70d1267a5a16410d5aa5c7ec8e851b7ad47f94-c62035d-113909389966`.
   Hardware benchmark startup and all 23 synthetic request shapes completed;
   throughput and KV details appear below. The
   [fixed-five Terminal C2/192K job](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37964456985)
   is running on that image with inference-server
   `6682a78c8a1009934838f58b3070927973ec7f4f`; it still needs the
   fixed-case reward and waiting/preemption check.
3. [C2 thinking-off pilot](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-thinkoff):
   changes only the server's `enable_thinking` default. Granite's fixed-case
   pilot reduced output volume markedly, and Gemma's archived latency fit
   makes output reduction promising. Run the same five cases and compare
   rewards, token counts, per-case clocks and API calls; treat it as a distinct
   quality policy, not a reference-equivalent speed result. Do not select it
   solely on a faster wall clock or a five-case score tie. Its
   [fixed-five dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37965857157)
   was cancelled before model work to prioritize the C4 SWE comparison;
   re-dispatch when clean runner capacity returns. It uses inference-server
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
the current-main five-case comparison supports expansion. With C2 SWE's
33.4-minute fixed-five evaluation and 1/5 solved count, the ten-case **SWE
only** expansion is queued in
[QB2 run 37971863336](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37971863336)
on Metal main, inference-server `3bb48bc36c9d47848de6cf01208e3bc6063acb34`.
Its first attempt was canceled while queued to let a guard occupy the known
dirty runner; attempt two is queued with the same inputs. The first five
remain identical to the C2 control. After C4 fixed-five completed faster
and retained both previously solved SWE cases, the
[C4 ten-case branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c4-ten)
and [QB2 dispatch 37985892182](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37985892182)
use the same ten SWE IDs at physical batch four. Report the first-five and
all-ten rewards separately for both variants. Terminal expansion still
awaits the current-main full-deadline result.

A [C2 one-hour Terminal pilot branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-60m)
was queued in [QB2 run 37970041530](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37970041530),
then cancelled before model work to prioritize the C4 SWE comparison on the
clean runners. Re-dispatch after the main C1/C2/C4 and KV evidence is in. It
uses the Metal-main image with inference-server
`202fa984f759d43aac5e927cf66f2989cd028e2f`. The first
[dispatch 37969856863](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37969856863)
used an abbreviated SHA and failed during checkout, before model work.
The pilot changes only the per-case agent
deadline from three hours to one hour. In the archived five-case run, COBOL
passed in nine minutes while the other four failed after 64–171 minutes, so
the shorter limit could remove long unsuccessful tails. Current main may take
different paths and solve cases later; keep the full-deadline run as the
quality control and compare all five rewards before selecting this budget.

For a downloaded Actions artifact directory, run
`python3 scripts/gemma4_agentic_summary.py ARTIFACT_DIR --output summary.json`.
The script emits only case IDs, numeric timings/tokens/rewards and server
counts/throughput samples. For Terminal it also sums case wall, model API and
residual non-API time and reports observed case parallelism; residual time
includes tools and orchestration and is not a device-idle measurement. It
omits prompts, patches, shell transcripts and
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
The dedicated generator's `prefill_forward` loops over prompt rows and calls
`model.prefill_device` once per row. This serial prefill path is a plausible
reason concurrent long-prompt TTFT grows more than decode TPOT; its device
cost and scheduler overlap still require profiling. If C4 long-prompt
results show the same pattern, batched or more overlapped prefill is a better
next engineering target than merely raising agent concurrency.
At 65K/128 C2 the server logged
one waiting request in some samples despite reported KV usage below 60%,
consistent with admission or logical-token constraints being relevant as
well as physical page occupancy. This is not yet proof of the exact cause;
review the C4 sweep and agentic timing before selecting concurrency.

The complete raw benchmark JSON and server log are downloaded locally under
`/home/mvasiljev/build/gemma-c2-benchmark-main/`. To extract numeric points,
individual TTFTs and peak server KV/waiting observations from this or a
subsequent benchmark artifact, run
`python3 scripts/gemma4_benchmark_summary.py ARTIFACT_DIR --output summary.json`.
This parser omits generated text. For the C2 run it found 23 completed points,
zero failed requests, peak two running/one waiting request, and peak 59% KV
usage. The C4 sweep is running. None of these synthetic points
measures official Terminal or SWE rewards.

### C2/192K KV sweep against C2/256K

The [192K benchmark run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37956805379)
started successfully on the separate Metal KV-hook image and completed the
same 23 synthetic request shapes with **zero request failures**. The only CI
failure was the unchanged 128/128 C2 release throughput gate (72.53 versus
217.65 tokens/s). For all 23 matched points, the median absolute output-rate
change from 256K was **0.12%**; the largest was 1.77% at 10K/1024 C2.

| Input/output, active requests | 256K output tok/s | 192K output tok/s |
| --- | ---: | ---: |
| 128/128, 2 | 72.56 | 72.53 |
| 8,192/1024, 2 | 64.93 | 64.94 |
| 10,000/1024, 2 | 45.19 | 45.99 |
| 32,768/128, 2 | 16.23 | 16.22 |
| 65,536/128, 2 | 7.70 | 7.71 |
| 131,072/128, 1 | 2.93 | 2.92 |

Peak sampled KV usage increased from **59.0% to 72.3%** with the smaller
pool; the 1.225 occupancy ratio nearly matches the 1.226 ratio of requested
pool sizes (355,584 / 290,048), consistent with the hook actually shrinking
the allocated pool. Each sweep had a peak of two running and one waiting
request. The
192K server also showed waiting samples at only 21.8% KV usage, so those
samples cannot be explained by exhaustion of its allocated KV pages alone;
prefill/scheduler admission remains a candidate. The
10K/1024 C2 outlier reproduced almost exactly: the first pair's TTFTs were
13.1 and 27.5 seconds at 192K versus 13.8 and 29.0 seconds at 256K; the
second pair was about 2.6 and 3.3 seconds in both. This points to a repeatable
cold first-pair effect at that shape, unrelated to the 192K change. The
smaller KV allocation therefore reserves fewer pages without a measured synthetic
throughput penalty; selection still depends on concurrent Terminal results.
Raw artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-c2-benchmark-kv192/` and
`/home/mvasiljev/build/gemma-c2-benchmark-kv192-summary.json`.

### C4 throughput and admission sweep

The [C4 benchmark run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963865248)
used the same Metal-main image, 256K context and dedicated model with
inference-server `7e6ac75afa416ae28274d8a663809a039ddc3c44`. All 23
attempted shapes completed with zero request failures. The release acceptance
gate failed at 128/128 C4: 131.48 output tokens/s versus its 217.65
`complete` threshold. The one-active-user points on the warmed C4 server
were a median **1.29% slower** than on the warmed C2 server, so keeping four
physical rows did not impose a large idle-row decode penalty in this sweep.
If live C4 rewards and KV admission hold up, one already-warmed C4 physical
trace can cover logical active counts one through four at single-request
granularity without switching model processes; the synthetic single-user
penalty is small enough to make that simpler policy plausible.

| Input/output tokens | C2 active-two output tok/s | C4 active-four output tok/s | C4/C2 |
| --- | ---: | ---: | ---: |
| 128/128 | 72.56 | 131.48 | 1.81× |
| 128/1024 | 72.82 | 142.75 | 1.96× |
| 1,024/128 | 66.69 | 120.73 | 1.81× |
| 2,048/128 | 61.57 | 105.13 | 1.71× |
| 4,096/128 | 53.46 | 81.53 | 1.53× |
| 8,192/128 | 42.12 | 58.87 | 1.40× |
| 8,192/1024 | 64.93 | 116.52 | 1.79× |
| 10,000/1024 | 45.19 | 55.41 | 1.23× |
| 16,384/128 | 28.44 | 35.21 | 1.24× |
| 32,768/128 | 16.23 | 18.18 | 1.12× |

For a fixed 128-token completion, the gain from adding two more active
requests declines steadily as input grows from 1K to 32K. Gemma's prefill
loop processes concurrent rows serially, so C4 adds useful decode overlap
but cannot provide the same gain when prompt processing dominates. Terminal
trajectories build large prompts; this scaling curve argues against assuming
that C8 will double Terminal throughput. A
[C8 catalog candidate](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c8)
is prepared for a benchmark on a clean runner after the active evaluation
queue, to locate that limit before spending hours on C8 agentic cases. Its
[pinned-main dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37984140248)
uses the same Metal image and full inference-server commit
`5f0374c51d6d883767f48af4c63a5027c2883437`. The first setup dispatch
with a short checkout SHA was canceled; the next failed server-type resolution
because its implementation ID used underscores instead of the catalog's
`gemma4-31b-qb2`. Neither reached hardware. The linked dispatch corrects
both inputs.

The benchmark selector reduced the 65,536/128 point to three concurrent
requests under the 256K shared context; C3 output was 7.96 tokens/s versus
C2's 7.70. This is a practical limit on long-prompt C4 scaling. C4 reached
four running requests, two waiting, and 75.9% peak sampled KV usage. Waiting
appeared in 12/159 samples (7.5%), versus 2/147 (1.4%) for C2/256K and
4/146 (2.7%) for C2/192K. The test shape mix differs between C2 and C4,
and ten-second samples miss short waits; these rates are a warning to inspect
live task admission, not a normalized throughput metric. Several
waiting samples had only 17–38% KV usage, so prefill/scheduler admission is
again involved; sampled occupancy alone does not establish page exhaustion.
At 10K/1024 the first four C4 TTFTs were 13.2, 70.2, 70.5 and 80.5 seconds,
while the next four were 2.6–5.3 seconds. The same cold-first-wave pattern
appeared at C2 and on both KV sizes, but grows more severe at four requests.
The mechanism has not been profiled. It argues for measuring and warming the
expected active-count/prefill signatures before concluding that C4 is safe
for tight latency deadlines. The raw artifact and numeric summary are under
`/home/mvasiljev/build/gemma-c4-benchmark-main/` and
`/home/mvasiljev/build/gemma-c4-benchmark-main-summary.json`.
During the cold 10K C4 wave, server samples from 18:34:20 through 18:35:10
showed two or three running requests, one or two waiting, only 17–23% KV
usage, and near-zero generation; samples from 18:34:30 through 18:35:10
also showed zero prompt throughput. The trace warm/capture log immediately
before this interval lasted about 0.25 seconds, far short of the 50-second
quiet period. This narrows the likely delay to prefill/device work or
scheduler admission rather than trace capture or KV capacity, but the logs
still do not separate compilation from device execution.
One source-level clue: the vLLM scheduler caps each prefill step at 8,192
tokens, while the model processes those steps row by row and uses internal
6,656-token full-attention or 1,024-token sliding chunks. A 10K prompt needs
an approximately 1,808-token continuation after the first 8,192; 8K and 16K
benchmark prompts do not create that tail shape. The repeated first-wave
delay is consistent with a cold partial-chunk program or scheduling path,
but the retained logs do not time kernel compilation or device operations,
so this remains a hypothesis for a targeted warm/cold profile.
A separate [later-main C4 benchmark](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37975935523)
was dispatched with Metal `6ea152854fe896a9980e02c51af3cd786a0c6599`
and the same C4 inference-server commit. Its image build succeeded in
67 minutes and the benchmark completed on `120-qb2-p04t05`, the host that
logged an 800 MHz AICLK warning in C4 SWE. It logged the same warning in this
benchmark, so cross-host throughput is confounded. Among the intervening main commits,
`bde0257bf3c` defers duplicate kernel builds without holding worker threads;
it might affect cold-prefill latency, but was measured upstream on a different
model. Compare the first and second 10K C4 waves and steady-state points to
the original pinned-main run; this follow-up is not part of the C1/C2/C4
same-image comparison.

The later-main benchmark finished all **23 shapes with zero request errors**.
Its generic release gate still failed: 128/128 C4 produced 105.50 output
tokens/s versus the 217.65 target. The matching pinned-main C4 point was
131.48 tokens/s on `120-qb2-p05t01`, a host without the AICLK warning.
Single-user short-prompt output rates were 9–23% lower on the later run;
most 16K–131K input points were within a few percent, with the 32K
single-user point 7% lower.
That shape-dependent difference is consistent with host clock effects and/or
intervening Metal changes, and these runs cannot apportion it. The later
test job took 37.2 minutes versus 32.7 minutes for the pinned-main sweep.

The important cold-prefill result **did not change**: at 10K/1024 C4, the
later run's first four TTFTs were 13.29, 70.56, 70.85 and 80.94 seconds,
versus 13.24, 70.23, 70.51 and 80.48 on pinned main. The next four were
2.59–5.24 seconds, versus 2.61–5.30. This close match on distinct runners
shows that the seven intervening Metal commits, including the generic
duplicate-build change, did not remove Gemma's cold 10K stall. It does not
identify the underlying compilation/device/scheduler step. The later run
had 13/176 ten-second samples with waiting and 67.0% peak KV use; the
pinned run had 12/159 and 75.9%. These samples do not support a KV-capacity
explanation for the repeated first-wave latency.

Both C4 benchmark starts reused host-local model weights: their Hugging Face
snapshot steps finished in about **0.1 seconds**, and API readiness followed
in about **4 min 44–49 s**. This contrasts with the approximately 7 min 45 s
snapshot and 12 min 40 s readiness on cold SWE runners. The model cache
persists across separate jobs on the same host, while a fresh host can still
pay the cold cost. Raw later-main artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-c4-benchmark-later-main/` and
`/home/mvasiljev/build/gemma-c4-benchmark-later-main-summary.json`.

## First current-main SWE result (C2, 9 October)

The [C2 fixed-five SWE Verified run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962800664)
completed all five cases without case errors on the Metal-main image and
inference-server `ebb2cf2870b86df3edd5b6636a1503f1e9426c1d`. Its
test job lasted about **49 minutes** including runner/server preparation;
evaluation wall was **2,006 seconds (33.4 minutes)** and its five case clocks
sum to 3,623 seconds, or 1.81 active cases on average. It resolved **1/5**,
the same count as the September fixed-five run, but the solved case changed
from Django to Matplotlib. The stochastic path means this is not proof of
score equivalence; repeat before attributing a solved-case change to
concurrency.

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
The same fixed-five SWE suite completed at C4 in
[QB2 run 37970495434](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37970495434)
on the Metal-main image and inference-server
`7e6ac75afa416ae28274d8a663809a039ddc3c44`; its per-case result is
analyzed below.
Once a physical batch size and context are selected, a single
`tb2.0,swebench` dispatch can run both suites against one warmed server,
amortizing startup. Separate jobs remain useful for parallel exploratory
measurements and independent failure isolation. The Shield partition change
accepts that combined selector; no combined result exists yet.

### Same-main C1 versus C2 SWE

The [C1 fixed-five SWE control](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37962784599)
also completed five cases with no case errors and resolved **1/5** (Django).
It used the same Metal-main image, fixed task IDs, sampling and 256K context;
its inference-server commit `fb5b77a4e26af79c455af2893babf1a9928bfc08`
has one physical decode row and one concurrent trial. C2 resolved
Matplotlib instead, so the equal count does not establish per-case quality
parity. Both CI jobs failed the generic full-set reference gate discussed
above, not a test execution check.

| Same-main fixed-five SWE | C1 | C2 | C2/C1 or difference |
| --- | ---: | ---: | ---: |
| Evaluation wall | 54.94 min | 33.44 min | 1.64× faster observed |
| Summed five case clocks | 54.94 min | 60.39 min | +9.9% |
| Observed active-case parallelism | 1.00 | 1.81 | More overlapping work |
| Full CI test job | 70.42 min | 49.20 min | 1.43× faster observed |
| Solved | 1/5 Django | 1/5 Matplotlib | Different case |
| Total input tokens | 3.122M | 3.358M | +7.6% |
| Total output tokens | 68,495 | 45,256 | −33.9% |
| Model API calls | 220 | 240 | +9.1% |
| Largest prompt | 32,253 tokens | 34,040 tokens | Both below 192K |
| Peak / p95 KV sampled | 31.9% / 28.1% | 35.7% / 33.0% | No sustained pressure |
| Samples with waiting | 0/330 | 1/201 | No sustained queue |
| Trace warm-to-capture time | 68.8 s / 220 | 72.7 s / 242 | Small relative to suite |

The observed suite/job speedups are real for these two runs but **are not a
controlled batching speedup**: at temperature 1.0 C2 made 20 more API calls
but generated 23,239 fewer output tokens, and different code/tool trajectories changed case clocks and solved
identities. The synthetic same-server 128/128 C1/C2 point isolates a nearly
2× aggregate short-context throughput effect more cleanly; repeated fixed
case runs and the C4 result are needed for an end-to-end policy.
Notably, C2's **summed** case clocks were longer, yet its suite elapsed time
was shorter because the cases overlapped. This directly supports task
parallelism as a useful mechanism, without assigning a precise share of the
21.5-minute wall reduction to it.
The C1 raw artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-swe-c1-main/` and
`/home/mvasiljev/build/gemma-swe-c1-main-summary.json`.
For context only, September's older C1 fixed-five SWE run took 120.1 minutes
and yielded about 46.4K output tokens; current-main C1 took 54.9 minutes
despite about 68.5K output tokens. The Metal commit and prefill policy changed
and the stochastic trajectories differ, so this is evidence of a much faster
observed end-to-end run, not a measured speedup attributable to one code
change.

### Same-main C4 SWE result

The [C4 fixed-five SWE run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37970495434)
completed all five cases with zero case errors. It resolved **2/5**:
Django and Matplotlib, the union of the cases solved by C1 and C2. This is
encouraging quality evidence on the fixed task list, but a single sampled
trajectory at temperature 1.0 does not establish a reliable score change.
CI reported failure because the generic acceptance gate compared the five-case
40% score to the full 500-case H100 target of 64.8%.

| Fixed-five SWE measure | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| Evaluation wall, min | 54.94 | 33.44 | 29.57 |
| Full test job, min | 70.42 | 49.20 | 45.32 |
| Sum of case clocks, min | 54.94 | 60.39 | 102.92 |
| Observed active-case parallelism | 1.00 | 1.81 | 3.48 |
| Resolved | 1/5 | 1/5 | 2/5 |
| Input tokens, M | 3.122 | 3.358 | 3.086 |
| Output tokens | 68,495 | 45,256 | 55,891 |
| Model API calls | 220 | 240 | 218 |
| Peak / p95 KV sampled | 31.9% / 28.1% | 35.7% / 33.0% | 46.6% / 37.2% |
| Samples with waiting | 0/330 | 1/201 | 3/178 |

| SWE case | C1 | C2 | C4 | C4 wall (s) | C4 input / output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Astropy 14096 | 0 | 0 | 0 | 1,627 | 603K / 15.3K |
| Django 11299 | 1 | 0 | 1 | 904 | 212K / 5.5K |
| Matplotlib 25332 | 0 | 1 | 1 | 1,175 | 996K / 7.8K |
| scikit-learn 14629 | 0 | 0 | 0 | 696 | 439K / 9.7K |
| Sympy 13551 | 0 | 0 | 0 | 1,774 | 837K / 17.7K |

C4 observed a **1.13× shorter evaluation wall than C2** and **1.86× than
C1**, while its summed case clocks were 70% longer than C2's. Concurrent
agent work, not uniformly faster cases, produced the shorter wall. C4 also
generated 23.5% more output tokens than C2, so their suite clocks cannot
isolate a decode throughput effect. Three of 178 ten-second server samples
had waiting requests; peak KV was only 46.6%, with no evidence here of
sustained KV pressure. Trace warm/capture pairs numbered 216 and summed to
69.5 seconds, again small against the suite wall.

This C4 run used runner `120-qb2-p04t05`. Its server emitted two startup
warnings that AICLK was expected at 1350 MHz but observed at 800 MHz,
clamped by firmware max-arbiter index 5. The C1 and C2 SWE server logs did
not emit that warning. The warning may affect C4 timing; these logs do not
show its duration or per-chip clocks during the cases, so no clock-normalized
speedup is claimed. A repeat on a healthy runner would help separate host
frequency from stochastic case paths. Raw artifacts and numeric summary:
`/home/mvasiljev/build/gemma-swe-c4-main/` and
`/home/mvasiljev/build/gemma-swe-c4-main-summary.json`.

### Startup cost and reuse

The C2 SWE job spent about **12 min 40 s** between starting the inference
server at 17:00:04 UTC and API readiness at 17:12:44. Its Docker image pull
took about one second, while the server's Hugging Face snapshot step for
`google/gemma-4-31B-it` took **7 min 45 s** (17:00:08–17:07:54); vLLM import,
device setup, weight loading and warmup occupied the remainder. Thus starting
another server for every short evaluation has a measurable cost independent
of case execution. Running the selected Terminal and SWE suites against one
warmed server should save one startup interval, subject to the combined job
using the same physical batch, context, model image and scoring settings.
The snapshot time is from this runner and may vary with cache state and
network throughput. The independent C2 benchmark on runner
`120-qb2-p05t01` took 7 min 44 s in the same snapshot step, while C2 SWE
ran on `qb2-120-p05t02`. Both were separate server starts and runner hosts;
these logs show repeated snapshot cost, though not whether each byte came
from the network rather than a lower-level cache.

## Current-main Terminal fixed-five result at C2

The [C2 fixed-five Terminal run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963842946)
completed all five official Terminal-Bench 2.0 cases with zero case errors
on the pinned-main Metal image and inference-server
`ebb2cf2870b86df3edd5b6636a1503f1e9426c1d`. It resolved **1/5**,
COBOL modernization, the same case resolved by the September C1 run.
The generic CI accuracy gate marked the five-case 20% score as a failure
against the full 89-case H100 target of 44.94%; that does not make the
executed cases invalid or establish target parity.

| Terminal case | Reward | Wall min | Model API min | Other min | Calls | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HTML filter | 0 | 171.3 | 99.8 | 71.5 | 50 | 199,608 |
| COBOL modernization | 1 | 9.6 | 9.2 | 0.4 | 6 | 19,663 |
| CompCert | 0 | 120.5 | 68.5 | 52.0 | 50 | 125,230 |
| FEAL | 0 | 28.8 | 13.3 | 15.5 | 6 | 28,917 |
| QEMU startup | 0 | 17.0 | 11.8 | 5.2 | 14 | 25,672 |

The evaluation wall was **175.87 minutes** and the five case clocks sum to
347.12 minutes, an observed active-case parallelism of **1.97**. Its full
test job lasted **182.57 minutes**. The 126 matched model calls used 2.717M
input and 399,090 output tokens; summed API time was 202.61 minutes
(58.4% of the case-clock sum), with 144.51 minutes outside the API. The
request timing fit is 7.20 seconds/call + 0.166 seconds/1K prompt tokens +
27.06 seconds/1K output tokens (R² 0.980). This fit is observational,
not device profiling. The older C1 matched-call fit gave 26.28 seconds/1K
output tokens, so these artifacts do not show a per-output-token decode
speedup from C2. They do show that output volume and agent overlap matter.

September's older-Metal C1 fixed-five evaluation took **7.84 hours** and
generated about **557K** output tokens, versus current-main C2's **2.93 hours**
and **399K** output tokens. The observed suite wall is 2.67× shorter,
but the runs differ in Metal commit, chunked-prefill policy, concurrency and
sampled trajectories. HTML still lasted about 171 minutes; CompCert fell
from 162 to 121, FEAL from 64 to 29, and QEMU from 64 to 17 minutes.
The case-clock sum fell about 26% while suite wall fell about 63%, directly
supporting overlap as a major contributor without assigning it an isolated
causal speedup. A same-main C1 control is still running.

The C2 server reached two running requests with **zero waiting samples**
among 1,056 ten-second samples. Sampled KV usage peaked at 52.6% (p95
34.4%); 143 trace warm/capture intervals summed to 61.5 seconds. No
AICLK-clamp warning appeared in its server log. This was a warm-weight
restart on the same host as C2 SWE: the Hugging Face snapshot step took
about 0.2 seconds and server readiness about 4 min 47 s. The result
supports testing a smaller KV pool, but its peak occupancy alone cannot
establish safety for four simultaneous long Terminal requests. Raw artifacts
and numeric summary are under `/home/mvasiljev/build/gemma-terminal-c2-main/`
and `/home/mvasiljev/build/gemma-terminal-c2-main-summary.json`.
