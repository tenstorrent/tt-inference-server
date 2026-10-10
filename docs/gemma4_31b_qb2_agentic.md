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
| SWE expanded ten, C2 | 3/10 solved, 61.5 min evaluation | First five 3/5; second five 0/5; AICLK warning |
| SWE expanded ten, C4 | 3/10 solved, 55.9 min evaluation | First five 2/5; second five 1/5; 21% more output than C2 |
| SWE expanded ten, C8 | 3/10 solved, 44.4 min evaluation | Same solved cases as C4; 800 MHz AICLK warning; 67.9% peak KV |
| C2 synthetic 128/128 | 72.56 output tok/s for two users | CI release throughput gate fails at 217.65 |
| C4 synthetic 128/128 | 131.48 output tok/s for four users | Long-prompt gain is much smaller; gate still fails |
| C8 synthetic 128/128 | 273.57 output tok/s for eight users | Release gate passes; cold 10K TTFT reaches 135 s |
| C2/192K versus 256K | Median absolute synthetic rate change 0.12%; live Terminal 1/5 at both sizes | 192K had one timeout and no measured wall advantage |
| Terminal fixed five, C1 | 0/5 solved, 270.1 min evaluation | Sampling paths differ from C2/C4; full-set target remains 44.94% |
| Terminal fixed five, C2 | 1/5 solved (COBOL), 175.9 min evaluation | 192K live run also 1/5, with more output and a timeout |
| Terminal fixed five, C4 | 1/5 solved (COBOL), 170.6 min evaluation | FEAL became a 170.6 min straggler; no sampled KV waiting |
| Terminal fixed five, C2 thinking off | 1/5 solved (COBOL), 37.4 min evaluation | 86.5% fewer output tokens; ten-case quality check dispatched |
| Terminal fixed five, C2 one-hour cap | 1/5 solved (COBOL), 102.1 min evaluation | Two explicit timeouts; ten-case control dispatched |
| Terminal expanded ten, C4 | 4/10 solved, 110.3 min evaluation | Original first five 2/5; added five 2/5; FEAL tail did not recur |
| Combined ten each, C2 thinking off | Terminal 3/10 in 54.4 min; SWE 3/10 in 66.1 min | One warmed server; Terminal faster with lower reward than C4/ten, SWE slower than C2/ten |

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
  hardware untouched while clean hosts accept the Gemma jobs. A
  [second hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37973784755)
  occupied the same host on its second attempt and later completed after all
  queued model jobs had clean runner assignments. The stopped container was
  not modified by this work.
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
later completed on clean hosts, with the occupied host held aside.
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
   finished on that image with inference-server
   `6682a78c8a1009934838f58b3070927973ec7f4f`. The fixed-case reward,
   timeout and waiting comparison appear below.
3. [C2 thinking-off pilot](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-thinkoff):
   changes only the server's `enable_thinking` default. Granite's fixed-case
   pilot reduced output volume markedly, and Gemma's archived latency fit
   makes output reduction promising. Run the same five cases and compare
   rewards, token counts, per-case clocks and API calls; treat it as a distinct
   quality policy, not a reference-equivalent speed result. Do not select it
   solely on a faster wall clock or a five-case score tie. Its
   [fixed-five dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37965857157)
   was cancelled before model work to prioritize the C4 SWE comparison;
   the [same fixed-five pilot](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37991891787)
   was then re-dispatched after the C2/C4 full-deadline results and
   completed. It uses inference-server
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
appends these exact IDs after the first five. With C2 SWE's
33.4-minute fixed-five evaluation and 1/5 solved count, the ten-case **SWE
only** expansion completed in
[QB2 run 37971863336](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37971863336)
on Metal main, inference-server `3bb48bc36c9d47848de6cf01208e3bc6063acb34`.
Its first attempt was canceled while queued to let a guard occupy the known
dirty runner; attempt two completed with the same inputs. The first five
remain identical to the C2 control. After C4 fixed-five completed faster
and retained both previously solved SWE cases, the
[C4 ten-case branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c4-ten)
and [QB2 dispatch 37985892182](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37985892182)
use the same ten SWE IDs at physical batch four. Report the first-five and
all-ten rewards separately for both variants. After the successful C8
synthetic sweep, the
[C8 ten-case branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c8-ten)
and [QB2 dispatch 37994423239](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37994423239)
used those same ten SWE IDs at physical batch eight and completed; results
appear below. Matched ten-case Terminal expansions are running at
[C2](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37996976665)
and [C4](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37996981703)
with the same pinned Metal-main image and ten IDs used for SWE. Their first
five are the original fixed-five control; score that subset separately from
all ten, and compare case wall and token volume alongside total wall. These
jobs retain the three-hour per-case deadline and temperature 1.0.

A [C2 one-hour Terminal pilot branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-60m)
was queued in [QB2 run 37970041530](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37970041530),
then cancelled before model work to prioritize the C4 SWE comparison on the
clean runners. After the C2 and C4 full-deadline fixed-five results, the
[same one-hour pilot was re-dispatched](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37991526452)
and completed; its timing and timeout results appear below. It uses
the Metal-main image with inference-server
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
usage. The C4 sweep finished later and is analyzed below. None of these synthetic points
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

### C8 throughput, admission and trace granularity

The [pinned-main C8 benchmark](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37984140248)
used the same Metal image as C2/C4 and inference-server
`5f0374c51d6d883767f48af4c63a5027c2883437`, which changes physical
decode rows and maximum agent trials to eight. It completed **all 23
synthetic shapes with zero failed requests** and the CI benchmark run
passed. The 128/128 active-eight point produced **273.57 output tokens/s**,
above the release `complete` threshold of 217.65; C4 active-four reached
131.48 on another healthy runner. This establishes useful higher-batch
synthetic capacity, not an eight-agent quality or suite-wall result.

| Input/output tokens | C4 active-four tok/s | C8 active-eight tok/s | C8/C4 |
| --- | ---: | ---: | ---: |
| 128/128 | 131.48 | 273.57 | 2.08× |
| 128/1024 | 142.75 | 311.22 | 2.18× |
| 1,024/128 | 120.73 | 226.11 | 1.87× |
| 4,096/128 | 81.53 | 121.12 | 1.49× |
| 8,192/128 | 58.87 | 75.82 | 1.29× |
| 8,192/1024 | 116.52 | 210.43 | 1.81× |
| 10,000/1024 | 55.41 | 79.08 | 1.43× |
| 16,384/128 | 35.21 | 40.83 | 1.16× |

At 32K/128, the shared-context selector capped C8 to seven live requests:
19.46 tokens/s versus C4's four-user 18.18, only a 1.07× gain. At
65K/128 both variants were capped to three users (8.09 versus 7.96
tokens/s), and 131K permitted one. This continues the sharp decline in
batch gains as serial per-row prefill and shared-context admission dominate.
The C8 server's one-active points were a median **7.0% faster** than C4's
one-active points across the 12 matched shapes, so there is no observed
idle-row penalty in this cross-host sweep. C8 ran on `qb2-120-p05t02` and
C4 on `120-qb2-p05t01`; neither logged an AICLK clamp, but host effects
remain unmeasured and prevent attributing that single-user difference to
physical batch size.

C8 reached eight running requests, six waiting requests, and 71.4% peak
sampled KV use. Waiting appeared in **27/171 samples (15.8%)**, compared
with C4's 12/159 (7.5%). Several C8 waiting intervals had only 16–36%
sampled KV use and no preemption was logged, again pointing to prefill or
scheduler admission rather than a simple page-capacity limit. At 10K/1024
the first eight TTFTs spanned **13.1–135.3 seconds**; the next eight were
2.6–6.7 seconds. The larger cold burst makes eight concurrent agents a
latency risk until expected prefill signatures can be warmed or the stall
is fixed. One physical-eight trace can serve logical counts one through
eight without a runtime batch switch, but the current evidence does not
justify replacing C4 or C2 for long-prompt Terminal jobs. Raw artifacts
and numeric summary are under `/home/mvasiljev/build/gemma-c8-benchmark-main/`
and `/home/mvasiljev/build/gemma-c8-benchmark-main-summary.json`.

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

### C2 ten-case SWE expansion

The [C2 ten-case SWE run, attempt two](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37971863336)
completed all ten cases with zero case errors on pinned-main Metal. It
resolved **3/10**: Django 11299, Matplotlib 25332 and scikit-learn 14629.
The **original first five scored 3/5** in this run, and the predeclared
second five scored **0/5**. The earlier separate C2 fixed-five run scored
1/5 on the same first task IDs. The changed solved set demonstrates how
unstable a single temperature-1 five-case score can be; it does not prove
that adding cases improved the model. The generic CI accuracy gate failed
because it compared this ten-case 30% with the full 500-case H100 target
of 64.8%.

| First-five SWE case | Reward | Second-five SWE case | Reward |
| --- | ---: | --- | ---: |
| Astropy 14096 | 0 | Django 11820 | 0 |
| Django 11299 | 1 | requests 2317 | 0 |
| Matplotlib 25332 | 1 | pylint 4970 | 0 |
| scikit-learn 14629 | 1 | Sphinx 8265 | 0 |
| Sympy 13551 | 0 | Sympy 16597 | 0 |

The ten-case evaluation wall was **61.54 minutes**, with 119.58 minutes
of summed case clocks and observed active-case parallelism **1.94**. Its
full test job lasted 67.85 minutes. The run made 403 model API calls,
using 5.749M input and 76,692 output tokens; the largest prompt was
40,005 tokens. The first five contributed 188 calls, 2.131M input,
40,079 output tokens and 47.87 minutes of summed case clocks. Those case
clocks should not be interpreted as an isolated first-five suite wall
because all ten cases shared two agent slots.

Server samples reached two running requests, peak/p95 KV 38.1%/31.2%,
and four waiting samples out of 370; 401 trace warm/capture pairs summed
to 97.0 seconds. There was no sustained KV queue. The run used the
warm-weight host `120-qb2-p04t05`, whose server again logged two startup
AICLK warnings with 800 MHz observed versus 1350 MHz expected. Thus its
wall time should not be treated as a normalized speed comparison with the
healthy-host C2 fixed-five control or the C4 ten-case run. Raw
artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-swe-c2-ten-main/` and
`/home/mvasiljev/build/gemma-swe-c2-ten-main-summary.json`.

### C4 ten-case SWE expansion

The [matched C4 run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37985892182)
completed **10/10 cases with zero case errors** on pinned-main Metal and
solved **3/10**. The original first five scored **2/5** (Django 11299 and
Matplotlib 25332); the added five scored **1/5** (requests 2317).
C2/ten also scored 3/10, but solved scikit-learn 14629 instead of requests
2317. A single temperature-1 trajectory cannot establish that C4 preserves
per-case accuracy, even though the aggregate reward is equal. The generic
64.8% full-set H100 gate again failed on this ten-case 30% subset.

| Ten-case SWE measure | C2 | C4 |
| --- | ---: | ---: |
| Evaluation wall, min | 61.54 | 55.90 |
| Sum of case clocks, min | 119.58 | 203.20 |
| Observed case parallelism | 1.94 | 3.64 |
| Solved first five / added five | 3 / 0 | 2 / 1 |
| Model API calls | 403 | 450 |
| Input / output tokens | 5.749M / 76.7K | 7.574M / 92.7K |
| Peak / p95 sampled KV | 38.1% / 31.2% | 49.0% / 42.7% |
| Waiting samples | 4/370 | 10/337 |

C4 shortened observed evaluation wall by **9.2%** while its summed case
clocks grew **70%** and output grew **21%**. Four overlapping trials hid much
of the extra work, but their changed paths prevent interpreting the wall
difference as isolated model throughput. In particular, the original first
five used 3.918M input tokens and 56,880 output tokens at C4 versus 2.131M
and 40,079 at C2. The C2 run's runner logged an 800 MHz AICLK warning; the
C4 runner `qb2-120-p05t05` did not, so host clocks also confound the wall
comparison. C4 reached four running requests, and its 10 waiting samples
out of 337 did not indicate sustained KV saturation. Its 444 trace
warm/capture intervals summed to 105.1 seconds. Raw artifacts and numeric
summary are under `/home/mvasiljev/build/gemma-swe-c4-ten-main/` and
`/home/mvasiljev/build/gemma-swe-c4-ten-main-summary.json`.

### C8 ten-case SWE expansion

The [matched C8 run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37994423239)
completed **10/10 cases with zero case errors** on pinned-main Metal. It
solved **3/10**, exactly the same three cases as C4/ten: Django 11299,
Matplotlib 25332 and requests 2317. The original first five scored 2/5
and the added five 1/5. C2/ten also scored 3/10 but solved scikit-learn
14629 instead of requests 2317. All three still fail the generic CI gate
against the full-set H100 SWE target of 64.8%; none establishes target
accuracy on the complete 500-case suite.

| Ten-case SWE measure | C2 | C4 | C8 |
| --- | ---: | ---: | ---: |
| Evaluation wall, min | 61.54 | 55.90 | 44.35 |
| Sum of case clocks, min | 119.58 | 203.20 | 284.38 |
| Observed case parallelism | 1.94 | 3.64 | 6.41 |
| Solved first five / added five | 3 / 0 | 2 / 1 | 2 / 1 |
| Model API calls | 403 | 450 | 425 |
| Input / output tokens | 5.749M / 76.7K | 7.574M / 92.7K | 5.935M / 82.8K |
| Peak / p95 sampled KV | 38.1% / 31.2% | 49.0% / 42.7% | 67.9% / 57.4% |
| Waiting samples | 4/370 | 10/337 | 10/267 |

C8 shortened observed wall **20.7% versus C4** and **27.9% versus C2**.
Its eight active requests produced 6.41 case-clock minutes per evaluation
minute; individual case clocks grew with concurrency, but overlap more than
offset that growth. C8 and C2 ran on the same `120-qb2-p04t05` host, which
logged the 800 MHz versus 1350 MHz AICLK warning in both runs; C4 used a
healthy host. C8 still differs in token volume and case paths, so these are
suite-wall observations, not controlled model-throughput multipliers.
Output tokens divided by evaluation wall were 20.8, 27.6 and 31.1 tokens/s
at C2, C4 and C8, respectively. Those mixed-workload rates rise much less
than the short 128/128 synthetic 72.6, 131.5 and 273.6 tokens/s points:
frequent prefill, tool turns and case scheduling dilute the decode-batch
gain. Different generated-token totals prevent treating the ratios as
isolated model throughput.
Ten of 267 server samples had waiting requests, the maximum sampled KV was
67.9%, and no server preemption was logged. There is capacity margin for
this ten-case SWE trajectory at 256K, though larger or longer cases need
their own admission check. The longest C8 case, requests 2317, occupied
the full 44.35-minute evaluation wall; it remained solved. The 406 trace
warm/capture intervals summed to 96.9 seconds. A single warmed physical
eight-row trace handled varying logical active counts without a live
batch-switch or restart, which is the useful granularity for this SWE
configuration. Raw artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-swe-c8-ten-main/` and
`/home/mvasiljev/build/gemma-swe-c8-ten-main-summary.json`.

With the SWE/ten capacity and reward check complete, a
[matched C8 Terminal/ten dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38001414052)
was attempted with the same pinned Metal image and Terminal IDs as C2/ten
and C4/ten. It failed **before model loading or evaluation** on
`120-qb2-p04t05`: the fabric topology mapper could not map the four-chip
logical mesh onto the discovered four ASICs (`topology_mapper.cpp:556`).
The same host had completed C8 SWE minutes earlier; this run provides no
Terminal speed or reward evidence. The two startup attempts and server
logs are preserved under
`/home/mvasiljev/build/gemma-terminal-c8-ten-startup-fail/`. A retry needs
a healthy host or a bounded device list/reset/list and mesh-open check by
the host operator. Do not fold this infrastructure failure into the model
score denominator. Long-prompt prefill and shared-context admission can
limit C8, so a successful full suite is still needed before preferring it
for Terminal.

The first [C4/ten repeat dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010281587)
was canceled during runner setup after assignment to the previously occupied
`p04t07` host. A second [C8/ten Terminal dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010505443)
was likewise canceled during setup after assignment to the host with the
fabric mapping failure; neither reached model work. Wrapper commit
`a375735` added a read-only exact-runner hold to the existing dispatch
workflow. The [occupied-host hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010437751)
then reserved `p04t07`, and the [named health hold](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010838406)
reserved `p04t05` without touching containers or devices. A temporary
healthy-host hold was canceled after the health hold started. The
[C4/ten repeat](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010466649)
is running on clean `p05t06`, and the
[C8/ten Terminal retry](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38010947583)
is running on clean `p05t05`. Compare the latter's case rewards and wall
with C2/C4 ten-case controls once it completes; the canceled attempts
remain infrastructure scheduling history, not reward observations.

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
causal speedup. The same-main C1 control completed later and is compared
below.

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

### C4 Terminal versus C2 on pinned main

The [C4 fixed-five Terminal run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963848727)
also completed all five cases with zero case errors and resolved **1/5**,
COBOL. Its CI failure was again the five-case 20% score against the 44.94%
full-set H100 target. It used the same Metal image and five tasks as C2,
but a four-row physical decode trace and four simultaneous agent trials.

| Fixed-five Terminal measure | C2 | C4 |
| --- | ---: | ---: |
| Evaluation wall, min | 175.87 | 170.60 |
| Full test job, min | 182.57 | 177.08 |
| Sum of case clocks, min | 347.12 | 460.68 |
| Observed active-case parallelism | 1.97 | 2.70 |
| Solved | 1/5 COBOL | 1/5 COBOL |
| Input / output tokens | 2.717M / 399K | 3.091M / 410K |
| Recorded API calls | 126 | 142 |
| Sum of API / other case time, min | 202.61 / 144.51 | 247.66 / 213.02 |
| Peak / p95 sampled KV | 52.6% / 34.4% | 50.5% / 34.3% |
| Samples with waiting | 0/1,056 | 0/1,024 |

C4's observed suite wall is only **3.1% shorter** than C2's, even though
it overlapped more case work. Its case-clock sum was 33% longer and its
request path had 16 more calls, 14% more input tokens and 3% more output
tokens. As with SWE, stochastic tool and text trajectories prevent a
controlled end-to-end batching speedup claim. The synthetic short-prompt
throughput gain does not translate into a similar suite-wall gain when a
slow case sets the deadline.

| Terminal case wall, min | C2 | C4 | C4 change |
| --- | ---: | ---: | ---: |
| HTML filter | 171.3 | 168.3 | −3.0 |
| COBOL modernization | 9.6 | 14.7 | +5.1 |
| CompCert | 120.5 | 84.3 | −36.2 |
| FEAL | 28.8 | 170.6 | +141.8 |
| QEMU startup | 17.0 | 22.7 | +5.7 |

FEAL was the C4 straggler: **40.2 minutes** of recorded model API time
and **130.4 minutes** outside those calls, compared with C2's 13.3 and
15.5 minutes. The C4 trajectory's first tool command, `cat /app/feal.py`,
completed in about 0.01 seconds in its terminal recording. The gap from
that step to the next model step was 78.6 minutes, while its corresponding
recorded API time was 8.1 minutes; about **70.5 minutes of that gap** are
unaccounted for by that recorded API time or shell command. The C2
FEAL trajectory had a 19.0-minute analogous gap, 3.9 minutes of recorded
API time and about 15 minutes residual. C4's trial log records 14
output-limit warnings across 15 model steps, versus two across six in C2,
and many malformed JSON/parser warnings. Retries or agent parsing are
plausible contributors, but the retained timing fields do not prove how
the residual was spent. This is a targeted candidate for request-level
instrumentation and deadline-policy pilots, not evidence of KV exhaustion.

C4 reached four running requests in 84 of 1,024 ten-second samples and
three in 247; **none** showed a waiting request. Peak sampled KV was 50.5%,
similar to C2's 52.6%. Trace warm/capture intervals numbered 162 and
summed to 66.3 seconds, small against 170.6 minutes of evaluation. Its
host `qb2-120-p05t05` did not log an AICLK clamp; the Hugging Face model
snapshot was warm and API readiness took about 4 min 54 s. Raw artifacts
and numeric summary are under `/home/mvasiljev/build/gemma-terminal-c4-main/`
and `/home/mvasiljev/build/gemma-terminal-c4-main-summary.json`.

### C4 ten-case Terminal expansion

The [matched C4/ten Terminal run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37996981703)
completed **10/10 official cases with zero case errors** on the pinned
Metal-main image and solved **4/10**. The original first five scored
**2/5** (HTML filter and COBOL); the added five scored **2/5** (password
recovery and regex log). Its generic CI accuracy gate failed because 40%
on ten selected cases is below the 44.94% full 89-case H100 target; the
ten-case score is not an estimate of that full-set score.

| Terminal/ten case | Reward | Wall min | Terminal/ten case | Reward | Wall min |
| --- | ---: | ---: | --- | ---: | ---: |
| HTML filter | 1 | 88.29 | Cancel async tasks | 0 | 4.85 |
| COBOL modernization | 1 | 41.94 | Git multibranch | 0 | 12.81 |
| CompCert | 0 | 82.80 | Password recovery | 1 | 16.59 |
| FEAL | 0 | 33.28 | Regex log | 1 | 10.55 |
| QEMU startup | 0 | 3.85 | SQLite truncate | 0 | 55.51 |

Evaluation wall was **110.26 minutes**, case clocks summed to **350.47**
minutes (3.18 observed case-clock minutes per evaluation minute), and
API/other case time summed to **266.07/84.40** minutes. The run made 151
model calls and consumed 2.162M input and 417,937 output tokens. It
reached four active requests, two waiting samples out of 671, and 51.9%
peak sampled KV (p95 33.5%); no preemption or AICLK warning appeared.
Its 155 trace warm/capture intervals summed to 63.6 seconds.

This ten-case wall was **35.4% shorter** than the earlier C4 fixed-five
wall of 170.60 minutes, despite twice as many cases. Both runs used the
same `qb2-120-p05t05` host and the same model/settings, but their sampled
agent paths differed sharply: FEAL took **33.3 versus 170.6 minutes** and
HTML **88.3 versus 168.3**, while COBOL grew from 14.7 to 41.9. The
original first five alone summed to 250.15 case-minutes in the ten-case
run versus 460.68 in the five-case run; the added five contributed 100.32
case-minutes. The ten-case run's total output (418K) was near the five-case
run's 410K, and its non-API case time fell from 213.0 to 84.4 minutes.
These changes explain why adding cases did not increase this observed wall;
they are stochastic workload differences, not a claim that C4 made ten
cases inherently faster than five. The ten-case server also had four active
requests in **276/671** samples (41.1%) versus **84/1,024** (8.2%) in
the fixed-five run, and its median positive generation sample was 88.5
versus 67.4 tokens/s. More cases kept the four agent slots occupied more
often, consistent with better batch utilization, while the changed output
and tool paths prevent assigning a precise speedup to occupancy alone.
The larger denominator improves the
reward view, while a matched C2/ten comparison is still needed for batch
selection. Raw artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-terminal-c4-ten-main/` and
`/home/mvasiljev/build/gemma-terminal-c4-ten-main-summary.json`.

### Same-main C1 control and batch-policy comparison

The [C1 fixed-five Terminal control](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37963232263)
completed all five cases with zero case errors on the **same pinned Metal
image** and identical task IDs, scorer, sampling settings and three-hour
deadline. It resolved **0/5**. COBOL, the case solved by C2/C4 and the
older September run, failed on this stochastic C1 path; one trajectory
does not establish that concurrency caused the score difference. Its generic
CI accuracy gate failed against the full-set 44.94% H100 reference, as did
the 1/5 C2 and C4 subsets.

| Same-main Terminal measure | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| Evaluation wall, min | 270.06 | 175.87 | 170.60 |
| Full test job, min | 284.52 | 182.57 | 177.08 |
| Sum of case clocks, min | 270.06 | 347.12 | 460.68 |
| Observed active-case parallelism | 1.00 | 1.97 | 2.70 |
| Solved | 0/5 | 1/5 COBOL | 1/5 COBOL |
| API calls | 124 | 126 | 142 |
| Input / output tokens | 2.378M / 442K | 2.717M / 399K | 3.091M / 410K |
| Sum API / other case time, min | 210.29 / 59.77 | 202.61 / 144.51 | 247.66 / 213.02 |
| Peak / p95 sampled KV | 38.9% / 16.1% | 52.6% / 34.4% | 50.5% / 34.3% |
| Samples with waiting | 0/1,621 | 0/1,056 | 0/1,024 |

C2's observed evaluation wall was **1.54× shorter** than C1's and C4's
**1.58× shorter**, despite their **29% and 71% longer summed case clocks**.
Their agents overlapped substantially. This is end-to-end evidence that
parallel trials can shorten a fixed suite even when individual case paths
grow; it is not a controlled per-request model-speed estimate. C4 only
improved the observed C2 wall by 3%, with FEAL setting a 170.6-minute
tail. The equal C2/C4 solved count and different C1 outcome deserve a
same-configuration repeat or larger denominator before a quality claim.

| Fixed Terminal case wall, min | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| HTML filter | 142.2 | 171.3 | 168.3 |
| COBOL modernization | 11.4 | 9.6 | 14.7 |
| CompCert | 75.3 | 120.5 | 84.3 |
| FEAL | 37.6 | 28.8 | 170.6 |
| QEMU startup | 3.6 | 17.0 | 22.7 |

The C1 request fit across 124 matched calls is 4.17 seconds/call + 0.216
seconds/1K prompt tokens + **26.23 seconds/1K output tokens** (R² 0.996),
nearly identical in output slope to September's older-Metal C1 fit of
26.28 seconds/1K. Its 2.378M input and 442K output tokens occupied
210.29 minutes of recorded API time and 59.77 minutes of other case time.
The September C1 had about 557K output tokens, 270.8 minutes of API time
and 199.7 minutes outside the API; current-main C1's 270.1-minute suite
versus September's 470.5 minutes is therefore dominated in these artifacts
by different output volume and residual work, not a demonstrated per-token
decode acceleration. The runs differ in Metal source and prefill policy,
so the fit comparison remains observational.

The C1 server reached one running request, no sampled waiting, and 38.9%
peak KV; 131 trace warm/capture intervals summed to 55.4 seconds. Its
runner `qb2-120-p05t06` showed no AICLK warning. This was a cold model
snapshot (7 min 45 s) and about 12 min 27 s to API readiness, explaining
why its full test-job wall is 14.5 minutes above evaluation wall. Raw
artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-terminal-c1-main/` and
`/home/mvasiljev/build/gemma-terminal-c1-main-summary.json`.

### Live 192K KV comparison

The [C2/192K fixed-five Terminal run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37964456985)
finished all five trials; HTML filter recorded one `AgentTimeoutError` at
180.4 minutes. It solved **1/5, COBOL**, the same case as C2/256K. The
generic accuracy gate failed because 20% is below the full-set 44.94% H100
reference, and the timeout is a real case error. No server preemption or
AICLK warning appeared in its logs. The server reached two active requests,
had only **1/1,247** ten-second samples with waiting, and peaked at 58.2%
sampled KV (p95 36.3%). Thus 192K was adequate for this live fixed-five
trajectory; the result does not establish a speed benefit or universal
capacity safety for larger case sets.

| C2 Terminal measure | 256K pinned main | 192K KV branch |
| --- | ---: | ---: |
| Evaluation wall, min | 175.87 | 207.79 |
| Sum of case clocks, min | 347.12 | 388.19 |
| Solved / case errors | 1/5 / 0 | 1/5 / 1 timeout |
| Input / output tokens | 2.717M / 399K | 3.137M / 488K |
| API / other case time, min | 202.61 / 144.51 | 272.22 / 115.97 |
| Peak / p95 sampled KV | 52.6% / 34.4% | 58.2% / 36.3% |
| Samples with waiting | 0/1,056 | 1/1,247 |

The 192K wall was **18.2% longer** while output volume was **22.3% higher**.
Its 148 matched requests fit 31.68 seconds per 1K output tokens versus
27.06 for C2/256K's 126 calls, but these trajectories and Metal images
differ; the fit cannot isolate a KV-size effect. Synthetic rates were nearly
equal at matched shapes. Keeping 256K is the current conservative choice:
192K saves cache allocation and passed this live trajectory, but has no
observed wall or reward advantage to offset its smaller admission margin.
Raw artifacts and numeric summary are at
`/home/mvasiljev/build/gemma-terminal-c2-kv192/` and
`/home/mvasiljev/build/gemma-terminal-c2-kv192-summary.json`.

### Thinking-off fixed-five Terminal pilot

The [C2 thinking-off pilot](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37991891787)
used the pinned Metal-main image and the original five cases with the
three-hour deadline. The only server policy change was
`default-chat-template-kwargs: {"enable_thinking":false}`. It completed
all five trials with zero case errors and solved **1/5, COBOL**, the same
case as C2 with thinking on. Its generic CI gate failed on the same 20%
versus full-set 44.94% H100 comparison.

| C2 fixed-five Terminal measure | Thinking on | Thinking off |
| --- | ---: | ---: |
| Evaluation wall, min | 175.87 | 37.43 |
| Sum of case clocks, min | 347.12 | 70.85 |
| Solved / case errors | 1/5 / 0 | 1/5 / 0 |
| Model API calls | 126 | 119 |
| Input / output tokens | 2.717M / 399.1K | 2.286M / 53.8K |
| Sum API / other case time, min | 202.61 / 144.51 | 52.44 / 18.41 |
| Peak sampled KV / waiting samples | 52.6% / 0 | 37.0% / 0 |

Wall fell **4.70×** and output volume **86.5%** in this single run. The
HTML and CompCert trials each still made 50 model calls, but produced only
32,625 and 11,821 output tokens versus 199,608 and 125,230 with thinking
on. COBOL remained solved in 5.71 versus 9.58 minutes. FEAL and QEMU made
fewer calls and remained unsolved. This is a meaningful output-volume and
wall reduction, not evidence of a faster decode kernel. The thinking-off
request-time fit has low R² (0.689) and should not be used for a per-token
speed claim. No AICLK warning or server preemption appeared in its logs.
Across the suite, model calls changed only from 126 to 119, while average
output per call fell from about 3,167 to 452 tokens. Average recorded API
time per call fell from 96.5 to 26.4 seconds. This supports shorter model
responses as the principal wall-time mechanism; it does not establish that
the agent performed equivalent reasoning or that a different case set
would retain its rewards.
Granite also tested a bounded 8K full-thinking response. In Gemma's C2
fixed-five Terminal trajectory, only **6/126** responses exceeded 8,192
tokens, and clipping their excess at that boundary would remove about
**2.1%** of recorded output tokens (C4: 5/139 matched steps, about 2.4%).
A 4,096-token boundary would touch 38/126 C2 calls and remove at most
17.6% of recorded output tokens, with material truncation risk. These are
arithmetic upper bounds on tokens removed from the recorded paths; a real
cap changes subsequent agent prompts and rewards. An 8K cap therefore
cannot plausibly match the observed 86.5% output reduction from
thinking-off on these trajectories, and was not dispatched as a priority.
Terminal prefill also stays substantial under thinking-off: the C2 control
had a 17,091-token median prompt, 46,830-token p90 and 62,787-token
maximum across its 126 recorded calls; thinking-off had a 20,547-token
median and 41,642-token maximum across 119 calls. That helps explain why
short-prompt C8 synthetic scaling should not be projected onto Terminal,
where per-row prefill and admission matter more.

To check whether the quality result survives a larger fixed denominator,
the [C2 ten-case thinking-off branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-ten-thinkoff)
changes only that template setting from C2/ten. Its
[combined Terminal and SWE dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37999680461)
reused one warmed server for the same ten cases in each suite. A first
dispatch with an abbreviated inference-server SHA was canceled before
hardware work; the linked dispatch pins the full commit. Score original
first-five and added-five cases separately and require the solved-case set
as well as the total wall before selecting this policy. Raw pilot artifacts
and numeric summary are under
`/home/mvasiljev/build/gemma-terminal-c2-thinkoff/` and
`/home/mvasiljev/build/gemma-terminal-c2-thinkoff-summary.json`.

### Combined ten-case thinking-off check

The [C2 thinking-off combined run](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37999680461)
completed both ten-case suites with **zero case errors** on one warmed
server. Its only policy change from C2/ten was
`default-chat-template-kwargs: {"enable_thinking":false}`; the server log
confirmed that default, and neither Gemma agent config supplied a
per-request thinking override. The model image, cases, agent count,
sampling, deadlines and official verifiers were retained. The generic CI
accuracy checks failed on the 30% ten-case subsets against the 44.94%
Terminal and 64.8% SWE full-set H100 targets.

| Terminal/ten case | Thinking-off reward | C4 thinking-on reward |
| --- | ---: | ---: |
| HTML filter | 0 | 1 |
| COBOL modernization | 1 | 1 |
| CompCert | 0 | 0 |
| FEAL | 0 | 0 |
| QEMU startup | 0 | 0 |
| Cancel async tasks | 0 | 0 |
| Git multibranch | 1 | 0 |
| Password recovery | 1 | 1 |
| Regex log | 0 | 1 |
| SQLite truncate | 0 | 0 |

Thinking-off Terminal solved **3/10** (first five 1/5, added five 2/5)
in **54.41 minutes**, with 106.85 summed case-minutes, 82.04 summed API
minutes, 24.81 other case-minutes, 3.407M input and **98,023 output
tokens**. The C4 thinking-on run solved 4/10 in 110.26 minutes with 418K
output tokens; its batch width and trajectories differ, so that comparison
does not isolate the thinking flag. The matched C2/ten thinking-on control
is still running. The five-case pilot and this expansion both show large
Terminal output reduction, but the ten-case reward is lower than C4's and
the solved-case identities differ. This policy should remain an optional
speed setting until a larger or repeated quality check shows it maintains
the required reward.

| SWE/ten measure | C2 thinking on | C2 thinking off | C4 thinking on | C8 thinking on |
| --- | ---: | ---: | ---: | ---: |
| Evaluation wall, min | 61.54 | 66.05 | 55.90 | 44.35 |
| Solved total / original five | 3/10 / 3/5 | 3/10 / 2/5 | 3/10 / 2/5 | 3/10 / 2/5 |
| Input / output tokens | 5.749M / 76.7K | 5.697M / 95.0K | 7.574M / 92.7K | 5.935M / 82.8K |
| Sum of case clocks, min | 119.58 | 115.97 | 203.20 | 284.38 |

The thinking-off SWE run solved Django 11299, Matplotlib 25332 and
requests 2317, the same three as C4/C8 thinking-on; C2 thinking-on solved
scikit-learn 14629 instead of requests. Thinking-off SWE was **7.3%
slower** and produced **23.8% more output** than its C2 thinking-on
comparison. SWE responses were already much shorter than Terminal's under
thinking-on, and the off run took a different path (389 model calls versus
403). This experiment gives no reason to enable thinking-off globally for
SWE, despite the Terminal pilot's speedup.

The shared server reached two active requests, three waiting samples out
of 726, and 42.5% peak sampled KV across the two suites; no preemption or
AICLK warning appeared. The 554 trace warm/capture intervals summed to
119.7 seconds. Its hardware test job lasted 130.3 minutes versus 120.5
minutes summed evaluation wall, so running both suites on one server paid
startup/report overhead once. Raw artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-agentic-c2-ten-thinkoff/` and
`/home/mvasiljev/build/gemma-agentic-c2-ten-thinkoff-summary.json`.

### One-hour fixed-five Terminal deadline pilot

The [C2 one-hour cap pilot](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37991526452)
used the pinned Metal-main image and changed only `agent_timeout_sec` from
10,800 to 3,600. It completed all five case records, solved **1/5,
COBOL**, and recorded two `AgentTimeoutError`s: HTML filter at 60.44
minutes and CompCert at 60.40 minutes. Both cases scored zero after
171.25 and 120.48 minutes in the three-hour C2 control, so the observed
reward on this fixed five was preserved. The generic accuracy gate still
failed against the 44.94% full-set H100 reference.

| C2 fixed-five Terminal measure | Three-hour cap | One-hour cap |
| --- | ---: | ---: |
| Evaluation wall, min | 175.87 | 102.09 |
| Sum of case clocks, min | 347.12 | 174.53 |
| Solved / timed-out cases | 1/5 / 0 | 1/5 / 2 |
| Model API calls | 126 | 78 |
| Input / output tokens | 2.717M / 399K | 1.246M / 238K |
| Sum API / other case time, min | 202.61 / 144.51 | 119.06 / 55.47 |
| Peak sampled KV / waiting samples | 52.6% / 0 | 39.8% / 0 |

Wall fell **42.0%** because long unsuccessful paths were stopped. COBOL
still passed in 9.19 versus 9.58 minutes; FEAL and QEMU remained unsolved
on different stochastic paths. A one-hour cap can lose cases that need
later work, so the five-case tie is insufficient for a default change.
The [C2 ten-case one-hour branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-ten-60m)
changes only this deadline from C2/ten. Its
[matched Terminal dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38004228585)
has started on clean runner `qb2-120-p05t02`; compare each reward and the
original/added five separately with the three-hour C2/ten run. Raw pilot
artifacts and numeric summary are under
`/home/mvasiljev/build/gemma-terminal-c2-60m/` and
`/home/mvasiljev/build/gemma-terminal-c2-60m-summary.json`.
