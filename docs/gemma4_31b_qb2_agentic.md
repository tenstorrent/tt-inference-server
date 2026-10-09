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
  historical outcomes, not scores for current Metal main.
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
  cannot establish a safe smaller pool under higher concurrency.
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
Mini-SWE-Agent trajectories recorded 32–50 API calls per case, but their
archived summary lacks per-case wall and token timings.

## Current-main baseline setup

- Metal main pinned at `2c1e1ebdd638886821f35113a5fd0d6335d71608`.
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
- The corrected [Terminal 2.0 baseline](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37955350355)
  passes Shield's server-type resolution and is building its current-main
  image. The SWE baseline will use that image to avoid a second build. Shield
  invokes `run.py --dev-mode`, which passes the checked-out branch's model spec
  into Docker and mounts its source/config directories; this permits C1/C2/C4
  catalog experiments on the same Metal-main image without silently reusing
  the image's baked C1 catalog. The Metal KV-hook candidate needs a new image.

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
   complete fixed-case denominator and resource/score review.
2. [Context-sized KV hook](https://github.com/tenstorrent/tt-metal/tree/mvasiljevic/gemma4-31b-agentic-kv)
   and [C2/192K catalog](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-kv192):
   keep the physical batch at two and size the base full-attention pool from
   `max_model_len=196608`. The previous full-context 262144 setting returns
   the same pool as main. The hook requests 355,584 tokens at C2/256K and
   290,048 at C2/192K, including sliding-window and in-flight headroom, before
   the plugin's additional per-request output pages. The 64K/16K Terminal
   request budgets fit below 192K.
   The candidate requires device proof: startup/allocation, high page IDs,
   long prefill/decode, 1- and 2-request quality checks, and an agentic run
   without KV preemption before it can be selected. One host-side budget test
   and Ruff/pre-commit pass; no device claim is made yet. Its
   [benchmark dispatch](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/37956805379)
   is building a separate image from Metal `6b70d1267a5a16410d5aa5c7ec8e851b7ad47f94`
   and inference-server `84c126298977df6f3d90ff2ea1313c7a255ec4c8`.
3. [C2 thinking-off pilot](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-c2-thinkoff):
   changes only the server's `enable_thinking` default. Granite's fixed-case
   pilot reduced output volume markedly, and Gemma's archived latency fit
   makes output reduction promising. Run the same five cases and compare
   rewards, token counts, per-case clocks and API calls; treat it as a distinct
   quality policy, not a reference-equivalent speed result. Do not select it
   solely on a faster wall clock or a five-case score tie.

Use the same five case IDs and official verifier rewards for all timing
comparisons. Record completed/errored/cancelled counts, per-case wall and model
API time, token counts, request-running/waiting samples, KV occupancy,
preemptions, trace recapture count and server identity. Increase to ten cases
only after the five-case results preserve at least the baseline's solved cases
and show a useful wall-time gain. The full-set H100 targets remain recorded;
do not turn a missing reward or smaller denominator into a passing score.

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

For a downloaded Actions artifact directory, run
`python3 scripts/gemma4_agentic_summary.py ARTIFACT_DIR --output summary.json`.
The script emits only case IDs, numeric timings/tokens/rewards and server
counts; it omits prompts, patches, shell transcripts and model responses. Its
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
