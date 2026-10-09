# Gemma 4 31B QB2 agentic evaluation and serving experiments

Updated 2026-10-09 UTC. This file tracks the dedicated `gemma4-31b-qb2`
implementation on the four-chip P300X2 QuietBox. Results from different model
commits, task lists, or agent policies are kept separate.

## Existing evidence

- The [September 21 baseline](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/35667712636)
  used an older Metal implementation and the fixed first five Terminal-Bench 2.0
  and SWE-bench Verified cases. It scored **1/5** on each. The Terminal mean was
  5,646.8 seconds per case, with 4.61 million input and 557 thousand output
  tokens across the five cases. These are historical outcomes, not scores for
  current Metal main.
- That Terminal server logged 409 model-trace warm/capture cycles. Matched
  warm-to-capture intervals sum to 119.8 seconds (median 0.157 seconds, p95
  0.440 seconds), so trace recapture alone is a small part of its multi-hour
  evaluation. The first warmup was much slower and cold startup must
  be kept separate. Its 3,545 roughly ten-second KV samples reached 28.7%
  maximum and 19.7% p95; no sample showed a waiting request. These samples
  cannot establish a safe smaller pool under higher concurrency.
- The dedicated model's [README](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/gemma4_31b_qb2/README.md)
  supports batch 1–32. Its measured 128/128 case completed 0.357 requests/s at
  batch 1 and 5.847 requests/s with 32 concurrent requests. At 1024/128,
  batch-32 per-user decode dropped to 21.00 tokens/s from batch-1's 44.27.
  Agentic prompts are much longer, so those burst measurements do not choose
  the best agent concurrency.
- Granite's [fixed-five timing analysis](https://github.com/tenstorrent/tt-metal/blob/533b81b7bbd/models/autoports/ibm_granite_granite_4_2_30b/doc/agentic_evals/TERMINAL_BENCH_TIME_BREAKDOWN.md)
  measured 79.3% of accumulated agent execution inside model API calls at C2;
  other agent work overlaps another case's inference. Its thinking-off pilot
  produced fewer output tokens and shorter suite time on one stochastic
  trajectory at unchanged task/scorer, while expanded KV reduced sampled
  waiting at C10. Neither effect transfers automatically to Gemma.

## Current-main baseline setup

- Metal main pinned at `2c1e1ebdd638886821f35113a5fd0d6335d71608`.
- The inference-server [baseline branch](https://github.com/tenstorrent/tt-inference-server/tree/mvasiljevic/gemma4-31b-agentic-baseline)
  adds tool-call parsing to the dedicated dev catalog entry and restores the
  original five fixed Terminal-Bench 2.0 and SWE-bench Verified cases. It keeps
  server capacity 1, one concurrent agent, thinking enabled, temperature 1.0,
  top-p 0.95, top-k 20 and the earlier bounded output settings. It retains the
  recorded full-set H100 targets (Terminal 44.94%, SWE 64.80%) for context;
  five-case percentages are discrete diagnostics and the smaller request
  budgets prevent direct full-set H100 comparison.
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
  image. The SWE baseline will use that image to avoid a second build.

## Candidates and acceptance

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
   and Ruff/pre-commit pass; no device claim is made yet.

Use the same five case IDs and official verifier rewards for all timing
comparisons. Record completed/errored/cancelled counts, per-case wall and model
API time, token counts, request-running/waiting samples, KV occupancy,
preemptions, trace recapture count and server identity. Increase to ten cases
only after the five-case results preserve at least the baseline's solved cases
and show a useful wall-time gain. The full-set H100 targets remain recorded;
do not turn a missing reward or smaller denominator into a passing score.

The current model uses one physical decode batch equal to the server's
`max_num_seqs`, warmed at startup. It does not switch physical batch sizes
while serving. A new prefill signature can retire and recapture that trace;
the old run's measured recapture time makes this a lower priority than
decoding, task parallelism and KV admission. Any future dynamic-bucket change
must retain warmed traces for each bucket and verify cache/table identity
before replay.
