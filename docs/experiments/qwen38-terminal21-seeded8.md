# Prepared Terminal-Bench 2.1 seeded8

Ready for one CI dispatch after both independent gates passed. The first
p2048 smoke40687173740 failed during1024-chunk warmup before agents (OOM),
so it qualifies no task result. Metal126fde8d56bcb53853858f673a0199a5d585acc4
bounds physical prefill chunks to512 while retaining logical p2048 checkpoints
and native262144 context. Its immutable source-overlay image digest is
0611f1029baae3c347440a84f511ec8d156b1d3cb953e0471324910e8b17388d.
Local all64/B32/512 state/logit/continuation checks passed with all27 source
hashes matching. Actual-pool smoke40691014802 passed its original Astropy task,
10HTTP tool calls,4 captures,512 physical chunks and771 scheduler IDs. These
are separate correctness and real-pool-fit gates; neither is a TB accuracy
measurement. The first TB2.1
harness pilot40685950043 passed both short tasks, all8 original verifier tests,
and11/11 exact whole-turn prefix continuations on B32/p256. It generated6261
tokens without cap/timeout; this is not population accuracy.

Freeze seed1234 shuffle of sorted89 task names from pinned TB2.1 revision6;
use the first8, selected before the short pilot's outcomes. Exact task content
hashes are in qwen38_terminal21_seeded8.json. All8 attempts remain in the score,
including wrong answers and timeouts. No favorable task substitution.

Use the settled combined SWE serving implementation, original BF4 projection
precision, grouped/masked recurrence, flat prefill, B32/p2048, native262144
context, same23661297664-byte device cache budget and Sampling1D. C8 increases
load from the validated C2 while keeping expected working set below the
p2048 source screen; actual task histories may differ and must be measured.
The source screen1.118M virtual token IDs is an upper capacity estimate, not
an observed allocation guarantee. Do not multiply every agent's maximum context
and call it observed demand. Keep the idle-server/occupied-port guard.

Retain16K per-response output and medium reasoning: the short successful
pilot maxed at1815 tokens and does not justify changing output distribution.
Allow3600s requests (16K at pilot ~11tokens/s fits with slack),10800s agents
and12600s suite, one wave. Historical hard tasks were censored at5400s; one
completion-capable sample is preferable to repeating short timeout ladders.
These are diagnostic allowances, not guarantees that every hard task finishes.
If length/all-reasoning or JSON failures occur, report them separately from
agent deadline, model wrong answer, command/build failure and infrastructure.

Explicitly retain Harbor reasoning and preserve thinking in the template.
Record exact prompt+generated prefix continuity, block-aligned cache ceiling,
request lengths/times, preemption/working-set evidence, repetition and original
verifiers. p2048 has larger unavoidable partial-block loss than p256; separate
that source-predicted loss from any unexplained whole-block loss. No causal
speed/accuracy gain can be established from this one mixed-workload sample.
