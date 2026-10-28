# Terminal-Bench 2.1 accuracy and whole-turn prefix pilot

Two tasks (fix-git, openssl-selfsigned-cert), C2, one attempt, seed1234.
Exact content hashes are frozen in qwen38_terminal21_accuracy2.json from
Terminal-Bench2.1 revision6, population89; no TB2.0 substitution.

Use the combined SWE serving implementation from c2b67932: Metal6faa07f,
plugin1badbd02, original precision, grouped/masked recurrence and flat prefill,
resident32, block256, 23661297664-byte per-device cache. This first pilot
qualifies JSON harness and per-turn continuity, not concurrency performance.
The two short file/git tasks avoid long external builds and permit manual
artifact/verifier validation before expanding task count.

Offline native-template replay found that dropping reasoning loses the prior
reply's token prefix even though prior-input LCP remains near100%. Explicitly
retain reasoning in Harbor history AND preserve_thinking in the template.
Middleware now counts exact previous-input-plus-generated-token LCP, preserving
legacy prior-input metrics. Prompt and generated arrays share the existing
bounded state/token budget; only counts and hashes enter audit logs. Missing
generated IDs remain explicitly unavailable, not guessed.

Keep16K output initially to observe real tool JSON completion/cap incidence.
Prior calibration showed the difficult pair still censored at5400s; B32
full16K generation can exceed1800s. Use3600s request,5400s agent and6600s
suite to allow completion, with explicit classification if any bound fires.
These are diagnostic allowances, not a speed or full-suite forecast. Derive
next bounds from actual response lengths/turn times, never all-timeout speed.

Next: inspect original task verifiers, normal termination, every response
finish reason/JSON parse, repeated commands and exact whole-turn prefix.
Separate client rendering losses from cache losses. Bin cache losses by prior
reply length and time away; join serving preemption/allocator evidence before
assigning eviction. Expand task population and client concurrency only after
this small harness run completes; calculate working-set memory from observed
prompt distribution and source geometry first.

## Measured two-task result

Run40685950043 on qb2-120-p05t02: both original verifiers passed (2 Git tests,
6 certificate tests), both agents finished normally. Git used9 calls/284.14s,
certificate4 calls/382.89s; the scored suite lasted about402s after task startup.
There were13 successful HTTP200 stop responses,6261 output tokens, maximum1815
output tokens/request,191.50s/request and8042 input tokens. No truncation,
summarization, output cap, repetition stop, request error or preemption occurred.

All11 returning turns preserved the exact prior input PLUS generated IDs.
55296/56486 shared tokens were cached (97.893%); every hit was exactly the
shared whole-turn length rounded down to256, so the1190 uncached shared tokens
are explained by partial checkpoint blocks. No unexplained prefix loss was
observed at this low load. Captures stayed4, retirements and host-logit reads0.
Repeated Git inspection/merge commands were productive conflict/editor repair;
both agents' final confirmation calls are the normal Terminus completion path.
This is a harness/continuity proof, not population accuracy or a scored causal
comparison with dropped reasoning. Longer-reply and larger working-set behavior
remain unqualified.

The agentic workflow returned0 and artifacts were collected before server
shutdown. The optional AI-summary action subsequently logged a uv-cache
permission error (exit2), and the run_specs display found no files (exit1).
Those administrative errors did not alter the preserved task verifiers or
model evidence. The hardware job was marked successful. The earlier dispatch
40685813003 failed before device allocation because a short source SHA was
mistakenly treated as a branch; the identical run was redispatched with the
full40-hex source SHA and all attempts are retained in local records.
