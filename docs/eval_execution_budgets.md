# Evaluation execution budgets

`EvalTask.wall_clock_timeout_seconds` optionally bounds one evaluation subprocess,
including startup, dataset loading, HTTP requests, retries and scoring. It is not
an idle-token timer. `EvalTask.max_attempts` sets the total HTTP attempt count
(including the first attempt) for EVALS_COMMON and takes precedence over the
device's `eval_max_retries`. It is converted before emission: lm-eval's
`max_retries` counts only the attempts after the first, so `max_attempts=N`
emits `max_retries=N-1`. Both accept positive integers only.

Example policy for a bounded diagnostic (not a new qualification default):

```python
EvalTask(task_name="aime25", wall_clock_timeout_seconds=3600, max_attempts=1)
```

No existing task budget is changed by this patch. Owners must explicitly select
budgets appropriate to the task/device and review the resulting completion rate.
These fields do not change output tokens, context, precision, samples or scoring.

On deadline, TTIS kills the evaluation's owned POSIX process group (not the
inference server), retains files already emitted, returns 124, and reports an
incomplete failed task even if partial results exist. The workflow returns a
nonzero evaluation outcome, independently of experimental accuracy waivers.
Buffered results not yet written by the harness cannot be recovered by this
wrapper. Children that deliberately escape the process group are outside this
contract. Existing calls without a budget retain their previous execution path.

## GPT-OSS-120B p300x2 measured budgets

Shield runs `37097968441` (eight clients) and `37104338100` (four clients)
replaced estimates with full-run measurements on the Quetzal p300x2 package.
The four-client run reached only 24/30 AIME samples in three hours, then GPQA
also reached its three-hour deadline. The eight-client run completed AIME in
about 2h46m, but a 900-second read-idle limit converted six requests into
partial/error sentinels; GPQA processed 48/198 samples in 2h15m before its
three-hour deadline. Server telemetry in the four-client run showed a healthy
request remain preempted for about 18 minutes before decode resumed.

The reviewed retry therefore uses eight clients, a 1800-second streamed read
idle limit, a four-hour AIME task budget, and a ten-hour GPQA task budget. The
official high-reasoning tasks retain their 120K generation allowance. MMLU is
the low-reasoning path and uses a 32K generation cap, matching the GPU-reference
command recorded in issue #1322. The workflow-level 18-hour limit remains the
final bound. These values are evidence from this package and device; they are
not defaults for other models or hardware.

This addresses unbounded harness execution and measured scheduler starvation;
it does not change why the model produces long answers. Both cited Shield jobs
were terminal before these values were selected, so no running job was changed.

## Why `model_kwargs["timeout"]` is not a stall detector

It is tempting to shorten `model_kwargs["timeout"]` so a dead server is noticed
quickly. That does not work, and the reason is worth recording because it was
got wrong twice in review.

In the pinned EVALS_COMMON harness the client session is built as:

```python
# lm_eval/models/api_models.py:808
connector=conn, timeout=ClientTimeout(total=self.timeout)
```

`total=` is a **whole-request** budget covering connect, send and reading the
entire response body. It is the only `ClientTimeout` construction in the
package, and there is no `sock_read` or `sock_connect` knob anywhere in it.
Setting `"stream": "true"` does **not** turn it into a per-chunk idle timer --
the streamed body is read inside the same bounded operation.

The consequence is that a single `total=` budget cannot satisfy both
requirements at once:

- detect a dead server within minutes, and
- not truncate a legitimate long generation.

How bad that is depends on decode speed, so state it as a **break-even
threshold** rather than a duration. Both graded tasks set
`max_gen_toks: 120 * 1024` = **122,880 tokens**, so a full-length generation
fits inside a budget `T` only if decode sustains `122880 / T`:

| `timeout` | required rate | required per-token |
|---|---|---|
| 600 s | 204.8 tok/s | 4.9 ms |
| 1800 s (harness default) | 68.3 tok/s | 14.6 ms |
| 7200 s | 17.1 tok/s | 58.6 ms |
| 14400 s (current) | **8.53 tok/s** | **117.2 ms** |

The nearest traced, measured rate for `openai/gpt-oss-120b` -- the model these
tasks target -- is **6.0 tok/s steady (~167 ms/token)**, with 2.1 tok/s
end-to-end on short requests, from the 2026-09-05 device run recorded in
tt-quetzalcoatlus `docs/GPT_OSS_120B_DECODE_PERF.md` (p300x2, batch=1, trace
on). At 6.0 tok/s a full generation takes ~5.7 h; at 2.1 tok/s, ~16.3 h. The
break-even for 14400 s therefore sits about 42% above the fastest rate measured
on this model, and every value proposed for this field is on the truncating side
of it.

Two caveats, stated because the alternative is a number nobody can source:

- **No TPOT has been measured under the actual eval condition** --
  `max_concurrent=32` with 122k-token generations on the device the graded runs
  use. The 6.0 tok/s sample is batch=1 on two chips with 128-token requests. A
  sustained >= 8.53 tok/s on the real path would falsify the concern above, and
  that is a measurable question rather than a matter of argument.
- An earlier revision of this document claimed ~405 ms/token and ~13.8 h. That
  figure came from a serve with `QUETZAL_NO_TRACE=1` force-set (eager, roughly
  7x slower than traced) and is withdrawn. For contrast, a traced 96.1 ms/token
  (10.4 tok/s) has been measured -- but on Qwen3.6-27B on tt-quietbox
  (`serving/verified_runs/20260817/collectives_sweep/`), a different model on a
  different box, so it is a reference point and not evidence about these tasks.
  At that rate the same 122,880 tokens take 3.28 h and would fit.

`timeout` is therefore left at its existing value; moving it is a guess in
either direction until `max_gen_toks` is clamped to what the device context
actually supports, which is what would make a per-request budget computable.
That clamp is a separate change.

## What does bound a hung run

`wall_clock_timeout_seconds` bounds the **task**, not the request. That is the
distinction that matters: it can detect a hung evaluation without capping any
individual generation, because it does not care how long one response takes --
only how long the whole subprocess has been running.

On deadline it kills the task's owned POSIX process group, retains files already
written, and reports rc=124 as an incomplete task which is never scored as a
pass; the run then continues to the next task. Without it, a stalled task runs
until `on-dispatch.yml`'s 1080-minute cap cancels the job, and the report uploads
are skipped because they sit behind `!cancelled()` -- which is how three Quetzal
gpt-oss-120b runs were lost (35299987641, 35656042558, 36374616815; the last ran
18h01m and uploaded zero artifacts).

Note that this field only takes effect because of the wiring fix in this change:
it was previously declared, validated and documented but never passed to
`run_command`, so no budget set anywhere had any effect.
