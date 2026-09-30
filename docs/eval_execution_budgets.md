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

This addresses unbounded harness execution, not the cause of long model answers.
The motivating GPT Shield run's apparent retry exhaustion remains an inference
until its client log is available. No running jobs are modified by this patch.

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

With `max_gen_toks: 120 * 1024` and a ~405 ms/token TPOT, a full generation is
roughly 13.8 hours. Every value proposed for these tasks -- 14400, 7200, 600,
and the harness default of 1800 (~4,450 tokens) -- is below that, so these
tasks have been exposed to truncation at all of them. `timeout` is therefore
left at its existing value here; moving it is a guess in either direction until
`max_gen_toks` is clamped to what the device context actually supports, which is
what would make a per-request budget computable. That clamp is a separate change.

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
