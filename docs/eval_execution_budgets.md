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

## Detecting a dead server, versus bounding a live one

Two different mechanisms, and they are easy to confuse.

`model_kwargs["timeout"]` is a socket **read** timeout. For a task with
`"stream": "true"` it resets on every chunk, so it is an **idle** bound: it
cannot truncate a healthy generation, which emits a token every few hundred
milliseconds, and it fires only when the server has stopped producing.

That makes a large value actively harmful. At 7200 a dead server goes
undetected for two hours; at 14400, four. `on-dispatch.yml` caps the job at
1080 minutes, so a couple of stalls consume the budget and the run is cancelled
with no verdict -- and the report uploads are skipped because they sit behind
`!cancelled()`. Three Quetzal gpt-oss-120b runs were lost this way
(35299987641, 35656042558, 36374616815; the last ran 18h01m and uploaded zero
artifacts). The graded `aime25` / `gpqa` pair therefore uses **600s**: a stall
is caught in ten minutes and the remaining tasks keep their budget.

**A non-streaming task must not copy that value.** With `"stream": "false"`
(the `EvalTask` default) the same field bounds the entire response, so a short
value truncates legitimate long generations rather than detecting failure.
Check `gen_kwargs["stream"]` before changing any `timeout`.

`wall_clock_timeout_seconds` is the backstop for the case the idle bound cannot
see: tokens trickling just fast enough to keep resetting the read timeout. It
kills the task's owned POSIX process group and reports rc=124 as an incomplete
task, which is never scored as a pass, and the run continues to the next task.
