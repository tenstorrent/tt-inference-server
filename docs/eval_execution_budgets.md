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

## Relationship to the lm-eval client timeout, and why longer is worse

`reference_config/evals/eval_config.py` also sets a per-task `model_kwargs`
`timeout`. That is a client **read** timeout, not an execution budget: it does
not bound startup, dataset loading or scoring.

**Raising it is counter-productive.** `on-dispatch.yml` caps the whole job at
1080 minutes, so a stuck task at a 4h client timeout burns 4h of that budget
before it even errors, and at 12h it burns 12h -- in both cases the run is
cancelled with no verdict, and the report uploads are skipped because they sit
behind `!cancelled()`. Three Quetzal gpt-oss-120b runs have been lost exactly
this way (35299987641, 35656042558, 36374616815; the last ran 18h01m and
uploaded zero artifacts). Failing fast is what lets the remaining tasks finish
and the run produce a report at all.

The graded aime25 / gpqa pair therefore uses 7200s, matching the identical
tasks elsewhere in the file, and carries an explicit
`wall_clock_timeout_seconds=10800`. Two tasks at 3h is 6h worst case, leaving
12h of the cap for server bring-up, benchmarks and spec tests.

`wall_clock_timeout_seconds` is the real bound: it kills the task's owned POSIX
process group and reports rc=124 as an incomplete task, which is never scored
as a pass, and the run continues to the next task.
