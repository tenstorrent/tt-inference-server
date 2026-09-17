# Evaluation execution budgets

`EvalTask.wall_clock_timeout_seconds` optionally bounds one evaluation subprocess,
including startup, dataset loading, HTTP requests, retries and scoring. It is not
an idle-token timer. `EvalTask.max_attempts` sets the total HTTP attempt count
(including the first attempt) for EVALS_COMMON and takes precedence over the
device's `eval_max_retries`. Both accept positive integers only.

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
