# Runner Agent Prompt Pattern

This document defines the **canonical prompt pattern** for any agent or CI job
that invokes `run.py`.  Follow it exactly — deviations are the most common
source of hung pipelines, lost output, and silent failures.

---

## Canonical invocation

```bash
PYTHONUNBUFFERED=1 python run.py <repo_path> "<task description>"
```

With an explicit API key (avoid in CI — prefer the env-var or key file):

```bash
PYTHONUNBUFFERED=1 python run.py --api-key "$TT_CHAT_API_KEY" <repo_path> "<task description>"
```

---

## Mandatory rules

### 1 — Always run synchronously; never background the process

`python run.py` **must** be called as a plain, foreground command.

| ❌ Forbidden | Why |
|---|---|
| `python run.py … &` | Backgrounds the process; the caller returns immediately while the orchestrator is still running. Any subsequent steps that depend on the result (exit code, PR URL, artefacts) will race or silently read stale state. |
| `nohup python run.py … &` | Same problem, compounded: output is redirected to `nohup.out` and the process is detached from the session entirely, making it impossible to wait on or stream logs from. |
| `disown`, `setsid`, `screen -dm`, or any other daemonisation technique | All share the same root cause — the caller no longer blocks on the child. |

**The call must block until `run.py` returns.**  Capture the exit code and act
on it:

```bash
PYTHONUNBUFFERED=1 python run.py /path/to/repo "your task here"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Orchestrator failed (exit $EXIT_CODE)" >&2
  exit $EXIT_CODE
fi
```

### 2 — Expect a long runtime (15+ minutes)

The orchestrator runs multiple LLM-backed agents in a debate/consensus loop
(implementation → review → rebuttal → re-review → PR).  **A single run
routinely takes 15 minutes or more**, and complex tasks can take significantly
longer.

- **Do not set a short timeout** (e.g. 30 s or 2 min) around the call.
- **Do not interpret silence as hanging** — agents may spend several minutes
  thinking or waiting for model responses without printing anything.
- If your shell or CI system has an inactivity timeout, ensure `PYTHONUNBUFFERED=1`
  (see below) is set so that log lines flush immediately and keep the connection
  alive.

### 3 — Always set `PYTHONUNBUFFERED=1`

Python buffers stdout/stderr when they are not connected to a TTY (the normal
case in CI, subprocesses, and agent tool calls).  Without this variable:

- Progress logs may not appear until the process exits (or never, if it crashes).
- CI inactivity watchdogs may kill the job because no output is seen for minutes.
- Debugging a failed run becomes extremely difficult.

Set it as a prefix on the command line:

```bash
PYTHONUNBUFFERED=1 python run.py …
```

Or export it in the environment before the call:

```bash
export PYTHONUNBUFFERED=1
python run.py …
```

---

## Full prompt template

Use the following block verbatim as the system or user prompt when an agent is
responsible for invoking the runner.  Replace `<REPO_PATH>` and `<TASK>` with
the actual values.

```
You must invoke the multi-agent orchestrator to complete the requested task.

Use the following command EXACTLY — do not modify the environment prefix,
do not add & or nohup, and do not alter the argument order:

    PYTHONUNBUFFERED=1 python run.py <REPO_PATH> "<TASK>"

Critical constraints:
- Run the command synchronously (foreground). Never append & or nohup.
- The command MUST block until the process exits on its own.
- The process may take 15 minutes or longer to complete. This is normal.
  Do not interrupt it, do not assume it has hung, and do not impose a
  short timeout.
- PYTHONUNBUFFERED=1 must be present so that logs stream in real time.
- Check the exit code after the process returns:
    0 -> success (PR has been opened)
    non-zero -> failure (review the output for the reason)
```

---

## Quick-reference checklist

Before triggering any runner invocation, verify every item:

- [ ] Command is foreground — no `&`, `nohup`, `disown`, or equivalent
- [ ] `PYTHONUNBUFFERED=1` is set as an env-var prefix or exported before the call
- [ ] No timeout shorter than 30 minutes wraps the call
- [ ] Exit code is checked after `run.py` returns
- [ ] Output (stdout + stderr) is captured or streamed for debugging

---

## Background: why these constraints exist

`run.py` orchestrates several LLM agent turns in sequence:

1. **Implementer** — explores the repo with tool calls and makes the change
2. **Reviewers** (security + correctness) — audit the diff in parallel
3. **Debate rounds** (up to `max_debate_rounds`, default 3) — implementer
   addresses objections, reviewers re-evaluate
4. **PR creation** — pushes a branch and opens a GitHub pull request

Each LLM call can take 10–60 seconds, and the full loop compounds them.
The process is entirely sequential and stateful: intermediate results live in
in-process Python objects, not on disk.  Backgrounding the process with `&`
means the parent shell (or agent) has no way to know when it finishes, cannot
read the exit code, and may start subsequent steps before the PR exists.
Running with `PYTHONUNBUFFERED=1` ensures every `print()` call inside the
orchestrator reaches your log stream the moment it is emitted.
