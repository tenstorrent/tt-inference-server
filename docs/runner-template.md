# Runner Agent Prompt Pattern

This document defines the **canonical prompt pattern** for any agent or CI job
that invokes `run.py`.  Follow it exactly — deviations are the most common
source of hung pipelines, lost output, and silent failures.

---

## Canonical invocation (nohup+polling)

Always start `run.py` detached and poll `status.json` for completion.
This avoids the Bash tool's 2-minute default timeout, which kills the
orchestrator mid-run on any task that takes longer than 2 minutes.

```bash
RUNDIR=/tmp/myrun
mkdir -p "$RUNDIR"

# Step 1: launch detached — returns immediately
nohup PYTHONUNBUFFERED=1 python3 run.py --run-dir "$RUNDIR" /path/to/repo "your task here" \
    > "$RUNDIR/output.log" 2>&1 & echo $! > "$RUNDIR/orchestrator.pid"
echo "Started orchestrator PID $(cat $RUNDIR/orchestrator.pid)"

# Step 2: poll status.json until status != "running" (separate Bash call, timeout=1200000)
while true; do
  STATUS=$(python3 -c "import json; print(json.load(open('$RUNDIR/status.json'))['status'])" 2>/dev/null || echo running)
  [ "$STATUS" != "running" ] && break
  echo "Still running — $(wc -l < $RUNDIR/output.log) log lines so far"
  sleep 15
done

# Step 3: read the result
cat "$RUNDIR/status.json"
```

`status.json` written by `run.py --run-dir`:

```json
{"status": "running"}                          // written at startup
{"status": "done"}                             // written on success
{"status": "failed", "error": "<message>"}    // written on failure
```

---

## Mandatory rules

### 1 — Always use --run-dir

Pass `--run-dir <dir>` (or set `ORCHESTRATOR_RUN_DIR`) so that `run.py`
writes `status.json`.  Without it the polling loop has nothing to read and
cannot know when the process finishes.

### 2 — Always set `PYTHONUNBUFFERED=1`

Python buffers stdout/stderr when not connected to a TTY. Without this:

- Progress logs may not appear until exit (or never, on a crash).
- CI inactivity watchdogs may kill the job because no output is seen.

### 3 — Poll in a separate Bash call with `timeout=1200000`

The launch step returns immediately (nohup backgrounds the process). The
poll loop must be a **separate** Bash tool call with `timeout=1200000`
(20 minutes) to survive long runs.

### 4 — Expect a long runtime (15+ minutes)

The orchestrator runs multiple LLM-backed agents in a debate/consensus loop.
A single run routinely takes 15 minutes or more. Do not interpret silence
as hanging — agents may spend several minutes waiting for model responses.

### 5 — Check status.json, not the exit code

When using nohup, you cannot read the process exit code. Read
`status.json["status"]` instead: `"done"` means success, `"failed"` means
failure.

---

## Full prompt template

Use this block verbatim in any agent prompt that invokes the runner.
Replace `<REPO_PATH>`, `<TASK>`, and `<RUNDIR>` with actual values.

```
You must invoke the multi-agent orchestrator to complete the requested task.

Step 1 — launch (Bash call, any timeout):

    mkdir -p <RUNDIR>
    nohup PYTHONUNBUFFERED=1 python3 run.py --run-dir <RUNDIR> <REPO_PATH> "<TASK>" \
        > <RUNDIR>/output.log 2>&1 & echo $! > <RUNDIR>/orchestrator.pid

Step 2 — poll (separate Bash call, timeout=1200000):

    while true; do
      STATUS=$(python3 -c "import json; print(json.load(open('<RUNDIR>/status.json'))['status'])" 2>/dev/null || echo running)
      [ "$STATUS" != "running" ] && break
      echo "Still running — $(wc -l < <RUNDIR>/output.log) lines so far"
      sleep 15
    done

Step 3 — read the result:

    cat <RUNDIR>/status.json
    tail -50 <RUNDIR>/output.log
```

---

## Quick-reference checklist

- [ ] `--run-dir <dir>` passed (or `ORCHESTRATOR_RUN_DIR` set)
- [ ] `PYTHONUNBUFFERED=1` set as env-var prefix
- [ ] Launch and poll are **two separate Bash calls**
- [ ] Poll call has `timeout=1200000`
- [ ] Result read from `status.json["status"]`, not exit code
