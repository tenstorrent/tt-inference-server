# Runner Agent Prompt Pattern

This document defines the **canonical prompt pattern** for any agent or CI job
that invokes `run.py`.  Follow it exactly — deviations are the most common
source of hung pipelines, lost output, and silent failures.

---

## Dependency bootstrap (REQUIRED at the start of every runner agent)

The orchestrator's Python dependencies (`openai` and its transitive deps) must
be installed into a **per-agent local directory** before invoking `run.py`.
Do NOT use the system `python3` directly — it won't have the packages.

A pre-downloaded wheel cache lives at `/workspace/group/pip-wheels/` (persistent,
shared, read-only at runtime). Install from it into `/tmp/agent-deps/` at the
start of every runner agent:

```bash
pip install --quiet --no-index \
  --find-links /workspace/group/pip-wheels/ \
  --target /tmp/agent-deps/ \
  -r /workspace/group/multi-agent/requirements.txt
export PYTHONPATH=/tmp/agent-deps
```

This takes ~1-2 seconds, gives each agent its own isolated copy of the deps,
and requires no network access. `/tmp/` is per-container ephemeral so agents
can never interfere with each other.

---

## Canonical invocation (nohup+polling)

Always start `run.py` detached and poll `status.json` for completion.
This avoids the Bash tool's 2-minute default timeout, which kills the
orchestrator mid-run on any task that takes longer than 2 minutes.

```bash
RUNDIR=/tmp/myrun
mkdir -p "$RUNDIR"

# Step 0: bootstrap deps (see above — must come first)
pip install --quiet --no-index \
  --find-links /workspace/group/pip-wheels/ \
  --target /tmp/agent-deps/ \
  -r /workspace/group/multi-agent/requirements.txt
export PYTHONPATH=/tmp/agent-deps

# Step 1: launch detached — returns immediately
nohup python3 /workspace/group/multi-agent/run.py --run-dir "$RUNDIR" /path/to/repo "your task here" \
    > "$RUNDIR/output.log" 2>&1 & echo $! > "$RUNDIR/orchestrator.pid"
echo "Started orchestrator PID $(cat $RUNDIR/orchestrator.pid)"

# Step 2: poll status.json until status != "running" (separate Bash call, timeout=1200000)
while true; do
  STATUS=$(python3 -c "import json; print(json.load(open('$RUNDIR/status.json'))['status'])" 2>/dev/null || echo running)
  [ "$STATUS" != "running" ] && break
  echo "Still running — $(wc -l < $RUNDIR/output.log) log lines so far"
  sleep 15
done
echo "Orchestrator finished with status: $STATUS"

# Step 3: read result
cat "$RUNDIR/status.json"
tail -50 "$RUNDIR/output.log"
```

`status.json` written by `run.py --run-dir`:

```json
{"status": "running"}                          // written at startup
{"status": "done"}                             // written on success
{"status": "failed", "error": "<message>"}    // written on failure
```

---

## Mandatory rules

### 1 — Bootstrap deps first, every time

Run the pip install step before anything else. `/tmp/agent-deps/` is
per-container ephemeral and starts empty — skipping this step causes
`ModuleNotFoundError: No module named 'openai'`.

### 2 — Always use --run-dir

Pass `--run-dir <dir>` (or set `ORCHESTRATOR_RUN_DIR`) so that `run.py`
writes `status.json`.  Without it the polling loop has nothing to read.

### 3 — Always set PYTHONPATH

`export PYTHONPATH=/tmp/agent-deps` must be set before launching via nohup
so the detached process inherits it.

### 4 — Poll in a separate Bash call with `timeout=1200000`

The launch step returns immediately. The poll loop must be a **separate**
Bash tool call with `timeout=1200000` (20 minutes) to survive long runs.

### 5 — Expect a long runtime (15+ minutes)

The orchestrator runs multiple LLM-backed agents in a debate/consensus loop.
Do not interpret silence as hanging.

### 6 — Check status.json, not the exit code

When using nohup, read `status.json["status"]` for the outcome:
`"done"` = success, `"failed"` = failure.

---

## Full prompt template

Use this block verbatim in any agent prompt that invokes the runner.
Replace `<REPO_PATH>`, `<TASK>`, and `<RUNDIR>` with actual values.

```
You must invoke the multi-agent orchestrator to complete the requested task.

Step 0 — bootstrap deps (Bash call, any timeout):

    pip install --quiet --no-index \
      --find-links /workspace/group/pip-wheels/ \
      --target /tmp/agent-deps/ \
      -r /workspace/group/multi-agent/requirements.txt
    export PYTHONPATH=/tmp/agent-deps

Step 1 — launch (Bash call, any timeout):

    mkdir -p <RUNDIR>
    nohup python3 /workspace/group/multi-agent/run.py --run-dir <RUNDIR> <REPO_PATH> "<TASK>" \
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

- [ ] Deps bootstrapped: `pip install --no-index --find-links /workspace/group/pip-wheels/ --target /tmp/agent-deps/`
- [ ] `export PYTHONPATH=/tmp/agent-deps` set before launching
- [ ] `--run-dir <dir>` passed to run.py
- [ ] Launch and poll are **two separate Bash calls**
- [ ] Poll call has `timeout=1200000`
- [ ] Result read from `status.json["status"]`, not exit code

---

## Maintaining the wheel cache

The wheel cache at `/workspace/group/pip-wheels/` is written **once** during
setup and treated as read-only thereafter. To update it after a
`requirements.txt` change:

```bash
pip download --quiet -r /workspace/group/multi-agent/requirements.txt \
  -d /workspace/group/pip-wheels/
```

Never run `pip install` into the cache during an active orchestrator run.
