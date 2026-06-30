# Runner Agent Prompt Pattern

This document defines the **canonical prompt pattern** for any agent or CI job
that invokes `run.py`.  Follow it exactly — deviations are the most common
source of hung pipelines, lost output, and silent failures.

---

## Overview

Every runner invocation follows four steps:

1. **Clone** the orchestrator repo fresh into a temp dir (ensures the latest
   `main` is used without any manual sync step).
2. **Bootstrap deps** from the shared wheel cache into a per-agent local dir.
3. **Launch** `run.py` from the fresh clone, detached with `nohup`.
4. **Poll** `status.json` until the run completes.

---

## Step 0 — Clone the orchestrator repo (REQUIRED first)

Clone `tenstorrent/afuller-sandbox` into a unique temp dir so the runner
always uses the latest code from `main` and never depends on a local copy
at `/workspace/group/multi-agent/`.

```bash
ORCH_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/tenstorrent/afuller-sandbox.git "$ORCH_DIR"
echo "Cloned orchestrator to $ORCH_DIR"
```

`--depth 1` keeps the clone fast (~2 seconds). The temp dir is per-container
ephemeral, so clones from concurrent runs never interfere with each other.

---

## Step 1 — Dependency bootstrap (REQUIRED)

The orchestrator's Python dependencies (`openai` and its transitive deps) must
be installed into a **per-agent local directory** before invoking `run.py`.
Do NOT use the system `python3` directly — it won't have the packages.

A pre-downloaded wheel cache lives at `/workspace/group/pip-wheels/` (persistent,
shared, read-only at runtime). Install from it using `requirements.txt` from the
**fresh clone** (not from `/workspace/group/multi-agent/`):

```bash
pip install --quiet --no-index \
  --find-links /workspace/group/pip-wheels/ \
  --target /tmp/agent-deps/ \
  -r "$ORCH_DIR/requirements.txt"
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

# Step 0: clone orchestrator repo fresh — always use latest main, no manual sync needed
ORCH_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/tenstorrent/afuller-sandbox.git "$ORCH_DIR"
echo "Cloned orchestrator to $ORCH_DIR"

# Step 1: bootstrap deps from the wheel cache using the fresh clone's requirements.txt
pip install --quiet --no-index \
  --find-links /workspace/group/pip-wheels/ \
  --target /tmp/agent-deps/ \
  -r "$ORCH_DIR/requirements.txt"
export PYTHONPATH=/tmp/agent-deps

# Step 2: launch detached from the fresh clone — returns immediately
nohup python3 "$ORCH_DIR/run.py" --run-dir "$RUNDIR" /path/to/repo "your task here" \
    > "$RUNDIR/output.log" 2>&1 & echo $! > "$RUNDIR/orchestrator.pid"
echo "Started orchestrator PID $(cat $RUNDIR/orchestrator.pid)"

# Step 3: poll status.json until status != "running" (separate Bash call, timeout=1200000)
while true; do
  STATUS=$(python3 -c "import json; print(json.load(open('$RUNDIR/status.json'))['status'])" 2>/dev/null || echo running)
  [ "$STATUS" != "running" ] && break
  echo "Still running — $(wc -l < $RUNDIR/output.log) log lines so far"
  sleep 15
done
echo "Orchestrator finished with status: $STATUS"

# Step 4: read result
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

### 1 — Clone orchestrator fresh, every time

Run `git clone --depth 1 https://github.com/tenstorrent/afuller-sandbox.git "$ORCH_DIR"`
before anything else. This guarantees you always run the latest code from `main`
without any manual sync step. Do **not** reference `/workspace/group/multi-agent/`.

### 2 — Bootstrap deps from the fresh clone's requirements.txt

Use `-r "$ORCH_DIR/requirements.txt"` (pointing at the fresh clone), not a
hardcoded path. This ensures deps stay in sync if `requirements.txt` changes.

### 3 — Always use --run-dir

Pass `--run-dir <dir>` (or set `ORCHESTRATOR_RUN_DIR`) so that `run.py`
writes `status.json`.  Without it the polling loop has nothing to read.

### 4 — Always set PYTHONPATH

`export PYTHONPATH=/tmp/agent-deps` must be set before launching via nohup
so the detached process inherits it.

### 5 — Poll in a separate Bash call with `timeout=1200000`

The launch step returns immediately. The poll loop must be a **separate**
Bash tool call with `timeout=1200000` (20 minutes) to survive long runs.

### 6 — Expect a long runtime (15+ minutes)

The orchestrator runs multiple LLM-backed agents in a debate/consensus loop.
Do not interpret silence as hanging.

### 7 — Check status.json, not the exit code

When using nohup, read `status.json["status"]` for the outcome:
`"done"` = success, `"failed"` = failure.

---

## Full prompt template

Use this block verbatim in any agent prompt that invokes the runner.
Replace `<REPO_PATH>`, `<TASK>`, and `<RUNDIR>` with actual values.

```
You must invoke the multi-agent orchestrator to complete the requested task.

Step 0 — clone orchestrator fresh (Bash call, any timeout):

    ORCH_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/tenstorrent/afuller-sandbox.git "$ORCH_DIR"
    echo "Cloned orchestrator to $ORCH_DIR"

Step 1 — bootstrap deps (Bash call, any timeout):

    pip install --quiet --no-index \
      --find-links /workspace/group/pip-wheels/ \
      --target /tmp/agent-deps/ \
      -r "$ORCH_DIR/requirements.txt"
    export PYTHONPATH=/tmp/agent-deps

Step 2 — launch (Bash call, any timeout):

    mkdir -p <RUNDIR>
    nohup python3 "$ORCH_DIR/run.py" --run-dir <RUNDIR> <REPO_PATH> "<TASK>" \
        > <RUNDIR>/output.log 2>&1 & echo $! > <RUNDIR>/orchestrator.pid

Step 3 — poll (separate Bash call, timeout=1200000):

    while true; do
      STATUS=$(python3 -c "import json; print(json.load(open('<RUNDIR>/status.json'))['status'])" 2>/dev/null || echo running)
      [ "$STATUS" != "running" ] && break
      echo "Still running — $(wc -l < <RUNDIR>/output.log) lines so far"
      sleep 15
    done

Step 4 — read the result:

    cat <RUNDIR>/status.json
    tail -50 <RUNDIR>/output.log
```

---

## Quick-reference checklist

- [ ] Orchestrator cloned fresh: `git clone --depth 1 https://github.com/tenstorrent/afuller-sandbox.git "$ORCH_DIR"`
- [ ] Deps bootstrapped from fresh clone: `pip install --no-index --find-links /workspace/group/pip-wheels/ --target /tmp/agent-deps/ -r "$ORCH_DIR/requirements.txt"`
- [ ] `export PYTHONPATH=/tmp/agent-deps` set before launching
- [ ] `--run-dir <dir>` passed to run.py
- [ ] `run.py` invoked as `python3 "$ORCH_DIR/run.py"` (not `/workspace/group/multi-agent/run.py`)
- [ ] Launch and poll are **two separate Bash calls**
- [ ] Poll call has `timeout=1200000`
- [ ] Result read from `status.json["status"]`, not exit code

---

## Maintaining the wheel cache

The wheel cache at `/workspace/group/pip-wheels/` is written **once** during
setup and treated as read-only thereafter. To update it after a
`requirements.txt` change:

```bash
ORCH_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/tenstorrent/afuller-sandbox.git "$ORCH_DIR"
pip download --quiet -r "$ORCH_DIR/requirements.txt" \
  -d /workspace/group/pip-wheels/
```

Never run `pip install` into the cache during an active orchestrator run.

---

## Why fresh clone instead of local copy?

Previously the runner executed `run.py` directly from `/workspace/group/multi-agent/`.
This had three problems:

1. **Merged PRs didn't take effect** until someone manually ran `git pull` on
   the shared local copy — introducing an invisible delay between "merged" and
   "live".
2. **Non-deterministic versions** — the local copy's state depended on who last
   touched it, making runs hard to audit or reproduce.
3. **Parameter drift** — new CLI flags (e.g. `--max-tool-rounds` added in #25)
   weren't available to runners until the local copy was manually updated.

Fresh cloning solves all three: every run starts from the exact commit on `main`,
with no coordination or manual steps required.
