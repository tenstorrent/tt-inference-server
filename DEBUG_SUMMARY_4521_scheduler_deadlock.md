# #4521 debug summary: root cause found — scheduler deadlock, not a device-read stall

This supersedes the FD-queue attribution in
[`HANDOFF_conc32_device_read_stall.md`](HANDOFF_conc32_device_read_stall.md) and the
follow-up overnight log in `HANDOFF_4521_OVERNIGHT_2026-07-15.md`. Both are kept as-is
for the raw investigation trail; this doc is the condensed, final narrative.

**TL;DR:** the hang is not a tt-metal/device-level race at all. It is a pure Python
deadlock in vLLM's TT-custom scheduler (`vllm_tt.scheduler.AscendScheduler`, in the
tt-xla repo). A specific token-budget bookkeeping bug permanently starves decode
scheduling for every running request, with zero device activity — which is exactly why
gdb kept showing a thread idling in `FDMeshCommandQueue::read_completion_queue`: that
thread is healthy and just waiting for new work that will never come, not stuck
processing a stalled readback. A candidate fix is written, and validated clean across
21 consecutive runs (including the real production config) after reliably hanging on
run 1 in every prior test.

## How the root cause was found

1. **The "first attack against a fresh server" pattern.** Every confirmed hang
   tonight happened on the very first burst of load against a just-launched server —
   never on a server that had already served requests successfully. Testing this
   directly (fresh server, first request = a conc=64 burst) reproduced the hang
   immediately and reliably.
2. **"Self-recovery" turned out to be client-disconnect-triggered, not spontaneous.**
   Earlier observations of the hang resolving after ~80s all used a client-side
   `timeout` wrapper. Removing the client timeout entirely and just watching the
   server showed **10 full minutes of total silence with zero recovery** — the
   apparent "recovery" was the client's own timeout disconnecting and cancelling its
   requests, which (as understood only after finding the root cause) triggers a
   cleanup path that happens to unstick things. It is not a real device recovery.
3. **A stable, indefinitely-wedged process is much easier to debug than a
   flaky one.** With no self-recovery, gdb could be attached at leisure. Across all
   178 threads in the wedged process, only **one** tt-metal-namespaced function
   appeared anywhere in any backtrace: the completion-queue reader, idling in
   `pthread_cond_wait`, waiting on a condition variable for new read-work to arrive.
   Nothing else in the process was touching the device at all.
4. **`py-spy` (installed fresh into the venv) gave the real answer.** The vLLM
   EngineCore main thread was spinning harmlessly in `vllm/v1/engine/core.py`'s
   documented `time.sleep(0.001)` (`_process_engine_step`), which only fires when
   `not model_executed`. `model_executed` is computed as
   `scheduler_output.total_num_scheduled_tokens > 0` (`core.py:409`) — i.e. **the
   scheduler was returning a completely empty schedule, every single step,
   forever.** Not a device hang — a Python scheduler that had talked itself into
   permanently declining to do anything.
5. **A one-line diagnostic in `AscendScheduler.schedule()`** (temporary, removed
   after use) confirmed exactly one stale entry in `self.scheduled_req_ids`, and
   critically: **not present in `self.running`.**

## Root cause

`vllm_tt.scheduler.AscendScheduler` (`integrations/vllm_plugin/vllm_tt/scheduler/ascend_scheduler.py`
in tt-xla) tracks in-flight-this-step admissions in `self.scheduled_req_ids`, a set
used to gate its decode path:

```python
# schedule(): decode is only scheduled if nothing was scheduled during prefill
if len(self.scheduled_req_ids) == 0:
    ...decode the running batch...
```

During prefill admission, the shared per-step `token_budget` is split across however
many fresh requests are admitted in one wave. If the budget runs out **partway
through the last request** in a wide admission batch, that request gets a **partial**
prefill chunk instead of a full one. It is correctly kept out of `self.running` (not
yet fully prefilled) — but its ID was already added to `self.scheduled_req_ids` in the
same prefill-loop pass, unconditionally.

`update_from_output()` — the only place that ever clears `scheduled_req_ids` — only
checked `self.running`:

```python
# pre-fix
for request in self.running:
    ...
    if req_id in self.scheduled_req_ids:
        self.scheduled_req_ids.remove(req_id)
```

A partial/continuation request is by design never in `self.running` until it
finishes prefilling — so this loop can **never** see it, and its entry can never be
cleared. From that point on:

- Decode is permanently blocked for **every** running request (the gate above),
  not just the stuck one.
- Nothing ever finishes, so `self.running` never frees a slot.
- The stuck partial can never be re-admitted to finish its remaining chunk either
  (the prefill loop's own concurrency-cap check blocks it once `self.running` is
  full).
- So `scheduled_req_ids` can never clear.

A perfect, self-sustaining circular deadlock — zero device activity, no exception, no
timeout to break it. This also explains the one thing that *did* appear to "fix" it:
a client disconnect triggers `finish_requests()`, which explicitly does
`self.scheduled_req_ids.discard(request.request_id)` for RUNNING-status requests —
that's what was actually unwedging things, not any device-side recovery.

### Why concurrency ~32 specifically

The deadlock needs a wide-enough admission batch that the shared token budget doesn't
divide evenly across every admitted request. At `max_num_seqs=32` (the production
config), a burst of ≥32 concurrent fresh requests is exactly what creates that
condition — matching every observed trigger (`vllm bench serve` at conc=64,
`lm-eval-harness` at `num_concurrent=32`).

### Not first-wave-specific, and not cold-server-specific

Structurally, the bug doesn't care whether it's the first request ever served — only
whether a wide-enough admission batch hits the exact token-budget boundary. This is
proven directly by the standalone repro (below), which triggers it on what's
effectively a "second wave" against a scheduler that already has 31 unrelated
requests running. It's also corroborated by a real CI run (tt-shield run
[28732200234](https://github.com/tenstorrent/tt-shield/actions/runs/28732200234)):
the server had already served a full 37-minute `r1_gpqa_diamond` eval successfully
before hanging on the very first wave of the *next* eval, `mmlu_pro`
(`num_concurrent=32`). Empirically, cold-server-first-attack was simply the *most
reliable* way to trigger it in ad hoc testing — likely because a fresh server
receiving many simultaneous requests produces the cleanest possible synchronized wide
batch, with nothing else in flight to stagger the admission timing.

## Introduced by

tt-xla commit `39200a196` — **"[vLLM] Enable chunked prefill: decouple buckets +
runtime chunked SDPA (tt-xla#4986) (tt-xla#5283)"**, merged 2026-07-02. This commit
introduced the `fully_prefilled` / partial-chunk-continuation concept (a request can
now be admitted without immediately joining `self.running`) but didn't update
`update_from_output()`'s cleanup to account for that new case. `update_from_output()`
itself is unchanged since the original scheduler.

## Candidate fix

In `update_from_output()`, clear `scheduled_req_ids` based on everything **actually
scheduled that step** (`scheduler_output.num_scheduled_tokens`, which includes
partials) instead of iterating `self.running`:

```python
def update_from_output(self, scheduler_output, model_runner_output):
    for req_id in scheduler_output.num_scheduled_tokens:
        self.scheduled_req_ids.discard(req_id)
    return super().update_from_output(scheduler_output, model_runner_output)
```

One-hunk change; not yet committed (currently a working-tree diff in the local
`~/tt-xla` checkout).

## Validation

| Config | Attempts | Result |
|---|---|---|
| Single-layer debug build, 2048 ctx, chunk=1024, conc=64 | 10 | **10/10 PASS**, 0 hangs |
| Same, conc=96 / conc=128 (bigger admission backlog) | 6 | **6/6 PASS**, 0 hangs |
| **Real production config** — full 36-layer Qwen3-8B, 40960 ctx, launched via the unmodified production launcher, conc=64 | 5 | **5/5 PASS** |

**21 consecutive clean runs, zero hangs**, across three configs including the actual
production shape — every one of these exact configs reliably hung on the *first*
attempt before the fix.

### Hardware-free unit-level repro

A standalone, deterministic test drives the real `AscendScheduler` class directly —
no TT hardware, no model weights, no serving stack, ~5 seconds to run. It constructs
a genuine `VllmConfig` via vLLM's own `AsyncEngineArgs.create_engine_config()` (this
only resolves the model's HF config; it never touches a device) plus a small
hand-built KV-cache config, then drives `schedule()`/`update_from_output()` through a
scenario that reproduces the exact trigger: 31 already-running, unrelated requests,
then one more that gets a partial chunk when the shared token budget runs out.

- Against the **current (fixed)** code: passes — `scheduled_req_ids` clears, decode
  proceeds normally for the 31 running requests.
- Against the **pre-fix** code (verified via `git stash` on the real file, not a
  reimplementation): **genuinely fails** — zero tokens scheduled despite 31
  unfinished running requests.

This is a real regression test, not a demonstration copy: it runs whatever code is
actually in `ascend_scheduler.py`, so it will fail again if the fix is ever reverted.

## Open items

- Fix + regression test are not yet committed/PR'd to tt-xla.
- A tt-xla issue is being filed separately to track the fix itself (see draft in this
  repo, `ISSUE_DRAFT_tt_xla_ascend_scheduler_deadlock_4521.md`).
- Worth double-checking: an earlier datapoint suggested the same symptom also
  appeared on an older `~/tt-xla-2` workspace thought to predate this fix's
  introducing commit (`39200a196`, 2026-07-02) — if that workspace's pinned commit
  is confirmed to be genuinely before 2026-07-02, that would point at a *second*,
  independent bug with the same symptom (or an inaccurate commit-date assumption for
  that workspace). Not yet resolved; flagged here rather than asserted either way.
