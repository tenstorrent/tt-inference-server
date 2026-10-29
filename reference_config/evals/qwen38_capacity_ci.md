# B32 maximum-KV prefix study

This unscored run measures whether reclaiming duplicate MLP weights improves
prefix retention when each weight mode receives its largest validated safe KV
pool. Logical context stays262144, resident batch/admission32, checkpoint512.
The one-output synthetic probe measures cache mechanics, not SWE accuracy or
end-to-end agent throughput.

Image build40846316059 published immutable digest
`sha256:41aa0687bc668186baf9c4ed7942e135b60cf8828d9bbd67418cf53993a3d612`.
It inherits the compiled backend/plugin from the preceding flat-prefill image
and copies Metal`502069d1858aeaba995d90653eb4999193e6b4c8`:29 runtime files and
six explicit diagnostic helpers. The downloaded manifest matches all35 Git
blobs. Image building is not hardware qualification.

The explicit `QWEN38_RUN_CAPACITY_SCREEN=1` dev configuration runs the standalone
controller after weights/runtime environment resolution, before trace capture
or vLLM opens devices. Both1368-ID anchors precede adaptive larger-pool trials.
The registered planner limits each mode to eight receipts, including the
three-cycle qualification; the controller additionally limits each subprocess
to1800 seconds and the campaign to four hours. Both modes use512MiB normal DRAM
free at allocation-event peak,8MiB TRACE free and an8MiB-per-bank fragmentation
margin. Unknown errors, incomplete device close, timeouts, unmatched outputs or
an incomplete boundary search prevent server startup. Live timeout evidence is
captured before terminating a diagnostic child. A maximum requires a closed
three-cycle pass and a closed adjacent-ID failure of the same memory guards.
The explicit capacity configuration disables automatic server boot retries;
a failed campaign cannot silently start again before evidence review.

Raw receipts, inputs, allocation graphs, source manifest, per-trial commands and
logs persist under `workflow_logs/qwen38_capacity`, included in the existing CI
artifact upload. No raw evidence is committed to Git. Readiness allowance for
this experiment is23400 seconds so it includes the campaign and normal boot;
the subsequent prefix request budget remains3600 seconds.

After both native boundaries qualify, this run starts the unchanged1368-ID
packed-weight reference server and runs the212-request B32 probe. Materialize
the control-max and shared-max arms using
`test_module/llm_tests/prepare_qwen38_prefix_pool_arm.py` and the original
qualifying/unsafe receipts, image manifest and selection. Each arm gets a fresh
server, identical transcripts, fixed study salts and generation parameters.
Materialization disables the native campaign and preserves the memory
checkpoints without duplicate YAML keys. Real serving must attest selected
physical pool sizes and pass the same fixed-work probe before capacity is
accepted. Report cached versus eligible/recomputed tokens, executed worker
prefill, queues/preemptions and whole wall time, including the shared-MLP decode
cost. Control-max minus reference isolates existing spare headroom; shared-max
minus control-max isolates the gain from reclaimed weights.
