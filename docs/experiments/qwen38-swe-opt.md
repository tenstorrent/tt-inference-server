# Qwen3.8 SWE serving candidate

This experiment branch prepares a small serving validation of the opt-in grouped
recurrence in `models/autoports/qwen_qwen3_8_27b`. It is not a selected release
configuration. Model implementation and measured local results are on tt-metal
branch `mvasiljevic/qwen38-swe-opt`.

The pilot preserves the eight frozen task identities, mini-swe-agent2.2.8,
seed1234, temperature1/top-p0.95/top-k20, output16384, agent6300s,
request3600s and suite7200s. It uses medium reasoning like the current B8
medium arm; selection of reasoning policy awaits completed scored pilots.
Native context262144 and the checkpoint/tokenizer revision remain fixed.

| Setting | B8 medium control | Prepared candidate |
|---|---:|---:|
| Resident rows |8|16|
| Pilot concurrent trials |8|8|
| Attention/recurrent cache block and retention interval |256|512|
| Grouped recurrent decode policy |off|on|
| Cache backing budget per device |23,661,297,664 bytes|23,661,297,664 bytes|

The combination targets future16-client runs: larger blocks reduce snapshot
metadata/dispatch and increase cache token capacity; grouped recurrence avoids
two split updates at resident16. This is a combined serving configuration,
so its SWE time must not be attributed solely to the grouped kernel.
No scored or serving improvement is established by this preparation.

Host validation: actual model-spec resolution confirms all three block/retention
fields512, resident16, policy flag1, native262144 and pinned weights/tokenizer.
The focused SWE policy/audit/mount/session tests pass20 tests plus4 subtests.
Before dispatch require paired full-model state/logit checks and an immutable
source-overlay image receipt. Verify the adapter path and effective policy from
startup logs. Collect request usage, missing responses, outer exceptions,
verifier results and cache/preemption behavior before scaling25→100→500.

Only code and compact documentation belong in Git. Raw CSVs, Tracy captures,
logs, token arrays and per-draw/probe artifacts stay local or in CI artifacts.

The candidate also enables the same isolated-agent protocol as the clean B8
control, including the canonical Docker driver forwarding fix and provider/type
guards. Setup/verifier may use egress; agent execution uses the internal bridge
and host API only. Dispatch waits for the actual one-task protocol smoke audit.
