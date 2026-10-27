# Frozen25 isolated Qwen control

The isolated8 B8/256/C8 medium control CI40554052639 completed7/8 verifiers
with all8 normal submissions and verified isolation. This larger baseline uses
the frozen historical Gemma SWE100 cohort, every fourth ordered task for25;
task selection predates Qwen outcomes. See the checked-in selection manifest.

Keep the same B8,256-block, native262144 model context, weights/tokenizer,
Sampling1D and medium reasoning. Agent deadline6300s, individual model timeout
3600s and output cap16384 remain unchanged. Harbor concurrency stays8. The
suite deadline27000s allows four waves of6300s plus1800s cleanup; the pinned
CI job cap is50400s. The suite deadline increases for the larger task count;
individual task and generation budgets are preserved.

Only SWE Verified runs. This is an accuracy/runtime baseline for the larger
cohort, not a speedup or GPU-parity claim. Collect all25 task outcomes including
failures, actual network phase evidence, complete stable HTTP/model audits,
source identities and both task-envelope and full job time. No CSV/Tracy or raw
trajectories are committed.
