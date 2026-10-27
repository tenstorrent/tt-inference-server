# Prepared SWE500 shard 1

This is ordered complement slice 1 (100 tasks) after the historical frozen
SWE100 cohort. Together, shard0 and the four complement slices cover all500
frozen registry identities exactly once. The task sequence is determined by
registry order, independently of Qwen task rewards.

Runtime, medium reasoning, seed, sampling, isolation and all individual
budgets match the prepared SWE100 arm. Suite45900 seconds fits the14-hour
CI cap with4500 seconds for startup/export. No task budget is shortened.
The common source baseline is `bb3e141aace2f49656e8bf25af9c62a16702beac`. Only task-selection metadata and this
document differ. This is a prepared configuration, not a completed evaluation.
Dispatch remains gated on serving and candidate accuracy/runtime evidence.
Raw artifacts remain local/CI; only code/config and compact docs enter Git.
