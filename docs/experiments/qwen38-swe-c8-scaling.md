# Prepared Qwen C8 SWE100 / SWE500 partitions

Status: prepared, not dispatched. The current combined C8 qualification is
run40607188492, immutable TTI649c4342d6a1aab5f99c7c6ece672721182a98e9.
Its actual results, isolation, response accounting and stable exports must
qualify the configuration before any partition is launched. The previous
C16 pilot14/25 with three timeouts is not a pass for the held C16 arms.

Reuse the frozen manifest reference_config/evals/qwen38_swe_scaling_cohorts.json
(SHA4ef795e4f6c817624d996a7a78453bfa13869765a10ca3c14df39a1f9ca89aa4).
Parts0/1 are the first/last50 of historical Gemma SWE100. Remove those100
from frozen SWE500 in original order; parts2–9 are the eight consecutive50
blocks of the remainder. The ten parts are disjoint and cover exactly500.
No task is added, removed or substituted based on a Qwen reward.

HTTP C8 with100 tasks needs thirteen6300-second waves to guarantee the
individual agent budgets, beyond the pinned14-hour CI job. Each50-task
partition uses seven waves plus1800 seconds overhead:45900 seconds, leaving
4500 seconds inside the50400-second job for startup and collection. Task
setup/verifier work and actual tails must still be measured. This preserves
individual6300-second agent/3600-second request limits and16384 output cap.

Every selected part must use one common immutable image and model/plugin
source: candidate Metal6faa07fb5320284381ca1a9ab4651489366d7436, plugin
1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf, image digest
677859aa1eba574ffb95214247ed1849ad17cddfe1faf1884c0dd73a5bf53d07.
The exact serving YAML is unchanged from the C8 qualification. Resident16,
block512, grouped/masked/flat enabled, native262144 context, Sampling1D,
medium reasoning, seed1234, temperature1/top_p0.95/top_k20, frozen dataset
and actual agent isolation are unchanged. Each part has a distinct immutable
TTI ref differing only in its task selection/phase and compact part note.

SWE100 is the aggregate of parts0/1; SWE500 is the aggregate of all ten.
Do not mix pilot8/25 outcomes into either score. Keep every attempted task,
including timeouts, in the denominator. Check actual task checksums and
coverage, source/image/generation equivalence, isolation, response accounting
and retrieval findings before accepting the aggregate. Preserve failed
attempts separately; never silently replace them with a favorable retry.
Report summed runner hours and elapsed cohort wall time separately, including
startup, tails and export overhead. A dispatched job is not a result.

Only code/configuration and compact docs are pushed. Logs, CSV/Tracy, token
arrays, raw trajectories, weights and ZIP archives stay local or in CI.
