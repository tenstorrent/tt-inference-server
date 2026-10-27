# Flat-prefill immutable serving smoke

This arm adds `QWEN38_FLAT_PREFILL=1` to the reviewed grouped+masked B16/block512
serving configuration. Metal source is
`6faa07fb5320284381ca1a9ab4651489366d7436`. The local stage passed independent
review: full-model state/logit/token equivalence, natural-chat checks that select
physical256/512, separate B32 watcher, three-repeat TTFT8771.253→7727.073 ms,
matched device profiles, and measured L1-input/grid alternatives. Sampling1D and
all precision/cache/context contracts are retained.

Build40599125742 produced the immutable source-only image
`sha256:677859aa1eba574ffb95214247ed1849ad17cddfe1faf1884c0dd73a5bf53d07`.
All27 runtime file hashes match the tested snapshot and the published Metal
source. The stable audit exporter accepts only this exact additional digest;
its helper still uses the original pinned base image.
No mutable image or arbitrary-overlay allowlist change is permitted.

The single known small task is `astropy__astropy-14309`, from the frozen cohort.
HTTP concurrency is1 while resident batch is16. Keep medium reasoning, seed1234,
16384 output-token cap,6300-second agent and3600-second request deadlines,
7200-second suite, actual owned-network isolation and native262144 context.
The mask-only image's earlier C1 run40579353560 solved this task normally with
complete API accounting. This new run is serving/ABI integration evidence, not
a general accuracy estimate or SWE speedup comparison.

Acceptance requires the exact model/image/plugin/TTI sources; all64 layers and
`flat_prefill_layers=64`, `masked_recurrence_layers=48`; normal task submission,
verifier reward, complete HTTP/trajectory accounting and actual isolation phases;
and successful stable exact audit export after owned cleanup. Retain any failed
startup attempts separately. AI summaries and issue comments remain disabled.
Raw logs, token arrays and ZIP/CSV/Tracy artifacts stay outside Git.
