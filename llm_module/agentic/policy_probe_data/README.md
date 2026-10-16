# Unscored batched-prefill CI diagnostic

This dedicated branch routes the Terminal benchmark entry point to an API-only mechanism test. It does not run Harbor tasks and emits status `na` for a completed diagnostic. Never use its output as SWE or Terminal accuracy.

Model snapshot: `spansweep1-source`, true model request batching with native batch-axis BF16 reduce-scatter. Local evidence includes50 exact reduced layer/shape cases, same-operand collective controls, and preceding full60 alias/null/scratch/cache-restoration gates. Final-source full60 and scored acceptance are still in progress.

The model overlay manifest pins all changed files; plugin runner target remains validated final-history SHA256 `5aef4eb2a817d8bc63643676744f8b0fc88eb1e99ef93e12004a4264f6ad1c30`. C16, F606208, TRACE128MiB, TP4, original RoPE, native FCFS and2P1D remain configured. Prefill batch cap8; long/wide experimental paths are off. No kernel change.

The frozen synthetic token workload isolates request handling and cache reuse; it is separate from the archived real-query replay and from the correctly formatted qualitative chat checks included by the harness.

Host validation: sampling contracts25 passed/1 skipped; repository pytest2899 passed/12 skipped/7 failed. The same seven failures reproduce on unchanged parent `ec50aff2` in an isolated checkout with the same interpreter (tokenizer fixture, streamed Unicode, inherited Harbor/preset expectations). Lint, format, copyright and hash-checked patch application passed. The pytest commit hook is skipped only after recording these completed candidate/parent runs; no full-suite clean-pass claim.
