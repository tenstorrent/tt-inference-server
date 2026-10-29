# Qwen B1 serving regression check

This unscored probe compares `QWEN38_COMPACT_B1_RECURRENCE=0/1` on fresh
full64-layer, B1 servers with native262144 context. The two arm commits must
differ only in that flag. They share one immutable source overlay of Metal
`f023ac0e0bcb70118f09eb994128e65dfdbc7411`, with the selected precision policy
unchanged and the compiled vLLM0.30/backend inherited from the qualified image.
No Claude unpadded-prefill or union-state changes are included.

The source-owned `b1_serving_probe` route runs before scored agentic evaluation.
For each exact native-chat input length128,4096 and129, it issues one128-output
warmup and three128-output measured streaming requests, serially. Prefix caching
is disabled. Every request must return128 delta token IDs and matching usage;
warmed greedy output IDs must be identical. It then sends all six shared
readiness qualitative prompts through the server's native chat tokenizer with
`enable_thinking=False`, normal EOS and a256-token cap. Manual output review and
exact control/candidate comparisons remain required; an HTTP success alone is
not a quality verdict. Truncation is recorded rather than treated as EOS.

Timing starts immediately before the HTTP request. TTFT ends at the first
token-ID event; TPOT uses127 intervals between the first and last token-ID
events. Decode TSU is127 divided by that interval. End-to-end output throughput
uses128 tokens divided by complete client request wall time. These numbers
include serving and network overhead. Report warmed individual values and
medians separately for128/128,4096/128 and the129-token boundary. The native
full64 result40.673091 TSU used different timing boundaries and a different
compiled backend; it is a comparison reference, not a serving prediction.

The canonical sampler retains `Sampling1D(max_top_k=32,max_batch_size=32)`.
Request-level greedy k1 does not select the discarded width32 sampler
specialization that hung in an earlier native experiment. The build artifact
includes exact installed sampler source for this distinction.

Raw inputs, generated IDs/text, per-event timing and result JSON stay in CI
artifacts under `qwen38_b1_serving`; server logs and immutable image/source
manifests supply runtime provenance. No profiler or Tracy runs with the server.
This bounded check does not claim a completed new vLLM integration stage,
larger-batch performance, SWE accuracy or SWE task throughput.
