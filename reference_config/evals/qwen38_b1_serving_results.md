# Full 64-layer B1 serving check

`QWEN38_COMPACT_B1_RECURRENCE=1` raises warmed 4096/128 decode throughput from
**37.4368 to 42.4047 tokens/s/user (+13.2700%)** on the matched QB2 serving pair.
All three primary candidate repeats exceed 40: 42.404697, 42.408180, 42.401641.
Primary client TTFT is 420.004→418.389 ms; this does not establish a meaningful
prefill improvement. End-to-end output throughput, including prefill, is
33.5743→37.5012 tokens/s. These are unscored serving measurements.

| Native-chat input/output | Control decode TSU | Compact decode TSU | Gain | Client TTFT, ms | End-to-end output tokens/s |
|---|---:|---:|---:|---:|---:|
|128/128|37.9917|43.1093|13.4703%|113.979→115.274|37.0241→41.8121|
|4096/128|37.4368|42.4047|13.2700%|420.004→418.389|33.5743→37.5012|
|129/128 boundary|37.9886|43.1093|13.4797%|115.127→118.020|37.0120→41.7661|

Each row reports three warmed-repeat medians after one timed HTTP warmup excluded from reported medians.
Decode TSU is 127 divided by elapsed client time between first and last streamed
output-token events; primary TPOT is 26.711665→23.582293 ms. All 24 paired
performance requests returned 128 single-ID SSE events and matching usage.
Prefix caching was disabled. Both full 64-layer physical B1 models advertised native
262144 context, retained selected precision, 72 physical 4096-token cache blocks,
and prepared the same 129 buckets. Cold preparation took 1278.317/1289.374 seconds
and is excluded from the table. No device profiler was used.

The earlier native full-model result of 40.673091 TSU remains a separate reference.
Current 4096 serving throughput is 1.731606 TSU higher, but the compiled backend,
prompt/token trajectory, cache ownership and buffered/native versus client-SSE
boundaries differ. This gap is not an isolated serving-overhead measurement.
Only the matched serving compact-off/on comparison supplies the 13.2700% delta.
The optimization applies to physical B1, one-token, unmasked decode; larger
resident batches retain their existing path.

## Correctness and quality

Both arms use 12 identical tokenizer input records, all 12 performance token-ID
sequences/text/usage/generation settings, and all six shared native-chat
qualitative outputs/finish reasons. Within each arm, warmup and all three
repeats also match exactly. The live `/tokenize` route rendered user messages
with `add_generation_prompt=true` and `enable_thinking=false` before completions.

The six quality requests allowed normal EOS and were read individually. Haiku, French
translation and complete Fibonacci code reached EOS; explanation, story and
thermodynamics are coherent 256-token capped prefixes, not completed answers.
The generated Fibonacci function passed restricted CPU checks at −1, 0, 1, 2, 8.
Forced 128 timing requests intentionally set `ignore_eos=true`; their later
newline/role/thinking tails are timing stress, not meaningful quality answers.

Each worker log reconciles 18 prefill/completion/releases, 2506 sampled/read
results and 2488 model replays. Returned IDs total 2503 and already include EOS.
There is one additional in-flight result on each of the three EOS-ended quality
calls, consistent with async lookahead/late-output suppression. Every timing
call has exactly 128 samples and 127 model replays. Exact discarded token values
were not logged. Verified installed model-runner source documents lifecycle
handling before deferred outputs; its relevant async methods match upstream
1bad, but the separately installed async module was not captured in readback.

Trace captures remain 3→3, with zero reported retirements and zero host-logit
reads throughout the 18-request workload. Shutdown trace destruction is not
separately instrumented. Owned-resource cleanup is independently complete in
both runs. Preparation DRAM checkpoints differ by +98,304 bytes/device for the
candidate; this is not a peak or unique-owner inventory claim.

## Exact evidence and limits

- [Control 40926118763](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40926118763), TTI `df02557da5ae4ad2fe17d5958c6e6cf5cb852c26`.
- [Candidate 40931138892](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40931138892), TTI `ef8461f67c628f73441cf76d61cf37a77d3b8d95`.
- Both actual runners are `qb2-120-p05t02`; wrapper `cae0049e9a05b6858af98f5402bdb4d78898514c` is identical. Final measured TTI trees differ only in the compact 0/1 flag. Both explicitly allow one startup attempt.
- [Image build 40923312357](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/40923312357), build wrapper `9766ee44db66636b27c86c41d4d4cc3c956cfdba`; all 33 runtime hashes match Metal `f023ac0e0bcb70118f09eb994128e65dfdbc7411`.
- Immutable image: `ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:cae3e50bc45fa12f7ff34410fdaf3b5a5362bc7296a3979a1112e61a5b1c9939`.
- Readback attests vLLM `0.30.0+empty` / plugin `0.1.0`. Actual installed model_runner SHA256 is `f09e7719214d6e3e1d2b82e4abc3534b9779a1358949a9adda0625f2d1c3422c`, including hybrid/prefill patches; the dispatch ref alone does not identify pristine installed source. Canonical Sampling1D retains max_top_k 32, so request k1 does not enter the discarded width-32 specialization.

The stable **two-file exporter is incomplete in both arms**: it requires a
`qwen38-http-*.jsonl` file, while the audit middleware records only
`/v1/chat/completions` and this probe uses `/v1/completions`. Helper receipts
explicitly report the missing file. Worker JSONL and client inputs/calls/results
are retained in workflow-log artifacts and reconcile as above. No stable-export
pass or scored-agent audit is claimed. This is a bounded B1 regression check,
not a completed new integration/release, full accuracy benchmark or SWE result.

An initial image readback used the wrong interpreter and was repaired before
hardware. An initial control was canceled to correct the image allowlist;
cancellation raced model loading, and failed removal/old-owner evidence was
preserved. Fresh same-runner absence/root-fuser/4-device-list proof 40925849632
passed before replacement, without reset. The final pair used single-attempt
startup and complete owned cleanup. No local shared 0.26/Granite environment,
Claude unpadded/union code or local device lease was changed.

Compact numerical evidence is in [qwen38_b1_serving_results.json](qwen38_b1_serving_results.json).
Raw evidence remains in CI and locally under
`/home/mvasiljev/build/qwen38-swe/local-tsu-ttft/serving-ci/`.
Independent review: [clean-pass for this bounded check](qwen38_b1_serving_review.md).
