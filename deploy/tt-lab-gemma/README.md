# Gemma 4 26B-A4B text inference on Blackhole card 1

This deployment uses the existing Inference Server pipeline and a persistent
`tt-lab gemma --native-device --serve` child. It serves
`google/gemma-4-26B-A4B-it` on `http://127.0.0.1:8001`, separately from GPT-OSS
on port 8000/card 0. All model layers execute on the card; the server performs
tokenization and API handling. It supports text, greedy decoding, streaming,
and one request at a time with a 4096-token combined prompt/output limit.

Start or inspect the installed **user** service:

```sh
systemctl --user start ttlab-gemma-inference.service
systemctl --user status ttlab-gemma-inference.service
journalctl --user -u ttlab-gemma-inference.service -f
curl http://127.0.0.1:8001/tt-liveness
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-26B-A4B-it","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":32,"temperature":0}'
```

The unit is started on demand; it is not enabled for automatic login/boot.
To install it on this checkout again:

```sh
cp deploy/tt-lab-gemma/ttlab-gemma-inference.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

`server.env` selects physical card 1, the model artifacts, localhost binding,
and the disabled-thinking chat template. `start.sh` uses the existing Python
venv with an isolated Transformers 5.17.0 overlay at `gemma-reference/python`.
It does not modify the GPT-OSS runtime environment. No additional hugepage
reservation is needed by the current Gemma path.

The selected TTQ uses BFP8 dense transformer matrices, experts and output
projection, BF16 embedding lookup, FP32 decode KV, and a separate BF16 matrix-prefill KV cache. The BF16-output checkpoint remains
available as `gemma4-26b-a4b.ttq`; select it with `TT_LAB_TTQ` in `server.env`.
Performance and the measured output-precision tradeoff are recorded in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-performance.md`.

The experimental `gemma4-26b-a4b-experts-bfp4.ttq` checkpoint is also supported
by this runtime and has passed native arithmetic and live API checks. It
improves sustained decode by 7–11% in measured probes, but increases numerical
drift in the small quality regression set. The default remains all-matrix
BFP8. See `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-bfp4-integration.md`
for the measured precision tradeoff and evaluation commands.

Run the live benchmark against an idle endpoint:

```sh
python3 deploy/tt-lab-gemma/benchmark.py --long --output gemma-benchmark.json
```

It checks factual/math/code responses, streaming, repeated requests, cache
resets after a 1025-token prompt, and invalid API parameters. Add `--full-context`
to test a 4088-token prompt plus eight output tokens. The service uses
`TT_GEMMA_PREFILL=512` and `TT_GEMMA_SHARED_PREFILL_STATE=1` for
prompt-state/expert-routing groups, while each matrix microbatch remains bounded
to 32 vectors. Groups above 128 require shared state; shared state requires
matrix prefill and matrix prefill requires 16-aligned groups. Set the group to
at most 128 before disabling sharing. The current checkpoint shares one token
state buffer per DRAM channel and divides independent token phases among four
worker groups. `TT_GEMMA_DIRECT_EXPERT_STORES=1` additionally publishes each
expert down-projection shard directly to shared DRAM outputs, eliminating the
full-output L1 gather. It requires matrix prefill and retains publication
barriers and rank-ordered accumulation. Set only this flag to `0` to restore
the gathered-output control. The service also enables
`TT_GEMMA_PACKED_EXPERT_INPUT=1` (reuse once-packed BF16 expert inputs) and
`TT_GEMMA_SOFTMAX_SCALE_PACK=1` (SFPU normalization scaling plus BF16 packing).
Both require matrix prefill and can be disabled independently. They add no
DRAM allocation or normal token-loop allocation. The service also enables
`TT_GEMMA_PREFILL_CONSTANT_CACHE=1`, lazily caching layer constants in 94,784
bytes of worker L1. It requires matrix prefill and fused experts; disable
the cache before disabling either dependency. The service now enables
`TT_GEMMA_PREFILL_BATCH_ROUTER=1` to batch router projection and distribute
normalization/selection across token-owner groups. It requires matrix prefill
and shared token state; disable it before disabling either dependency.
The service also enables `TT_GEMMA_PREFILL_LOCAL_GELU=1` (token-local shared
activations; requires matrix prefill/shared state) and
`TT_GEMMA_PACKED_EXPERT_HIDDEN=1` (once-packed BF16 expert hidden gather;
requires matrix prefill/fusion). Disable each before disabling its dependency.
The service also enables `TT_GEMMA_DIRECT_PREFILL_STORES=1`, publishing dense
projection shards directly to the eight existing shared token-state copies.
It requires matrix prefill/shared state and retains FP32 outputs and
publication barriers. Disable it before disabling either dependency.
`TT_GEMMA_SHARED_DENSE_INPUT=1` additionally prepares dense input layouts
once across four eight-token owners, then publishes disjoint slices with the
existing multicast or unicast gather. It requires matrix prefill/projection
reuse; disable it before disabling either dependency. Incomplete eight-token
groups retain the previous preparation path.
`TT_GEMMA_SPLIT_DECODE_ATTENTION=1` gives each decode query head two workers:
QK splits the time range and AV splits output features, preserving dot-product
and chronological accumulation order. It requires matrix prefill and vector
attention; disable it before disabling either. Dedicated decode-only firmware
uses the existing FP32 KV cache. Setting the flag to `0` restores the original
sixteen-owner decode image.
Fresh synthetic 1K/near-4K TTFT is 3.502/14.662 s; sustained decode is 65.90
short, 48.84 at 1K and 40.30 near 4K (11.54% faster near 4K). Varied-4K
decode is 40.35 tok/s with unchanged continuation; its TTFT remains 14.750 s.
Full-model, delayed-worker, compatibility and live full-context API checks
pass, together with 153 kernel checks and 17.56 million byte-matching long-context
logits. See `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-split-decode-attention.md`
for binary identity, controls, numerical checks and rollback.
Decode-only normal/capture/diagnostic firmware is 25224/26456/25412 bytes,
all below the unchanged 32768-byte guard. The preceding shared-input change is
documented in `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-shared-dense-input.md`.
The preceding direct-publication change is documented in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-direct-prefill-stores.md`.
The earlier checkpoint's
120 isolated GELU checks and conversion limits are documented in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-prefill-gelu-handoff.md`.
Normal/stress matrix images are 32108/32648 bytes under the unchanged
32768-byte guard; normal wide decode is 32704 bytes. No new buffers are
allocated. The previous cache checkpoint is documented in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-prefill-constant-cache.md`.
Correctness checks preserve the existing packer's underflow-boundary behavior;
see `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-prefill-pack-reuse.md`.
The preceding direct-store checkpoint is documented in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-direct-expert-stores.md`.
Live API checks include the full 4096-token
boundary and request reset. The largest DRAM channel uses 3.980 GiB of the
4080 MiB allocator limit: capacity is tight. The buyer-experience target is
still unmet. See `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-shared-prefill-state.md`.
The following optimization entries describe earlier checkpoints.
See `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-wide-prefill.md` for measurements,
memory bounds and correctness checks. `TT_GEMMA_PREFILL_ATTENTION_PAIR=1`
processes two prompt positions in parallel across the 32 workers while keeping
each query's attention reduction chronological. It reduces measured fresh
prefill latency by about 19% at 1K and 28% near 4K versus the preceding build;
the firmware-size refactor costs approximately 1% decode throughput. Set the
flag to `0` and restart to disable pairing. Numerical checks, controls, and
the preserved previous binary are documented in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-prefill-attention-pair.md`.
`TT_GEMMA_EXPERT_FUSION=1` keeps expert gate/up, GELU, and down-projection
handoff on-chip, removing intermediate gathers and DRAM transfers. It adds
roughly 4–6% prefill latency reduction without a material decode change;
all-matrix BFP8 remains the default. Set it to `0` and restart to disable.
See `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-expert-fusion.md`.
`TT_GEMMA_PREFILL_LOCAL_STATE=1` removes full-worker barriers around private
prompt-state transfers, retaining the attention/GELU load-side barriers and
matrix publication barriers that protect peer buffers. It reduces measured
TTFT by about 3–4%, with no material decode change. Set it to `0` and restart
for the full-barrier control. Normal and deliberately skewed-worker checks are
documented in `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-local-state.md`.
Local attention now keeps sixteen cache entries per Dst chunk; global
attention retains its safe eight-entry chunks. This preserves arithmetic
and reduced measured fresh 1K/near-4K TTFT at that checkpoint to 8.57/42.43 s,
with about 45.3/36.2 tok/s sustained decode. See
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-local-dst16.md` for controls
and validation. The separate QK/AV replay experiment was rejected and removed.
`TT_GEMMA_PREFILL_PROJECTION_REUSE=1` now shares input packing/layout across
K/V/Q and shared gate/up projections. Fresh 1K/near-4K TTFT is 8.27/41.25 s,
with effectively unchanged decode; varied-1K TTFT is 8.43 s. The flag requires
multivec and can be set to `0` for an ablation. Full-model, long-state, and
API validation are recorded in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-projection-reuse.md`.
Padded matrix tails remain diagnostic-only, not a serving change.
`TT_GEMMA_SHARED_EXPERT_OUTPUTS=1` now shares the identical gathered expert
outputs per DRAM channel. It saves 264 MiB on the card (33 MiB/channel) at
the then-current 128-token routing-group size. Fresh synthetic 1K/near-4K TTFT was
8.19/40.97 s; varied-1K TTFT is 8.35 s. Decode is effectively unchanged.
The flag requires grouped expert prefill and can be set to `0` for an ablation.
All publication barriers and expert accumulation order remain intact.
Full-model, delayed-writer/reader, and compatibility evidence is in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-shared-expert-outputs.md`.
`TT_GEMMA_MATRIX_PREFILL=1` now enables matrix-engine attention during prefill,
with a separate linear BF16 packed cache and sixteen prompt positions processed
together. Decode retains FP32 KV and switches back to the normal firmware.
Fresh synthetic 1K/near-4K TTFT is 6.924/28.363 s (15.50%/30.76% faster);
sustained decode stays about 65.44 short and 36.11 near 4K. Short TTFT regresses
by about 9 ms. Full-model proxy, delayed-worker/long-state, sampled quality,
and live API checks pass. The sampled quality comparison is not broad accuracy
certification, and the buyer-experience prefill target remains unmet.
To restore FP32 prefill, also disable packed expert input, softmax scale/pack,
direct expert stores and shared token state and return the routing
group to at most 128 before setting this flag to `0` and restarting this service.
Evidence and limitations: `/home/ttuser/workspace/tt-lab-gemma/docs/gemma-matrix-prefill.md`.
`TT_GEMMA_MULTIVEC=1` executes eight vectors together on beneficial matrix
shapes, with per-vector fallback for the remaining shapes and tails. Set it
to `0` to disable (also disable projection reuse). This preserves arithmetic and improves measured time to
first token by about 1–2%; decode remains unchanged. Validation and the
shape-selection policy are recorded in
`/home/ttuser/workspace/tt-lab-gemma/docs/gemma-multivec-integration.md`.
The worker deadline is two hours
to allow long prompts. Legacy `/v1/completions` SSE omits
usage; length-limited synthetic probes derive counts from the exact request
length and length finish. Native worker logs provide the corresponding measured
prefill/decode times and generated-token counts.

Adapter tests (no card opened):

```sh
PYTHONPATH=/home/ttuser/workspace/gemma-reference/python \
 /home/ttuser/workspace/tt-inference-server/.venv-ttlab/bin/python \
 -m pytest -q deploy/tt-lab-gemma/test_adapter.py
```

For build, import, numerical validation, memory layout, and benchmark evidence,
see `/home/ttuser/workspace/tt-lab-gemma/docs/gemma.md`. This source tree is an
isolated snapshot of the working Inference Server checkout. The original
checkout and running GPT-OSS system service were not changed.
