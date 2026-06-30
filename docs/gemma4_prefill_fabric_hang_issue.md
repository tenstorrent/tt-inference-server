# [Gemma4] Large single-shot eager prefill hangs the device (`system_memory_manager.cpp:702` fetch-queue timeout) above ~100K ISL

## Summary

On `google/gemma-4-12b-it` (P300x2 / Blackhole, 4-chip `1x4` mesh, TP=4), a single
prefill request at **~120K input tokens hangs the device** during the eager
(un-chunked) prefill and the engine dies with:

```
RuntimeError: TT_THROW @ tt_metal/impl/dispatch/system_memory_manager.cpp:702: tt::exception
TIMEOUT: device timeout in fetch queue wait, potential hang detected
```

The request is **admitted and scheduled correctly** (`num_scheduled_tokens=122893`,
reaches `Running`), so this is **not** a KV-cache / admission / OOM problem. It is a
device-side stall while enqueuing a **full-sequence** op (a per-head RMSNorm) during
prefill. Requests up to **~100K ISL pass** cleanly; the wall sits between ~100K and
~120K.

## Environment

| | |
|---|---|
| Model | `google/gemma-4-12b-it` |
| Hardware | P300x2 (Blackhole), MeshDevice `1x4`, TP=4 |
| tt-metal | `a4967d5f39d` |
| vLLM (TT fork) | `9d88cd5` (`v0.1.dev14166+g9d88cd582`) |
| `max_model_len` / `max_num_batched_tokens` | `131072` / `131072` |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | `5.0` |
| KV config | hybrid KV groups disabled (single `UniformTypeKVCacheSpecs` pool, 131,136 tokens) |

> Note: this hang reproduces independently of the KV-cache-group configuration; it is a
> property of the un-chunked prefill path, not of how the KV pool is grouped.

## Symptom / stack trace

The failing op is the per-head V RMSNorm in the Gemma4 prefill attention path. Key frames:

```
vllm_tt_plugin/model_runner.py:2212  submit_prefill
models/demos/gemma4/tt/generator_vllm.py:581  prefill_forward
models/tt_transformers/tt/generator.py:844    prefill_forward_text
models/tt_transformers/tt/generator.py:1183   prefill_forward_single_user_text
models/demos/gemma4/tt/model.py:1206          ttnn_prefill_forward
models/demos/gemma4/tt/attention/prefill.py:171  prefill_forward
models/demos/gemma4/tt/attention/prefill.py:61   _prefill_forward_single
    -> tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)
    -> ttnn EnqueueMeshWorkload (LayerNormDeviceOperation)
    -> FDMeshCommandQueue::write_program_cmds_to_subgrid
    -> system_memory_manager.cpp:702  TT_THROW("TIMEOUT: device timeout in fetch queue wait, potential hang detected")
```

`system_memory_manager.cpp:702` is the **on-timeout handler** of the prefetch
fetch-queue wait loop — the host enqueued the program command sequence for the op and
the device prefetcher/dispatcher never made forward progress, so after
`TT_METAL_OPERATION_TIMEOUT_SECONDS` the wait loop threw. It is a device stall, **not** a
command-buffer capacity overflow.

## Root cause

In `models/demos/gemma4/tt/attention/prefill.py::_prefill_forward_single`, **SDPA is
chunked** for `seq_len > 32768` (for numerical correctness), but the surrounding ops —
QKV projection, the per-head Q/K/V RMS-norms, and RoPE — **run on the entire sequence in
one shot** (no `chunk_start_idx` offset). At ~123K tokens, enqueuing one of these
full-sequence ops (the per-head V-norm) drives the device into a dispatch stall that
exceeds the per-op timeout.

```python
# prefill.py (abridged) — these run full-sequence:
xqkv = apply_qkv_projection(hidden_states, weights)
tt_q, tt_k, tt_v = split_qkv_heads_prefill(...)
tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, ...)
tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, ...)
tt_v = apply_per_head_norm(tt_v, None, ...)   # <-- hangs at ~123K
tt_q = apply_rope(tt_q, cos_cache, sin_cache)
...
# only SDPA below is chunked for seq_len > 32768
```

Gemma4's prefill explicitly does **not** yet honor a per-chunk sequence offset
(`chunk_start_idx`/`chunk_page_table` are discarded), so the whole prefill cannot be
broken into bounded sequence-token chunks — proper `chunk_start_idx` support is noted in
the code as a follow-up for bounded-memory long prefill.

## Why this is not a KV / admission / OOM issue

- The request reaches `Running` with `num_scheduled_tokens=122893` — vLLM admitted it.
- KV allocation succeeds: single `UniformTypeKVCacheSpecs` group, `48/48` per-layer
  caches allocate, `GPU KV cache size: 131,136 tokens`, no OOM.
- The throw originates in the dispatch fetch-queue wait during op execution, not in any
  allocation or scheduling code.

## Reproduction

1. Bring up `google/gemma-4-12b-it` on P300x2 with `max_model_len=131072`,
   `max_num_batched_tokens=131072` (hybrid KV groups disabled so ~120K is admissible;
   see tenstorrent/tt-metal#48283 + tenstorrent/vllm#429).
2. Send a single request with ~120K input tokens, e.g.:
   ```
   vllm bench serve --backend openai-chat --model google/gemma-4-12b-it \
     --endpoint /v1/chat/completions --dataset-name random \
     --random-input-len 122880 --random-output-len 32 --num-prompts 1 --max-concurrency 1
   ```
3. Engine crashes with the `system_memory_manager.cpp:702` timeout during prefill.

Observed ISL envelope (real requests, single user):

| ISL | Result |
|-----|--------|
| 32K | pass |
| 64K | pass |
| 100K | pass (TTFT ~52s) |
| ~120K (122,893 tok) | **hang / device timeout** |

## Proposed fix

Implement **token-chunked prefill** for Gemma4 so the projection, per-head norms, RoPE,
and SDPA all run on bounded chunk lengths instead of the full sequence in one shot:

- Honor the Generator's per-chunk offset (`chunk_start_idx` / `chunk_page_table`) in
  `models/demos/gemma4/tt/model.py` / `attention/prefill.py` rather than discarding it.
- Drive each prefill chunk through `prefill_forward_single_user_text` with the correct
  cache offset so no single op operates on the full ~131K sequence.

This bounds the per-op program/dispatch work and should lift the prefill ceiling toward
the full `max_model_len`.

### Interim mitigation (not a fix)

`system_memory_manager.cpp:702` is a **timeout**, and `TT_METAL_OPERATION_TIMEOUT_SECONDS`
is currently `5.0`. Raising the timeout may let ~120K complete if the op is merely slow
rather than truly deadlocked — worth a quick experiment to confirm whether the limit is
time-bound or a hard stall, but it does not address the underlying full-sequence-op
scaling.

## Related

- tenstorrent/tt-metal#48283 — Gemma4: disable hybrid KV-cache groups (lifts the KV
  admission ceiling from ~23K to the full pool, which is what makes ~120K reach the
  prefill path where this hang occurs).
- tenstorrent/vllm#429 — vllm-tt-plugin: unwrap `UniformTypeKVCacheSpecs` (companion to
  the above).
