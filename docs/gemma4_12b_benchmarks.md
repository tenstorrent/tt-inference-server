# Gemma 4 12B — vLLM serving benchmarks (P300x2, hybrid-KV-off)

Updated benchmark sweep for `google/gemma-4-12b-it` on QB2 (P300x2 / Blackhole),
measured against a live `vllm bench serve` client with the **hybrid-KV-disabled**
configuration that lifts the KV admission ceiling from ~23K to the full 131K pool.

## Environment

| | |
|---|---|
| Model | `google/gemma-4-12b-it` |
| Hardware | P300x2 (Blackhole), MeshDevice `1x4`, TP=4 |
| tt-metal | `a4967d5f39d` + hybrid-KV-off (tenstorrent/tt-metal#48283) |
| vLLM (TT fork) | `9d88cd5` (`v0.1.dev14166+g9d88cd582`) + `UniformTypeKVCacheSpecs` unwrap (tenstorrent/vllm#429) |
| `max_model_len` / `max_num_batched_tokens` | `131072` / `131072` |
| `max_num_seqs` | `1` |
| `block_size` | `64` |
| Trace / sampling | `trace_mode=all`, `sample_on_device_mode=decode_only` |
| KV cache | hybrid groups disabled → single `UniformTypeKVCacheSpecs` pool, **131,136 tokens**, **1.00x** concurrency |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | `5.0` |

## Methodology

Each data point: a single request (`--num-prompts 1 --max-concurrency 1`), random
dataset, **128 output tokens**, OpenAI chat-completions endpoint:

```bash
vllm bench serve --backend openai-chat --model google/gemma-4-12b-it \
  --endpoint /v1/chat/completions --host localhost --port 8000 \
  --dataset-name random --random-input-len <ISL> --random-output-len 128 \
  --num-prompts 1 --max-concurrency 1
```

## Results

| ISL | TTFT (s) | TPOT (ms) | Decode rate (1/TPOT, tok/s) | End-to-end output (tok/s) | Status |
|-----|----------|-----------|------------------------------|----------------------------|--------|
| 2K   | 8.7  | 41.3 | 24.2 | 9.19 | pass |
| 4K   | 9.1  | 41.3 | 24.2 | 8.91 | pass |
| 8K   | 9.8  | 41.7 | 24.0 | 8.49 | pass |
| 16K  | 14.1 | 41.7 | 24.0 | 6.61 | pass |
| 32K  | 27.5 | 42.7 | 23.4 | 3.89 | pass |
| 64K  | 47.7 | 44.3 | 22.6 | 2.40 | pass |
| 100K | ~52 (standalone) | — | — | — | intermittent device hang (see below) |

### Observations

- **Decode cost is essentially context-independent.** TPOT drifts only 41.3 → 44.3 ms
  from 2K → 64K (~23–24 tok/s decode). The "end-to-end output tok/s" column falls with
  ISL only because the long prefill (TTFT) dominates a 128-token output window, not
  because per-token decode slows down.
- **TTFT scales ~linearly with prefill length above ~8K.** Below that there is a
  ~8.5 s fixed floor (chat-template processing + per-request setup) that dominates small
  ISLs, which is why 2K/4K/8K TTFT are all clustered around 9 s.
- KV admission is never the limiter at single-request concurrency: the full 131,136-token
  pool admits every ISL up to `max_model_len`.

## 100K / large-prefill intermittency

100K is **reachable but flaky**. It is gated by the eager-prefill device deadlock
documented in `gemma4_prefill_fabric_hang_issue.md` (tenstorrent/tt-metal#48289), which
manifests as:

```
RuntimeError: TT_THROW @ tt_metal/impl/dispatch/system_memory_manager.cpp:702
TIMEOUT: device timeout in fetch queue wait, potential hang detected
```

This is **not a KV / admission / OOM limit** (the request reaches `Running` with
`num_scheduled_tokens` equal to the full ISL) and **not a clean ISL ceiling** — it is a
probabilistic device-side stall in the un-chunked full-sequence prefill ops (per-head
RMSNorm), which becomes likelier at larger prefills and after a run of requests:

| Run | Clean passes | First hang |
|-----|--------------|-----------|
| Sweep A | 2K–16K | **32K** (`num_scheduled_tokens=32781`) |
| Sweep B | 2K–64K | **100K** (`num_scheduled_tokens=102413`) |
| Standalone (fresh device, earlier) | 32K, 64K, 100K all passed | ~120K |

A 32K prefill is only ~1–2 s of real compute, so the 5 s op timeout firing there means a
genuine deadlock rather than a "slow op" — i.e. raising `TT_METAL_OPERATION_TIMEOUT_SECONDS`
would not reliably fix it. Each hang kills the container and wedges the device (requires
`tt-smi -r` before the next run).

## Summary

**Gemma 4 12B serves reliably up to 64K** input tokens on P300x2 with the hybrid-KV-off
config, at a steady ~23–24 tok/s decode and TTFT scaling linearly with prefill. 100K is
achievable but intermittently triggers the eager-prefill fabric deadlock; landing it
reliably requires token-chunked prefill (tenstorrent/tt-metal#48289 proposed fix).

## Related

- `docs/gemma4_prefill_fabric_hang_issue.md` — large-prefill device hang (tenstorrent/tt-metal#48289)
- tenstorrent/tt-metal#48283 — disable hybrid KV-cache groups (lifts KV ceiling 23K → 131K)
- tenstorrent/vllm#429 — vllm-tt-plugin: unwrap `UniformTypeKVCacheSpecs`
