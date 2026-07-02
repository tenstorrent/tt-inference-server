# Known Limitations & Fixed Bugs (Gemma 4 on QB2 / P300X2)

Current state of the Gemma 4 12B/31B integration on QB2 (P300X2, driven as a
`P150x4` 4-chip TP mesh). Read before promising a context length or debugging a
"why can't we serve X" question.

## Context / token limits

### 31B — `max_context: 49152` (hard cap)
- Hybrid KV cache is **off**, so all 60 layers allocate a full-length KV buffer.
  The largest pool that fits QB2 DRAM (with prefill scratch headroom) is ~49K.
- Uncapped (131072) it **OOMs at boot** during KV allocation (~layer 31/60).
- Servable prompt tops out at the **32768** power-of-2 prefill bucket. The next
  bucket (65536) exceeds the 49152 pool and trips the **page-table
  "negative dimension" fault**.
- Enforced via spec: `max_context: 49152` + `GEMMA4_MAX_TOKENS_ALL_USERS: "49152"`
  (sizes the all-user KV pool via `Gemma4ForCausalLM.get_max_tokens_all_users`).

### 12B — `max_context: 131072` (but real limit is lower)
- Hybrid-off, the ~131K all-user KV pool fits DRAM, so the page-table fault
  cannot occur (every prefill bucket ≤131072 fits) and no
  `GEMMA4_MAX_TOKENS_ALL_USERS` override is set.
- The real ceiling is the **>64K eager-prefill fabric hang** (tt-metal#48289).
  Guarded at the eval layer with `max_input_tokens = 64K`, which pads to the
  65536 bucket (largest validated).
- `max_context` stays 131072 (not a hard 65536 cap) to leave decode block
  headroom above a 64K-input prompt.

## Architectural constraints (must-set config)
- **`MESH_DEVICE: P150x4`** — the custom `p300_x2` descriptor laid TP collectives
  over the wrong fabric links and corrupted decode logits. The default
  `p150_x4` (1,4) descriptor is correct and matches the passing vLLM nightly.
- **`sample_on_device_mode: decode_only`** — required so cross-shard argmax runs
  on device; otherwise host sampling only sees device 0's vocab shard and token
  ids ≥ 65536 are unreachable in decode.
- **`fabric_config: FABRIC_1D`**, `GEMMA4_PAGE_BLOCK_SIZE: 64`.
- **Eager prefill** (no prefill trace); prefill trace is unused for these models.
- **transformers 5.x** required for the `gemma4_unified` architecture (pinned in
  the tt-metal `python_env` as of tt-metal #47817 / #47172). A 4.x transformers
  raises *"model type gemma4_unified but Transformers does not recognize this
  architecture"* — caught by a fail-fast preflight.

## Open hardware/software issues
- **Fabric hang — tt-metal#48289.** >64K eager prefill wedges an ethernet core
  (*"Timed out while waiting for active ethernet core"*), a FABRIC_1D NOC
  read-response stall. Recovery: `tt-smi -r`. Mitigated by the 64K eval-layer
  input cap.
- **Hybrid KV cache disabled for Gemma 4.** Chunked prefill is implemented on
  the tt-metal side (PR #45355) but **not enabled on the inference-server vLLM
  plugin side**. With hybrid on, vLLM's `SlidingWindowSpec` sets num_tokens to
  `max_num_batched_tokens` and the KV pool is split evenly across 6 groups
  (~23K/group), so the full-attention layer can't get the large group it needs.
  Gemma 3 sidesteps this by disabling hybrid KVC — which is what we do here too
  (`GEMMA4_BOUNDED_SLIDING_KV_CACHE=0` / hybrid-off), at the cost of the smaller
  ~49K pool on 31B.

## Big bugs we fixed (don't reintroduce)
- **vLLM plugin clobbered the TT vLLM with a CUDA wheel.** The
  `vllm-tt-plugin[runtime]` extra declared an unconstrained `vllm` dep + torch/
  transformers pins, so uv backtracked to an ancient PyPI **`vllm 0.2.5` (CUDA)**
  wheel, clobbering the `VLLM_TARGET_DEVICE=empty` editable build → server
  crashed on boot with `ImportError: libcudart.so.12`. **Fixed upstream in
  tenstorrent/vllm #433** (plugin no longer re-pulls vllm / restates torch+
  transformers). We pin vllm `375df057` (dev HEAD = #431 + #433). Do **not**
  patch the shared Dockerfile to "reinstall vllm after the plugin".
- **Gemma4 tokenizer failed in the benchmark client.** `vllm bench serve`
  pulled transformers 4.x and hit `AttributeError: 'list' object has no
  attribute 'keys'`. Fixed by forcing `transformers>=5.10.2,<6` in the
  `.venv_benchmarks_vllm` venv via a per-venv `uv --override`
  (`requirements/benchmarks-vllm-overrides.txt`). Contained to that venv only.
- **Tool parser missing / inert → SWE-Bench 0.00.** Gemma4 reasoning+tool
  parsers were originally baked into the image. Moved into `vllm-tt-plugin`
  (tenstorrent/vllm #431). The tool parser stays inert unless the server runs
  with `--enable-auto-tool-choice`; now set via spec `vllm_args`
  (`enable-auto-tool-choice: true`) so a flag-free release/nightly works.
- **Page-table negative-dim fault vs KV-cache boot OOM** — two distinct failure
  modes on 31B; both resolved by the 49152 cap + pool sizing above.

## Model status
- Both 12B and 31B are `EXPERIMENTAL` on P300X2 (no enforced perf tiers).
