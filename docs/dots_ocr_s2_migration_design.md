<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# dots.ocr S0 → S2 migration: continuous-batched paged serving design

**Status:** design / plan only — no code written. Output of an end-to-end read of
the dots.ocr pipeline, the tiered tt_symbiote vLLM adapter, the Tenstorrent vLLM
TT plugin runner/scheduler, and the paged-attention caches in both
`tt_symbiote.models.dots_ocr._attention` and `tt_symbiote.modules.ttnn_attention`.

**Goal (agreed):** drive dots.ocr **amortized throughput** to ~**1 s/page** — i.e.
a 49-image batch should finish in ~49 s of wall clock, not 49 × 5 s. The lever for
that is **vLLM continuous batching**, which requires moving dots.ocr from serving
tier **S0_GREEDY_ENGINE** to **S2_PAGED**.

**Non-goal:** reducing *single-page* wall-clock latency. S2 does not help (and can
hurt) a single isolated request — see §2. Single-stream latency has a separate
lever set (on-device sampling / async decode) tracked elsewhere.

---

## 1. Why S2 is the right destination for amortized throughput

The amortized cost of a token step is `host_overhead / sequences_in_flight`. S0
pins `sequences_in_flight = 1` (`max_num_seqs=1`); every page pays the full
per-token host round-trip alone. S2 lets vLLM keep **N independent pages decoding
through one shared device step**, so the fixed host overhead (sampling, scheduler,
readback) is divided by N. With N≈8–32 in flight, the ~5 s/page single-stream cost
amortizes toward the ~0.6–1 s/page target.

This is a *throughput* result, not a latency result. Each individual page still
takes ~5 s end-to-end; we just run many at once.

---

## 2. Why S2 does **not** reduce single-page latency (premise check)

Per **decode token**, the host-boundary traffic is:

| Path | Per-token host work |
|---|---|
| **S0 today** | blocking d2h readback of **1 token** (`pipeline.py:1344-1353`) → host allocates+zeros a `[1,1,152064]` f32 **one-hot** (`tt_symbiote_generators.py:538-548`) → vLLM **argmax over 152K on host** → feed token back |
| **S2** | d2h of the **full 152,064-wide logits** every token → vLLM samples on host |
| **native `pipeline.generate()`** | none — deferred/pipelined/batched readback (`pipeline.py:1514-1594`) + on-device token feedback (`:1320`) |

For a **single** stream, S2 moves *more* data per token than S0 (full logits vs one
token) and is the same or slower. S2 wins only when the batch is full. The
single-page latency levers (keep argmax + token feedback on-device, enable async
decode) are orthogonal to the tier and out of scope for this doc.

---

## 3. Current S0 architecture (what we are migrating away from)

```
vLLM engine (max_num_seqs=1, greedy)
  └─ _TTSymbioteGenerator (tier=S0_GREEDY_ENGINE)
       prefill_forward → _prefill_s0  → pipeline.prefill(ids, pixel_values, grid)  → token → one-hot logits
       decode_forward  → _decode_s0   → pipeline.decode_step(prev_token)           → token → one-hot logits
       allocate_kv_cache → None   (model owns its KV; vLLM page table ignored)
```

dots.ocr's TTNN compute lives in **standalone graph objects**
(`TTNNDotsOCRPrefillGraph`, `TTNNDotsOCRDecodeGraph`) whose `forward` ends in
`_argmax_token_on_device(...)` and returns a **token**, not logits
(`pipeline.py:440, 508`). State that makes S0 single-stream:

- **One global decode position.** `_decode_cache_position` is a single replicated
  counter advanced on-device once per step (`pipeline.py:1176-1183, 1264-1294`).
  All streams share it — strict lockstep.
- **Internal fixed page table.** dots.ocr's cache builds `page_table = arange(...)`
  (`models/dots_ocr/_attention.py:96-97`) and writes with `batch_idx=0`
  (`:170-171`). It has **no `set_vllm_page_table`** — that hook exists only on the
  *separate* `modules/ttnn_attention.py` class (`:151`).
- **`hf_model` is the original torch model.** The S2 adapter calls
  `self.hf_model(input_ids, past_key_values=cache, ...)`
  (`tt_symbiote_generators.py:481-488, 512-517`). For dots.ocr that HF forward is
  **CPU torch** — the TTNN path is *not* reachable through it.

These three facts are the migration.

---

## 4. Target S2 architecture

```
vLLM engine (max_num_seqs=N, continuous batching, vLLM samples)
  └─ _TTSymbioteGenerator (tier=S2_PAGED)
       allocate_kv_cache(num_blocks, num_kv_heads, block_size, head_size) → dots.ocr paged KV (vLLM-sized)
       prefill_forward → _prefill_s2 → set_vllm_page_table(pt) → dots_ocr_paged_forward(ids, cache, cache_position) → LOGITS [B,S,V]
       decode_forward  → _decode_s2  → set_vllm_page_table(pt) → dots_ocr_paged_forward(tok, cache, start_pos)      → LOGITS [B,V]
```

vLLM owns: scheduling, block allocation, **dynamic page tables per step**,
per-sequence positions, EOS removal, and sampling. The TT runner already batches
multiple prefills into one call, pads decode to `max_num_seqs`, and removes
finished sequences each step (verified in the runner exploration). dots.ocr must
become a model that *fits* that contract.

---

## 5. The structural gap (four mismatches to close)

| # | Contract S2/vLLM requires | dots.ocr today | Fix lives in |
|---|---|---|---|
| G1 | model `forward` returns **logits** | graphs return on-device **argmax token** (`pipeline.py:440,508`) | tt_symbiote |
| G2 | **per-sequence** cache positions; **variable** batch that shrinks on EOS | one **global** replicated position, fixed lockstep batch (`pipeline.py:1176-1183`) | tt_symbiote |
| G3 | honor vLLM's **dynamic page table** each step | fixed internal `arange`, `batch_idx=0`; **no `set_vllm_page_table`** (`_attention.py:96-97,170`) | tt_symbiote |
| G4 | multimodal prefill under **continuous batching** (independent images, mixed prefill/decode) | vision+scatter assumes 1 image (or 1-per-DP-stream same-grid) (`pipeline.py:933-1124`) | tt_symbiote + adapter |

---

## 6. tt_symbiote-side workstreams (the heavy lift)

### W1 — Paged-logits forward (closes G1)
Add a serving forward that returns logits, reusing the existing TTNN modules
(embedding → decoder stack → final norm → lm_head) **without** the terminal
`_argmax_token_on_device`. Two viable shapes:
- (a) A new `forward_logits(...)` on the pipeline that runs the decoder stack and
  lm_head and returns the `[B, S, vocab]` logits tensor (host or on-device), or
- (b) Wire the TTNN decoder/lm_head into the HF `DotsOCRForCausalLM.forward` so the
  adapter's existing `self.hf_model(...)` call (`tt_symbiote_generators.py:481`)
  executes on TTNN. (b) matches the adapter as-written but is more invasive to the
  HF module; (a) is cleaner but needs a small adapter change to call the pipeline
  forward instead of `hf_model`.

**Recommendation:** (a) — keep TTNN compute in the pipeline, add a thin
`pipeline.forward_logits` and have `_prefill_s2/_decode_s2` call it when the model
exposes `_tt_pipeline`. Mirrors how S0 already special-cases the pipeline.

### W2 — Per-sequence positions + variable batch (closes G2)
- Replace the single global `_decode_cache_position` with a **per-row position
  vector** sourced from vLLM's `start_pos` (the runner passes
  `input_positions = num_tokens-1` per row). The decode SDPA already takes a
  `cache_position` (`_attention.py:260-270`); it must accept per-row values.
- Support a **batch that shrinks**: the runner removes finished sequences and pads
  decode to `max_num_seqs` with `start_pos = -1`. The forward must treat `-1`
  rows as inactive (no KV write/read) and tolerate the active count changing
  step-to-step. No internal lockstep counter, no `stop_on_eos` loop — vLLM owns
  termination.

### W3 — vLLM dynamic page table on dots.ocr's cache (closes G3)
dots.ocr's cache (`models/dots_ocr/_attention.py`) needs the **`set_vllm_page_table`
hook** that today exists only on `modules/ttnn_attention.py:151`. Two options:
- (a) **Port** dots.ocr's pipeline onto the shared
  `modules.ttnn_attention.TTNNPagedAttentionKVCache` (one cache class, the hook is
  already T3K-validated — design §9.2 of the integration doc), or
- (b) **Add** an equivalent `set_vllm_page_table` to dots.ocr's cache class and
  replace the `arange` table + `batch_idx=0` fills with vLLM's block ids.

**Recommendation:** (a) if the dots.ocr paged ops are compatible with the shared
class (they call the same `paged_fill_cache` / `paged_sdpa_decode` ops); else (b).
Unifying avoids two divergent paged caches.

### W4 — Multimodal continuous-batched prefill (closes G4 — hardest)
Today vision runs once per prefill over one image (or one-per-DP-stream, same grid).
Under continuous batching, a single prefill step can carry **several different
images at different grids**, interleaved with text-only and decode requests. Work:
- Run the vision tower + scatter-fuse **per request within a batched prefill**, then
  hand fused embeddings into the paged decoder prefill keyed by each request's page
  table slice.
- Decide image-token placeholder accounting against vLLM's multimodal pipeline
  (the native processor is already reused via `_copy_native_multimodal_registration`,
  `tt_symbiote_generators.py:591-640`).
- **Interim simplification:** gate S2 to a fixed input geometry (the letterboxed
  1848×1176 we already enforce) so all images share a grid — collapses the
  multi-grid problem for milestone 1.

---

## 7. tt-inference-server-side workstreams (small, mostly config)

| # | Change | File |
|---|---|---|
| T1 | Flip `serving_tier` → `S2_PAGED` for dots.ocr (adapter auto-routes) | tt_symbiote `models/_runtime_pins.py` (read at `tt_symbiote_generators.py:104-116`) |
| T2 | Make `_prefill_s2`/`_decode_s2` call the new `pipeline.forward_logits` when `_tt_pipeline` is present (per W1a) | `tt_symbiote_generators.py:462-520` |
| T3 | Flow `pixel_values`/`image_grid_thw` through `_prefill_s2` (today only `_prefill_s0` flattens them) | `tt_symbiote_generators.py:384-391, 462-470` |
| T4 | `allocate_kv_cache`: confirm the 4-tuple `(num_blocks, num_kv_heads, block_size, head_size)` builds dots.ocr's paged cache (per W3) | `tt_symbiote_generators.py:240-296` |
| T5 | `vlm.yaml`: raise `max_num_seqs` (e.g. 8–32), drop S0 single-stream notes, enable on-device sampling so vLLM reads tokens not 152K logits | `workflows/model_specs/dev/vlm.yaml` |
| T6 | Remove the standalone A1a `--dp-batched` path once S2 lands (or keep as fallback) | `run.py`, `run_docker_server.py`, `run_dots_ocr_batched_server.py` |

No Docker rebuild is required for tier/config flips beyond shipping the new
tt_symbiote version that contains W1–W4.

---

## 8. Risks & open questions

1. **Vision prefill cost dominates at high concurrency.** Decode amortizes across N,
   but each page still pays a full vision prefill (~2.8K–11K patch tokens). If
   prefill is the bottleneck, continuous batching of decode won't reach 1 s/page;
   measure the prefill/decode split first.
2. **KV capacity.** dots.ocr keeps ≥64 pages/stream; the vision prompt is large
   (`_attention`/`pipeline.py:559-566`). vLLM's block budget at `max_num_seqs=N`
   must fit N × (vision + output) tokens — may cap practical N below 8.
3. **bf16 KV is mandatory** — bf8 corrupts dots.ocr decode (`pipeline.py:543-553`);
   the vLLM-sized cache must stay bf16, doubling KV memory vs text LLMs.
4. **Trace capture under variable batch.** The decode graph is traced for a fixed
   shape; a variable active count / `reset_batch` may force re-capture or a
   padded-to-`max_num_seqs` fixed trace. Confirm traced decode survives the runner's
   per-step layout changes.
5. **Greedy-only today.** dots.ocr argmaxes; under S2 vLLM samples real logits.
   Keep greedy (temperature 0) for output parity with the validated S0 results,
   else OCR accuracy must be re-validated.
6. **Two paged-cache classes** (W3) — unify or the hook/correctness work doubles.

---

## 9. Validation plan

1. **Token parity:** S2 single-stream output must match S0 greedy output token-for-
   token on the 49 sample_docs (guards the logits refactor W1/W3).
2. **Concurrency scaling:** measure amortized s/page at N = 1, 2, 4, 8 (and 16/32 if
   KV fits); expect ~linear throughput until prefill- or KV-bound.
3. **Mixed-batch soak:** ≥50 images with staggered arrival so prefill and decode
   interleave (exercises G2 EOS-removal + G4 MM prefill).
4. **Prefill/decode profiling:** `DOTS_OCR_PROFILE_SYNC` split to confirm where the
   per-page time goes and whether 1 s/page is reachable given vision prefill.

---

## 10. Phased milestones

- **M0 — measure.** Profile the current 5 s/page into vision-prefill vs decode.
  Decides whether decode-batching alone can hit 1 s/page (risk §8.1). *Cheap, do first.*
- **M1 — S2 text/decode path, fixed geometry.** W1a + W2 + W3 + T1–T5, with the
  letterboxed single-grid simplification (W4 interim). Validate token parity +
  concurrency scaling at N=8. This is the bulk of the throughput win.
- **M2 — true multimodal continuous batching.** Full W4 (multi-grid, interleaved
  MM prefill). Lifts the geometry constraint.
- **M3 — retire A1a.** Remove the standalone DP=8 server (T6) once S2 meets/exceeds
  its throughput; keep one serving path.

---

## 11. Alternatives considered

- **A1a standalone DP=8 batched server (already built).** Achieves ~8× amortized
  throughput today via `pipeline.generate` over a static 8-batch — *no tier change*.
  It is the **fastest route to ~0.6 s/page amortized right now** and a sound interim
  while S2 (M1) is built. S2's advantage over A1a is *dynamic* admission (no waiting
  for a full batch of 8, no head-of-line blocking on a slow page) and integration
  with vLLM's standard OpenAI surface.
- **A2 vLLM-native static lockstep batch.** Forces static batching into the shared
  TT scheduler; high regression risk to continuous-batching text models, and still
  not true continuous batching. Rejected in favor of the proper S2 model work.

---

## References

- `tt-inference-server/vllm-tt-metal/src/tt_symbiote_generators.py` — tiered adapter
  (S0/S1/S2 dispatch, one-hot bridge, S2 paged forward, MM registration).
- `tt_symbiote/src/tt_symbiote/models/dots_ocr/pipeline.py` — the token-emitting
  prefill/decode graphs, global decode position, DP batched generate loop.
- `tt_symbiote/src/tt_symbiote/models/dots_ocr/_attention.py` — dots.ocr's paged KV
  (fixed `arange` table, `batch_idx=0`, no vLLM hook).
- `tt_symbiote/src/tt_symbiote/modules/ttnn_attention.py:151` — the validated
  `set_vllm_page_table` hook (target to unify onto).
- `tt_symbiote/docs/development/tt_inference_server_integration.md` §9 — the tier
  model and the S2 page-table-hook design + T3K validation status.
- `docs/dots_ocr_dp_batching_design.md` — the A1a/A2 batched-serving analysis (interim).
