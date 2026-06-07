# Adding gemma-4-31b-it (Forge, tensor-parallel, p300x2) to tt-shield CI

Issue-draft / checklist of everything required to onboard **gemma-4-31b-it** as a
Forge LLM (4-chip tensor-parallel on p300x2 / QB2) into the tt-shield nightly +
release CI. Companion to `PERF_gemma4_31b_it_forge.md` (perf/bring-up detail).

Branch: `kmabee/gemma4_31b_it_forge`.
Serving image: `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:aed6177ab424aa93f447118aac5a7b8ab1cafdbc_f752cce_79853177306`
(tt-forge **1.3.0.dev20260605003323**; built by tt-shield `on-dispatch-build-media-server`
from this branch).

> **Scope of this add (decoupled from the benchmark-client uplift):** this ships gemma-4-31b-it
> as **servable + eval-able now** so perf work can start testing against it immediately. The
> **e2e CI benchmark step will NOT pass until the separate `BENCHMARKS_VLLM` vllm/transformers
> uplift lands** (`ISSUE_benchmarks_vllm_uplift.md`, summarized in §B) — that uplift is
> intentionally **not** part of this add (it's a wide, shared change needing its own
> qualification). Until then gemma's release-flow *benchmark* step is a known-failing step;
> *serving* and *evals* are unaffected.

---

## A. Pieces in THIS add (done on this branch — serving + evals; NOT benchmarks)

| # | Area | Change | File(s) |
|---|---|---|---|
| 1 | Server runner (TP) | `VLLMForge_GEMMA4_31B` runner; env-var tunable (ENABLE_TRACE/CPU_SAMPLING/OPTIMIZATION_LEVEL), perf defaults trace-on / device-sampling / bfp8 | `tt-media-server/tt_model_runners/vllm_forge_gemma4_31b.py` |
| 2 | Server config | `(VLLMForge_GEMMA4_31B, P300X2)` = TP mesh topology only (mesh (1,4), device_ids, max_batch_size); dims env-driven | `tt-media-server/config/constants.py` |
| 3 | Forge wheel | image built with `tt-forge==1.3.0.dev20260605003323` | `tt-media-server/tt_model_runners/forge_runners/requirements.txt` |
| 4 | Model spec (**dev** catalog — CI reads dev) | gemma-4-31b-it forge/p300x2 entry; batch-16/4K via `env_vars` (MAX_MODEL_LENGTH/MAX_NUM_SEQS/GPU_MEMORY_UTILIZATION + trace/sampling/opt + p300_x2 mesh descriptor) | `workflows/model_specs/dev/cnn.yaml` |
| 5 | Evals | `ifeval`, downsampled (CI_NIGHTLY 0.1 / SMOKE 0.01); first TT user → published/gpu refs TBD | `evals/eval_config.py` |
| 6 | Perf reference | p300x2 128/128 conc-1 target (placeholder, borrowed from Qwen3-32B) | `benchmarking/benchmark_targets/model_performance_reference.json` |
| 7 | Nightly CI matrix | gemma-4-31b-it FORGE nightly on P300X2 | `.github/workflows/models-ci-config.json` |

> The benchmark-client `BENCHMARKS_VLLM` uplift is **not** in this table — it is a separate
> prerequisite for benchmarks only (§B), shipped independently.

### Key gotchas (learned during bring-up)
- **CI reads the `dev` catalog**, not prod: the shared tt-shield run job runs `run.py --dev-mode`
  unconditionally (`workflow_run-tests-with-inference-server.yml:327`). Local default `run.py` uses prod.
- **`model_name` must be lowercase** `gemma-4-31b-it` (case-sensitive `ModelNames.GEMMA_4_31B_IT`).
- **`TT_MESH_GRAPH_DESC_PATH` must be pinned** to the `p300_x2` descriptor (the runner doesn't auto-set
  it; the image maps `p300x2 -> p150`, which otherwise gives StableHLO "unknown mesh: @mesh").
- **Dims are env-driven** (cnn.yaml `env_vars`), mirroring the single-chip forge LLMs; `constants.py`
  carries only TP topology, and `model` comes from the `MODEL` env. → dev/prod can differ, no rebuild to retune.
- **4-chip connected mesh required** (degree `{2:4}`); a 2-chip mesh OOMs / topology won't map.

---

## B. PREREQUISITE for e2e benchmarks — benchmark client uplift (separate issue/PR, NOT in this add)

**Required for gemma's CI benchmark step to run, but intentionally excluded from this add.** It's a
wide, shared change (see `ISSUE_benchmarks_vllm_uplift.md`); gemma *serving* + *evals* work without it,
only the *benchmark* step is blocked. Shipping the add first unblocks perf work; the uplift follows on
its own branch/PR once qualified. **This is NOT gemma-specific — it affects every model on the default
`tools=vllm` benchmark path.**

- Symptom (gemma-4): all benchmark runs crash — `AttributeError: 'list' object has no attribute 'keys'`
  in `transformers/tokenization_utils_base.py:_set_model_specific_special_tokens`. gemma-4's
  `tokenizer_config` has `extra_special_tokens: ['<|video|>']` (a **list**); transformers 4.x expects a dict.
- Root cause: `requirements/benchmarks-vllm.txt` pinned `vllm==0.13.0`, which hard-pins `transformers<5`
  → 4.57.6. The serving image (and tt-xla qualification) uses **transformers 5.5.1 + vllm 0.19.1**, which
  loads the list form fine.
- Fix (prototyped + validated, then **reverted off this branch** in commit `6e6c22bb`→`7ef618a9` to
  keep this add independent; cherry-pick `6e6c22bb` for the uplift PR): bump the shared `BENCHMARKS_VLLM`
  venv to **vllm 0.19.1 + transformers 5.5.1** and `VLLM_PIN_VERSION 0.13.0 → 0.19.1`.
- **Scope / action:** the `BENCHMARKS_VLLM` venv is shared by ALL models (it's the default `vllm bench
  serve` client). This bump (incl. a transformers 4→5 major) therefore needs its **own issue + PR**, and
  must be **qualified against the existing single-chip forge LLMs** (Qwen3-4B, Llama-3.2-3B,
  Llama-3.1-8B, Qwen3-8B, Falcon3-7B on P150) + other benchmarked models before merge — not bundled into
  the gemma onboarding PR. Recommend cherry-picking `6e6c22bb` onto its own branch.

---

## C. Open blockers / next steps (per model)

### gemma-4-31b-it
- ✅ **In this add:** serving (4K/16) + evals work on CI. *(local: serve ✅, ifeval 0.89 ✅)*
- ⛔ **benchmark step blocked pending the §B uplift** (separate PR) — NOT a gemma-add task. The
      `extra_special_tokens`-list tokenizer crash is fixed by the uplift (validated locally: tokenizer
      loads + `vllm bench serve` runs on vllm 0.19.1 / transformers 5.5.1). Until the uplift merges,
      treat gemma's release-flow benchmark as known-failing; serving/evals/manual perf are unaffected.
- [ ] fill eval published/gpu reference scores from the first clean nightly (currently None).
- [ ] replace the placeholder perf-reference targets with measured gemma-4 numbers.
- [ ] decide prod-catalog promotion (currently dev-only, `status: EXPERIMENTAL`).

### Qwen3-32B (sibling, same setup)
- [ ] **release benchmark exceeds the 6h CI job cap** (run #5204 cancelled). Root cause: conc-1 benchmark
      runs are ~16× slower than aggregate at a batch-16 config (single request runs the batch-16 graph,
      ~0.3 tok/s), and the sweep runs conc=1 for every isl/osl pair. No clean in-branch lever — see
      `PERF_gemma4_31b_it_forge.md`. Options: tt-shield per-model `timeout-minutes`; skip conc-1 for
      forge-TP in `benchmark_config.py`; or run `workflow=evals` for now.
- [ ] decide batch-16/4K (from gemma sweep) vs 512/conc-1 for Qwen (never independently swept).

---

## D. CI evidence so far
- Build: tt-shield `on-dispatch-build-media-server` #268 ✅ (forge image from this branch).
- Release dispatch gemma `#5203`: serve ✅ + eval ✅; benchmark ❌ (tokenizer crash — expected until the
  §B uplift; not blocking this add's serving/eval scope).
- Release dispatch Qwen3-32B `#5204` ⏱️ cancelled @6h (benchmark runtime — separate, see §C).
- Local (p01t05, new image): both serve ✅ at 4K/16; gemma smoke eval ifeval 0.89 ✅; Qwen smoke
  eval/benchmark functional ✅. Benchmark tokenizer fix validated locally under the §B uplift (which is
  reverted off this branch); gemma e2e benchmark only runs once the uplift lands.
