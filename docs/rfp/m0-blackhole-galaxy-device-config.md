# Milestone-0 — Blackhole Galaxy device-level configuration determination

**Readiness item:** 5.4 — Required — owner: Model bring-up
**RFP references:** requirements §D.1/§D.2 (fixed system + torus descriptor), §D.3 (logical mesh is the operator's choice), Appendix B.0 (device-level prerequisites for setting targets), readiness §5.4.

This is the `tt-inference-server`-side determination of the device-level configuration for the 32-chip Blackhole Galaxy, plus the concrete scaffolding it produces. It settles the four open items from §5.4 and lands the artifacts that unblock the sweep/target work (prod spec promotion + target authoring).

The system is fixed by the RFP: **32 × P150 in a 4×8 2D torus**, mesh graph descriptor
`tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto`
(`device_topology { dims:[8,4], dim_types:[RING,RING] }`, one host). The **logical mesh (`MESH_DEVICE`) is deliberately left to the operator** (§D.3); every decision below holds regardless of that choice.

---

## Summary

| # | Item | Decision |
| - | ---- | -------- |
| 1 | `max_context` | **Full native context**, no reduced pool. gemma-4-31B-it → **262144** (256K). DeepSeek/Mistral → full native (placeholder pending model availability). |
| 2 | `max_concurrency` | **128**, expressed **per system**. |
| 3 | `fabric_config` | **`FABRIC_1D_RING`** — pinned for Milestone-0 (matches the vLLM plugin's 6U-Galaxy default). |
| 4 | Data parallelism & target basis | Logical mesh / DP **left free** to the operator; targets **expressed and enforced per system** so the choice cannot move the bar. Implemented in code. |

---

## 1. `max_context` — full native context

**Decision: run at the model's full native context on the Galaxy; do not cap the KV pool.**

- gemma-4-31B-it → **262144 (256K)**, the model card native context.
- `mistralai/Mistral-Small-4-119B-2603` → **262144 (256K)** — model card states a 256k context window and the vLLM recipe serves `--max-model-len 262144`.
- `deepseek-ai/DeepSeek-V4-Flash-0731` → **1048576 (1M)** — a million-token-context model ("DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"), full native context; Partner to confirm `max_position_embeddings`.

Rationale:
- The agentic evals (Appendix B.4) send large combined input+output and need a matching server context; capping context below native would lose tasks to the context limit.
- The ~49K cap on the existing gemma-4-31B-it **P300X2** spec is a QB2-specific DRAM ceiling (hybrid-off, all 60 layers hold a full-length KV buffer). A 32-chip Galaxy has far more DRAM, so that cap does not carry over — direct precedent: `Qwen/Qwen3.6-27B` already runs `max_context: 262144` on just a 4-chip P300X2.

**Verify (Performance):** confirm the hybrid-off KV pool for gemma-4-31B fits Galaxy DRAM at 262144 with prefill-scratch headroom on-device. This is a confirmation of an already-made decision, not an open choice.

## 2. `max_concurrency` — 128, per system

**Decision: `max_concurrency: 128`, interpreted as a per-system value.**

- Matches the RFP's own worked example (Appendix F.3, "128 concurrent maximum-length requests") and the `gpt-oss` Galaxy specs.
- It is the sweep's upper concurrency corner (Appendix B.0/B.1). `DeviceModelSpec._infer_data` still divides it across data-parallel engines for each engine's `max_num_seqs` — correct serving mechanics, independent of the grading bar (item 4).

## 3. `fabric_config` — FABRIC_1D_RING (pinned)

**Decision: `fabric_config: FABRIC_1D_RING`, pinned for Milestone-0.**

- The vLLM plugin (`tt-vllm-plugin/.../tt_worker.py::get_fabric_config`) already **defaults to `FABRIC_1D_RING` on a Galaxy** (6U cluster), so this aligns the spec with the validated default rather than relying on implicit behaviour.
- Set explicitly in `override_tt_config` on every Milestone-0 `BLACKHOLE_GALAXY` entry so the fabric is deterministic and declared, not inferred.
- (A 2D fabric over the torus remains a future optimisation; it is intentionally out of scope for Milestone-0 and not used here.)

## 4. Data parallelism and the target basis — per-system, enforced in code

**Decision:** the logical mesh (and therefore the data-parallel degree) is left free to the operator (§D.3); **targets for `BLACKHOLE_GALAXY` are expressed and enforced per system, not per replica**, so the bar is identical across data-parallel choices.

Why this needed a code change. `workflows/model_spec.py::scale_llm_perf_targets` multiplies the aggregate throughput target by `data_parallel_size` and scales `max_concurrency` by it, deriving a per-system aggregate from a per-replica subdevice table. With that mechanism, two operators on the same physical system but different DP degrees would be graded against different bars — the readiness doc's fairness problem.

**Implemented (this branch):**
- `DeviceTypes.expresses_targets_per_system()` (`workflows/workflow_types.py`) → `True` for `BLACKHOLE_GALAXY`.
- `get_perf_reference` (`workflows/model_spec.py`) resolves such a device's targets **directly from its `blackhole_galaxy` key** and **skips the subdevice lookup and `data_parallel` scaling**, even when `data_parallel_size` is set.
- Serving mechanics untouched (`_infer_data` still splits per-system concurrency across DP engines). Only the grading target is made DP-invariant.
- Regression-guarded: the Wormhole Galaxy DP specs keep the per-replica ×DP behaviour. See `tests/test_perf_reference_per_system.py`.

**Consequence for authoring (sweep/target work):** the `blackhole_galaxy` targets in `model_performance_reference.json` are absolute **per-system** values established from AIPerf (readiness §5.3), not per-replica figures to be scaled up.

---

## Scaffolding produced by this issue

Three `BLACKHOLE_GALAXY` device specs added to the **dev** catalog
(`workflows/model_specs/dev/llm.yaml`) — the unpinned dev→prod staging catalog
(`scripts/release/promote_dev_spec_to_prod.py`). Prod is not touched: the greenfield
models have no `tt_metal_commit` to pin, and prod entries would emit bogus release
images. All three pin `TT_MESH_GRAPH_DESC_PATH` to the fixed torus descriptor,
pin `fabric_config: FABRIC_1D_RING`, use per-system `max_concurrency: 128`, and
leave `MESH_DEVICE` to the operator.

| Model | Impl | Nature | `max_context` |
| ----- | ---- | ------ | ------------- |
| `google/gemma-4-31B-it` | `tt_transformers` (real) | Working starting config: gemma-4 serves today; tool-call/reasoning/thinking preserved. | 262144 (256K, full) |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `deepseek_v4_flash` (**stub impl**) | Greenfield scaffold, `[TBD — Partner]` on impl/parsers (no serving path exists, readiness §6.2). | 1048576 (1M native) |
| `mistralai/Mistral-Small-4-119B-2603` | `tt_transformers` (placeholder) | Greenfield scaffold, `[TBD — Partner]` on impl/parsers. `mistral3` MoE, multimodal (text-only for M0). | 262144 (256K, full) |

`deepseek_v4_flash` is a new stub `ImplSpec` (`workflows/model_spec.py`) whose
`code_path` points at the expected tt-metal home so the catalog loads; the Partner
replaces it with the implementation they contribute.

## Downstream

- **Prod spec promotion** (promote/author the prod `BLACKHOLE_GALAXY` specs): promote these dev entries with pinned commits once AIPerf targets exist.
- **Sweep + targets authoring**: author `blackhole_galaxy` targets as absolute per-system values (item 4).
- **[Scaling-quality coverage (§5.7)](m0-scaling-quality-sweep-coverage.md)**: the `max_context`/`max_concurrency` chosen here interact with the three-input-lengths-per-graded-concurrency rule. With the token pool defaulted to `max_context`, only ISLs 128 and 1024 reach concurrency 128 — so the Milestone-0 specs carry a `[TBD — Performance] max_tokens_all_users_override` (floor 278528) that must be sized to the measured 32-chip KV pool.
- Feeds Appendix B.0 of the partner RFP (`max_context`, `max_concurrency`, data-parallelism rows).
