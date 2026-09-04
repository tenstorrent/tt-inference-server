# Quetzal-generated GPT-OSS-120B — canonical SWE attempt (BLOCKED at fail-closed admission)

Overnight attempt (2026-09-03/04) to run the full SWE flow for **Quetzal-generated
`openai/gpt-oss-120b`** via canonical `run.py --impl quetzal` on a dedicated exabox QB2
node (`qb2-120-p04t08`, job 76030, 4× Blackhole p300c / mesh p150x4).

## Result: BLOCKED at serve (honest, no green theater)

The canonical serve stops **before** vLLM launch / weight load at the fail-closed
GPT-OSS-120B Quetzal topology-admission precheck:

```
run.py:1103 -> workflows/quetzal_topology_admission.py:306
QuetzalTopologyAdmissionError: GPT-OSS-120B Quetzal requires QUETZAL_TOPOLOGY_ADMISSION_JSON
```

`validate_gpt120_quetzal_preweight_admission` fires for any `gpt-oss-120b` quetzal
model_spec and requires a fresh (<900 s), same-allocation topology-admission receipt
(`quetzal.topology-admission-result.v1`) with a SHA-pinned qualified descriptor
(`descriptor_sha256 = f4c9fb5a…`, Ring / links=2, chip_count 4, mesh [2,2],
`device_holders_after 0`) plus SHA-pinned evidence.

- The on-node mesh descriptor **matches** the pinned `f4c9fb5a…` — the qualified topology
  config is present on this hardware. The block is the **missing producer**, not wrong hardware.
- The producer that legitimately emits the receipt (a bounded-mesh topology smoke, shipped by
  the qualified quetzal source `071e23cd`) is **not installed** on the node — only the consumer
  (`quetzal_topology_admission.py`) and its unit test are present. Installed quetzal is
  `2e7c0670` (a gemma serving branch).
- Hand-crafting a passing `QUETZAL_TOPOLOGY_ADMISSION_JSON` would be an *adapter around a
  fail-closed gate precheck* — **declined** per the honesty mandate.

## Not reached

`/v1/models` 200 · `QuetzalEngine` in logs · tool-call smoke · SWE multi-step loop · edit/submit ·
scored result. No patch. PCC **not measured**.

## Secondary findings

- The only **enrolled** gpt-oss-120b quetzal artifact in ttis `d1c15178` is **S1024** with
  prefill buckets `128,1024` (chunked prefill) → would additionally hit the known chunked-SDPA
  `attention_sink` gap even if admission passed.
- A separate **S8192/C8192 one-shot** candidate exists (`/mnt` candidates, `qualification_state:
  investigating`, experts bfp4_b) that would avoid the chunked path, but it is **not enrolled** in
  the fail-closed lane and still trips the same admission precheck. The catalog row here targets it.
- The `openai` tool-call parser + `openai_gptoss` reasoning parser **are** correctly wired in this
  ttis (the fix for the earlier bespoke malformed-tool-call failure) — present but unexercised.

## Files

- `quetzal_gptoss120b_swe_BLOCKED_76030_20260903.json` — full receipt (`ttis.local-swe-gate/v1`).
- `quetzal_gptoss120b_catalog_row_20260903.yaml` — authored `impl: quetzal` catalog row (mirrors
  the qualified Llama row) targeting the S8192 candidate. **Not** wired into `dev/llm.yaml` here.
- `quetzal_gptoss120b_serve_attempt_76030.log` — raw serve traceback.

## Update — LEGITIMATE unblock attempt (job 76034, real device receipt)

Per coordinator direction, the topology-admission gate was legitimately unblocked (no
fabricated receipt). Fetched the qualified quetzal source `071e23cd` (which ships the
producer), cloned it in-allocation, and ran the **real** `serving.topology_evidence`
bounded-mesh device smoke (mesh opened/closed/synchronized, exit 0) →
`serving.topology_admission`. This produced a genuine
`quetzal.topology-admission-result.v1` receipt (Ring/links=2, chip_count 4, mesh [2,2],
descriptor `f4c9fb5a…`, zero holders) whose pinned evidence SHAs (`smoke bf3311c6`,
`selection 1852bfcc`, `emit f296b704`) match the consumer's hardcoded constants.

With that receipt, the canonical serve advanced through **three** fail-closed gates:

1. ✅ topology admission (real device receipt)
2. ✅ quetzal source identity — installed quetzal `3a3873fe` (the candidate's actual
   compile `source_git_commit`) and set `TT_QUETZAL_COMMIT_SHA=3a3873fe` truthfully
3. ✅ bundle-manifest trusted-root proof (`QUETZAL_BUNDLE_MANIFEST_SHA256=012daa4f…`)

**New terminal blocker** — `run_vllm_api_server.py:512 _validate_quetzal_auxiliary_references`:
`RuntimeError: QUETZAL_AUXILIARY_ROOTS_JSON is invalid JSON`. The S8192 candidate's v2
bundle declares an auxiliary `streamed_cache` (`openai_gpt-oss-120b-streamed-cache`,
sha256 `2eef319a…`, 217 MoE expert tensorbins) whose digest-addressed read-only root is
**not staged** on `/mnt` (only the unrelated S1024 package's `2b2e528a` root exists).
Behind it, the candidate's `qualification_manifest` has an **empty `charter_pcc`**, so the
validator would next raise *"Quetzal TT-Metal runtime mismatch: package requires None"*.
Both are because the S8192 artifact is an unsealed **investigating** candidate — its own
manifest says "canonical TTIS endpoint, bounded SWE … remain". Sealing it (staging the
auxiliary, forging a `charter_pcc`) is a qualification/trusted-root step and was **not**
fabricated.

Net: the topology gate is genuinely cleared; the wall is now the artifact's unsealed
qualification state, not the topology contract. See
`quetzal_gptoss120b_swe_LEGIT_UNBLOCK_76034_20260904.json` and
`topology_receipt_and_serve_logs_76034.txt`.

Do **not** auto-merge. This documents a blocker; it does not claim qualification.
