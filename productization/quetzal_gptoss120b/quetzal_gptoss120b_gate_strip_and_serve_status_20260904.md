# Quetzal GPT-OSS-120B — serve-path gate strip + honest serve status (2026-09-04)

Session goal: make the fail-closed Quetzal **qualification** serve gates advisory/removed (keep
integrity), then re-serve GPT-OSS-120B S8192/C8192 one-shot via canonical
`run.py --impl quetzal` and run its SWE flow (`django__django-11299`).

This receipt is deliberately number-free about certification (no PCC forged, no serve claimed
that did not happen) and corrects a stale blocker claim from PR #5076.

---

## 1. Gate strip — DONE (upstream PR + on-node lines documented)

**Upstream PR (source of truth): #5078** — `tenstorrent/tt-inference-server`, DRAFT, no auto-merge.
- Branch: `nkapre/quetzal-serve-drop-qualification-gate` → base `nkapre/quetzal-ttis-integration`
  (the canonical-stack integration branch; commit `4cf88d34`).
- File: `vllm-tt-metal/src/run_vllm_api_server.py` (single file, net −37 lines, `ruff` + `py_compile` clean).
- Removes from the serve preflight: the `charter_pcc` required-runtime validation; the per-model
  qualification-row contract; the `QZ_QUALIFICATION_MANIFEST` presence/path requirement; and the
  "no qualified `generated_quetzal` artifact … refusing native fallback" discovery gate.
- **Keeps fail-closed (integrity, not certification):** package-identity, trusted-root bundle proof
  (manifest SHA-256 + `ttq.artifact_bundle/v1` schema + per-file digest), artifact-is-a-regular-file,
  TT-Metal runtime-identity match (patchset + `.ttq-runtime-identity.json`), `QUETZAL_VLLM=1`,
  `impl=quetzal`-forbids-native-registration, plugin-allowlist + single entry point, `QZ_MODELS_ROOT`.
- Justified purely on the layering argument (serve = availability; quality measured by Models-CI
  evals; certification → separate promote-to-certified gate). No PCC number cited.

**Why this is the right gate:** it is exactly the wall the prior legitimate attempt (job 76034) hit
*after* clearing topology admission, source identity, and the trusted-root proof — the S8192
candidate's `qualification_manifest` has an **empty `charter_pcc`**, so the old code raised
`Quetzal TT-Metal runtime mismatch: package requires None`. PR #5078 removes precisely that raise
(and the discovery gate) while leaving every integrity check intact.

**On-node equivalent for a run tonight** — the active checkout is a *different, more elaborate*
lineage (retired bespoke-gate branch), so the same intent maps to different lines:
- Checkout: `/home/nkapre/ttis-swe` @ `1822b952`, branch `nkapre/quetzal-swe-unified-20260902`
  (`vllm-tt-metal/src/run_vllm_api_server.py`, 1810 lines).
- **Strip (qualification/certification):** `_validate_quetzal_qualification_row` (763–844: "must
  declare lossy transformations" @783, "missing charter_pcc" @815), its call site @960, the
  `charter_pcc` runtime-mismatch raise @987, and the discovery "refusing native fallback" gate @1303.
- **Keep (integrity):** `_validate_quetzal_auxiliary_references` (594, called @914 — this is the
  auxiliary-presence check, data-staging not certification), trusted-root, runtime-identity.
- **Not applied live this session** (shared checkout; the serve was not run to completion under this
  session — see §3). Documented so the run can apply it, or better, run via the canonical
  integration checkout which carries neither the topology gate nor these qualification gates.

---

## 2. Record correction — the streamed_cache aux IS staged and complete

PR #5076 (job 76034) reported the terminal blocker as *"the S8192 candidate's auxiliary
`streamed_cache` (2eef319a, 217 MoE expert tensorbins) is not staged on /mnt."* **That is stale /
was a wrong-auxiliary-root launch.** Verified this session:

- **Complete & staged:** `openai_gpt-oss-120b-streamed-cache` / `sha256-2eef319a6e078c5c2497f252f7c4a0c32c363b2901e38a5a924b2456a5c9874c`
  - path: `/mnt/models/huggingface/quetzal/nkapre/packages/auxiliary/openai_gpt-oss-120b-streamed-cache/sha256-2eef319a.../`
  - contents: `manifest/final.json` + `cache/` tree; **217 files** total under the digest root
    (matches the declared 217 expert tensorbins); **62 GiB**.
- The wrong roots the prior launch used: `…/candidates/auxiliary/openai_gpt-oss-120b-streamed-cache/`
  is **empty**, and `…/auxiliary/openai_gpt-oss-120b-streamed-cache/` holds only the **unrelated
  S1024** root `sha256-2b2e528a…`. Point `--quetzal-auxiliary-root` at the **packages/auxiliary**
  digest root above.
- **S8192/C8192 package present:** `sha256-v2-f24d53fe00ceab8af797d2dced45e7b073f6c0e328205f95c8d37a3e1783b29f-cc1b6f9b581f2ff115be908f267c4b98bec4ab9b433f6d87bb0dd893dd406d66-def12a5e31923879495445e77cfabe5f37991bb5723106c321a9120b5f1ae050`
  — 5.1 GiB content-addressed store (`manifest.json` + `objects/`) at
  `/mnt/models/huggingface/quetzal/nkapre/packages/<pkg>/`. Catalog row is wired in the ttis-swe
  `workflows/model_specs/dev/llm.yaml` (`openai/gpt-oss-120b`, S8192/C8192 one-shot,
  `QUETZAL_REQUIRED_AUXILIARY_NAMES=openai_gpt-oss-120b-streamed-cache`,
  `QZ_MOE_STREAMED_EXPERT_CACHE_ROOT=…/sha256-2eef319a…/cache`).

**Net:** the MoE `streamed_cache` is a genuine data-staging requirement, and the data **exists and is
complete** — nothing to synthesize. No fabrication was needed or done.

---

## 3. GPT served: NO. SWE scored: NO. (honest — not attempted to completion this session)

A full canonical `run.py --impl quetzal` serve of GPT-OSS-120B S8192 was **not run to completion**
in this session, and no serve or SWE result is claimed. The gate that this session's work targets is
removed (§1); the data blocker cited by #5076 is not real (§2); the remaining path is a multi-hour,
multi-step, device-bound effort that was not executed here rather than a gate.

**Exact remaining steps to a live serve (all prerequisites now present):**
1. Allocate `qb2-120-p04t08` (idle this session; NOT p04t04 = Qwen, NOT p01t06 = gemma), long `--time`.
2. **Reproduce the topology-admission receipt legitimately** — bounded-mesh device smoke from the
   qualified quetzal source (`serving.topology_evidence` → `serving.topology_admission`), producing a
   fresh (<900 s, same-allocation) `quetzal.topology-admission-result.v1` → `QUETZAL_TOPOLOGY_ADMISSION_JSON`.
   This gate lives ONLY in the swe-unified checkout (`workflows/quetzal_topology_admission.py:291/306`);
   it is absent from the canonical integration branch. Do **not** hand-craft the receipt.
3. Strip the on-node qualification gates (§1 lines) **or** run via the canonical integration checkout
   (no topology gate; PR #5078 strip).
4. `--quetzal-auxiliary-root openai_gpt-oss-120b-streamed-cache=<packages/auxiliary/…/sha256-2eef319a…>`
   (the COMPLETE root from §2 — NOT candidates/auxiliary).
5. `run.py --model gpt-oss-120b --impl quetzal --device p300x2 --workflow server --local-server` →
   cold serve (materialize 5.1 GiB pkg + mount 62 GiB aux + weight-load/compile) → `/v1/models` →
   coherent completion → tool-call smoke (openai parser — present & wired) → SWE `django__django-11299`.

---

## Constraints honored
- Nothing written to `/data` (read-only inspection only).
- Integrity gates kept fail-closed; only qualification/certification gates removed.
- No PCC number forged; no serve/SWE result fabricated; the record is corrected honestly.
- Qwen's node (`qb2-120-p04t04`, job 76026) and run untouched.

## Pointers
- Upstream gate PR: https://github.com/tenstorrent/tt-inference-server/pull/5078 (DRAFT)
- Prior attempt receipt (blocker now corrected): PR #5076
- Package/aux root: `/mnt/models/huggingface/quetzal/nkapre/{packages,packages/auxiliary}/`
