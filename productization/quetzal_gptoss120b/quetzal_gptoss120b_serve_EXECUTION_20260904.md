# Quetzal GPT-OSS-120B — canonical serve EXECUTION attempt (2026-09-04, p04t08 job 76238)

Follow-on to the gate-strip receipt: this bank records the actual multi-hour device-serve
EXECUTION attempt of the S8192/C8192 one-shot candidate (no S16384 one-shot package is staged;
S8192 is the largest one-shot <=16K that avoids the chunked attention_sink gap).

## What was verified PRESENT and WORKING (real, on-device on p04t08)
- Node p04t08 (job 76238, 12h): `/mnt/models` NFS-mounted (10.32.13.1:/models) -> package + aux visible.
- Runtime `/var/tmp/nkapre/tt-metal` python_env: vllm 0.26.0+empty, transformers 5.15.0, numpy 1.26.4,
  entry point `quetzal_model_registry -> tt_quetzalcoatlus.vllm_plugin:register`, tt platform plugin
  activates, and `serving.quetzal_server` / `serving.artifact_bundle` import OK.
- Data (unchanged, complete): package `sha256-v2-f24d53fe...` 5.1 GiB CAS (manifest.json + objects);
  aux `sha256-2eef319a...` 217 files / 62 GiB / manifest/final.json at packages/auxiliary/.
- Topology producer source available: `sources/quetzal-071e23cd.bundle`.
- ttis d1c15178 checkout staged at `gpt-serve-work/ttis`; **gate strip applied** (charter_pcc
  validation, QZ_QUALIFICATION_MANIFEST requirement, and discovery "refusing native fallback" gate
  removed; `_validate_quetzal_auxiliary_references`, trusted-root, and runtime-identity kept;
  `py_compile` clean). Mirrors PR #5078. Patched file at
  `gpt-serve-work/ttis/vllm-tt-metal/src/run_vllm_api_server.py`.

## Real device execution attempted (topology-receipt reproduction)
Ran the genuine `serving.topology_evidence` device smoke (071e23cd producer) on p04t08 against the
Ring/2ch descriptor. Topology discovery succeeded (4 Blackhole chips 0-3, firmware 19.8.1, IOMMU
enabled), but `ttnn.open_mesh_device(MeshShape(2,2))` HANGS on
`Waiting for lock 'CHIP_IN_USE_0_PCIe' held by PID 959761`. That PID is an **active, legitimate
foreign tenant**: `ttuser`/vkovacevic running a live `vllm serve /data/jserbedzija/models/
gemma-4-31B-it-842da379 --served-model-name google/gemma-4-31B-it --port 8000` (launched by another
agent via `/localdev/vkovacevic/gemma4-ttft/serve.sh`), holding all 4 chips. slurm reported p04t08
"idle" and gave my hold job 76238 the allocation, but the node is PHYSICALLY occupied by this
untracked live serve. Not a leaked orphan — a real tenant's work; must not disturb (fleet
citizenship), and no sudo anyway. So the device on p04t08 is unavailable and the topology receipt
could not be produced there. My hold (76238) was released rather than squat a busy node.

## Terminal blocker — NOT a gate, NOT data, NOT forgeable: runtime patchset attestation
The catalog row requires `QUETZAL_REQUIRED_TT_METAL_COMMIT=b534549300fe...` **and**
`QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256=22fb0bd2...` (`QUETZAL_TT_METAL_PATCHSET_STATUS=applied`).
The only runtime co-located with the NFS data (p04t08 `/var/tmp/nkapre/tt-metal`) is:
- git HEAD exactly `b534549300fe...` (correct base commit),
- working tree CLEAN (no patchset applied on top),
- `.ttq-runtime-identity.json` = `{"base_revision":"b534549300fe...","patchset_sha256":null,"manifest_sha256":null}`.

So the required patchset `22fb0bd2` is genuinely ABSENT/unattested on this build. The KEPT runtime-
identity/patchset integrity gate (which the strip deliberately preserves) therefore legitimately
fails-closed: `required_patchset(22fb0bd2) != actual_patchset(null)`. This is real integrity, not
certification bureaucracy — it prevents serving on an unattested runtime ABI.

The properly patched + attested runtime is the 26.8 GiB podman image
`localhost/ttis-quetzal-gpt:071e23cd-ttisd1c15178-local-shadow`, which exists ONLY in the node-local
podman store on **p06t07** — and p06t07 has **no `/mnt/models` NFS mount** (confirmed: `/mnt/models`
is an empty local dir there), so it cannot see the 5.1 GiB package or 62 GiB aux. p06t07's missing
NFS mount is almost certainly why the original task said "NOT p06t07."

Net topology bind: attested-runtime (image) and data (NFS) live on different nodes and cannot be
co-located without a heavy 26.8 GiB cross-node image relocation (and p06t07 is the constrained node).
The package is also a CAS store that the serve orchestration must materialize before the artifact
files exist as regular paths.

## What was deliberately NOT done (honesty mandate)
- Did NOT forge the runtime patchset attestation to make the local build pass the kept integrity gate.
- Did NOT serve on an admittedly-non-matching (unpatched) runtime.
- Did NOT synthesize any data or SWE result.

## Result
- Gate relaxed: YES (PR #5078 + on-node d1c15178 strip applied, compiles).
- GPT served (S8192/16K one-shot): NO.
- SWE (django__django-11299) scored: NO.
- Exact stop point: runtime **patchset-attestation** integrity gate — local build is base-commit-correct
  but patchset `22fb0bd2` absent (identity null); attested image stranded on p06t07 (no NFS). Requires
  either a 26.8 GiB cross-node image relocation to a NFS-mounted node (then docker-server + CAS
  materialize + reproduced topology receipt + cold serve + SWE), or a legitimately patched+attested
  local tt-metal build co-located with the data. Neither forgeable.

## Constraints honored
- Nothing written to /data. Work under `/mnt/models/huggingface/quetzal/nkapre/gpt-serve-work` (NFS) +
  `~/qz_launch_contracts` + repo productization/. Integrity gates kept. No fabrication. Qwen node
  (p04t04 job 76026) and Gemma node (p01t06 job 75983) untouched.
