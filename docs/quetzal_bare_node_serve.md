# Serving a Quetzal-compiled model with `run.py --impl quetzal`

This is the minimal serve path for a Quetzal-generated (compiled) model on a bare
Exabox QB2 (`P300X2`, four Blackhole chips driven as a `P150x4` mesh). It is the
carved-out serve subset of the larger `#5042` bundle; the release-image contract,
Models CI enrollment, and release automation are deliberately **out of scope**
here (see the PR description).

## What must already exist

1. **Quetzal wheel in the vLLM image.** The image must have the
   `tt-quetzalcoatlus` wheel installed (it publishes the `quetzal_model_registry`
   vLLM plugin entry point and the top-level `serving` package). The pinned
   Quetzal commit must include the serve-required KV-cache-key patch —
   `serving/e2e_pcc.py:discover_cache_keys` grouping `cache_layers_<N>_keys`
   correctly (Quetzal commit `2f1490a4`, the source pin recorded in the row as
   `QUETZAL_REQUIRED_SOURCE_REVISION` / `TT_QUETZAL_COMMIT_SHA`). This patch
   **lives in the Quetzal package and is referenced, never vendored into ttis.**
2. **A generated package.** A content-addressed Quetzal artifact bundle
   (`sha256-<tree>-<manifest>`) with `compiled/…/{prefill,decode}/…`,
   `compiled_weights/…/weights.pt`, and `qualification_manifest.yaml`.
3. **The catalog row.** The `impl: quetzal` row for the model in
   `workflows/model_specs/dev/llm.yaml`, generated from the Quetzal catalog
   fragment — do not hand-edit the package paths:

   ```bash
   python3 scripts/generate_quetzal_model_row.py \
       --catalog <tt-quetzalcoatlus>/serving/catalog/dev_llm_quetzal.yaml \
       --model meta-llama/Llama-3.2-1B-Instruct \
       --package-id sha256-<tree>-<manifest> \
       --manifest-sha256 <hex> \
       >> workflows/model_specs/dev/llm.yaml
   ```

## Package admission (the bare-node write-bit problem)

The Quetzal plugin's default admission requires a **read-only mount point**
(it rejects any directory with write bits set). That is satisfied in a container
by a `:ro` bind mount, but a bundle staged by a bare `srun` sits on a **writable**
filesystem and fails closed with *"mutable (write bits set)"*. Historically this
was worked around ad-hoc with `fuse-overlayfs -o ro`.

The serve path now admits the package **by content hash** instead: for
`impl=quetzal`, `run_vllm_api_server.py::admit_quetzal_bundle` authenticates the
package against its trusted-root manifest by SHA-256 (pinned to
`QUETZAL_BUNDLE_MANIFEST_SHA256`) before any device use. It is **independent of
file write bits** and works on a writable FS, and it fails closed: a missing
`QUETZAL_PACKAGE_ROOT`, a missing wheel, a hash mismatch, or a v2 bundle whose
`QUETZAL_AUXILIARY_ROOTS_JSON` is missing/mismatched aborts before the device is
opened. This is not an adapter around the plugin's gate — it selects the
content-addressed admission path the Quetzal package already implements.

Two package layouts are admitted, both through `serving.artifact_bundle`:

* **Portable object-store bundle** (`<root>/manifest.json` + `objects/`): handed
  to `verify_bundle(root, expected_sha256=…, auxiliary_roots=…)`, which reads and
  hashes every payload; `auxiliary_roots` authenticates any v2 externals. This is
  the portable Llama-1B (v1) case.
* **Installed / shared bundle** (the read-only shared layout used by gpt-oss and
  the MoE models): there is no root `manifest.json` — the trusted-root proof is
  `<root>/.quetzal-bundle-manifests/<QUETZAL_BUNDLE_MANIFEST_SHA256>.json` and the
  payloads sit at their logical paths. `admit_quetzal_bundle` authenticates that
  proof by its pinned SHA-256, then verifies every serve-loaded artifact (the
  `QUETZAL_*`/`QZ_*` members) and any v2 `streamed_cache` auxiliary against the
  trusted inventory, reusing `serving.artifact_bundle`'s own manifest validator
  and hashing primitives.

For a **v2** bundle (external MoE `streamed_cache`), the catalog row must set
`QUETZAL_AUXILIARY_ROOTS_JSON` to the JSON map of auxiliary name → immutable,
digest-addressed root (`{"…-streamed-cache": "…/sha256-<tree-digest>"}`); the
admission hashes those external payloads too and fails closed if the digest,
size, or content-addressed basename does not match the trusted proof.

### Fallback: read-only mount (if content-address admission is unavailable)

If you must fall back to the mount-based gate, stage the bundle read-only with the
supported helper (not an ad-hoc one-liner):

```bash
scripts/quetzal_stage_bundle.sh <src-package-dir> <ro-mount-target>
```

Point `QUETZAL_PACKAGE_ROOT` / `QZ_MODELS_ROOT` at `<ro-mount-target>`.

## Offline weights / tokenizer

The row sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; model weights come
from the package (`QUETZAL_WEIGHTS`), and the tokenizer/config resolve from the
HF cache (`HF_HOME`). `ensure_weights_available()` honors this: when offline it
resolves the checkpoint from the populated HF cache
(`snapshot_download(..., local_files_only=True)` with no `local_dir`) instead of
re-planning a download into a fresh `local_dir`, which fails closed with no
network even when every file is already cached.

## Run

```bash
python run.py \
    --model Llama-3.2-1B-Instruct \
    --device P300X2 \
    --impl quetzal \
    --workflow serving \
    --docker-server --dev-mode
```

`--impl quetzal` is mandatory: the row is `default_impl: false`, so omitting it
keeps upstream's native tt-metal implementation. Tool calling
(`--enable-auto-tool-choice --tool-call-parser llama3_json`) is wired from the
row so SWE/agentic scores are not silently voided.
