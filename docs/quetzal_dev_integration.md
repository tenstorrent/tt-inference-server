# Quetzal development model integration

The development catalog exposes generated Quetzal implementations for:

- `Qwen/Qwen3.6-27B` on `P300X2`, context 4096, concurrency 1; and
- `google/gemma-4-31B-it` on `P300X2`, context 1024, concurrency 1.

They are explicit, non-default implementations. The existing native model remains
the default when `--impl` is omitted. Select Quetzal with:

```shell
python3 run.py --model Qwen3.6-27B --tt-device p300x2 \
  --impl quetzal --workflow server --docker-server --dev-mode \
  --override-docker-image <quetzal-enabled-development-image>

python3 run.py --model gemma-4-31B-it --tt-device p300x2 \
  --impl quetzal --workflow server --docker-server --dev-mode \
  --override-docker-image <quetzal-enabled-development-image>
```

## Artifact contract and current blocker

The catalog does not reference the machine that produced the artifacts or its NAS.
Each spec names an immutable package ID formed from the attested artifact-tree and
weights-tree SHA-256 digests. Every generated-code, metadata, and weights path is
beneath:

```text
/home/container_app_user/cache_root/quetzal/packages/<package-id>/
```

`cache_root` is TTIS's existing persistent Docker volume or `--host-volume` bind
mount. The package must have this relative layout:

```text
compiled/<artifact-name>/full/prefill/generated.py
compiled/<artifact-name>/full/prefill/metadata.json
compiled/<artifact-name>/full/decode/generated.py
compiled/<artifact-name>/full/decode/metadata.json
compiled_weights/<weights-name>/full/weights.pt
qualification_manifest.yaml
```

The qualification manifest is required even when explicit graph paths are set:
Quetzal discovery uses it to admit the exact declared reduced-precision
transformations, including Gemma's BFP8 attention and MLP weights.

TTIS does **not** currently fetch, verify, or materialize a Quetzal package into
that content store. Its release image also does not currently install the Quetzal
vLLM entry-point package and serving modules. Consequently, catalog resolution is
host-testable, but an actual server launch must fail until a development image and
installer provide both dependencies. Do not replace these paths with a workstation
or NAS path.

The required follow-up is a signed-package installer that verifies the root
manifest and all file hashes before atomically publishing the package directory,
plus a pinned TTIS development image that installs the Quetzal plugin non-editably.
Generated-provider registration is fail-closed; missing plugin or artifacts must
never select `tt_transformers`.

## Nightly selection

`.github/workflows/models-ci-config.json` can now represent an explicit `impl`
selector on each implementation row, and release promotion preserves that selector
when matching a development catalog template. This removes the previous ambiguity
between native and Quetzal implementations that both use vLLM.

Do not alter the native nightly entry. Add Quetzal as a second P300X2 implementation
row only after the CI matrix consumer passes its `impl` value to `run.py`, and after
the pinned development image and immutable package are available to that job. Until
then, nightly enrollment would only create an expected infrastructure failure.
