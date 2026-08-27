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

TTIS does **not** fetch, verify, or materialize a Quetzal model package into that
content store. Do not replace these paths with a workstation or NAS path. Install
the content-addressed bundle into the persistent cache before server startup with
Quetzal's `ttq-artifact-bundle install`, supplying the root-manifest SHA-256 from
the signed release record. The installer verifies every object, stages privately,
and publishes the compiled tree last.

The base release image still does not contain Quetzal. Build a uniquely identified
development derivative from a digest-pinned base and an exact Quetzal commit:

```shell
scripts/build_quetzal_dev_image.sh \
  --base-image ghcr.io/tenstorrent/tt-inference-server/<image>@sha256:<digest> \
  --quetzal-source /path/to/tt-quetzalcoatlus \
  --quetzal-commit <full-40-character-commit> \
  --tag <registry>/ttis-quetzal:<tt-metal>-<vllm>-<quetzal>
```

`vllm.tt-metal.src.quetzal.Dockerfile` installs a regular wheel with no editable
source path, checks the resulting environment, and verifies the exact
`vllm.general_plugins` entry point during the build. It deliberately derives
from, rather than conditionally changing, the
standard image so the ordinary tt-metal/vLLM image identity stays unambiguous.
The wrapper refuses a dirty checkout or a HEAD different from the requested
commit, exports that commit with `git archive`, and supplies the export as a
named BuildKit context. No `.git` directory, token, credential helper, SSH agent,
or network repository URL is forwarded into the build.
The v0.20.0 base currently carries two pre-existing `uv pip check` conflicts.
The derivative records the exact conflict set before installation and requires
the post-install set to be byte-identical, so Quetzal cannot add or mask a
dependency conflict while that independently owned base-image debt remains.
Generated-provider registration remains fail-closed: a missing plugin or model
package must never select `tt_transformers`.

This closes the image-construction path, not release qualification. The two real
model bundles, trusted manifest digests, container build, clean-QB2 launch, nightly,
and CS-owned acceptance rows are still required.

## Nightly selection

`.github/workflows/models-ci-config.json` can now represent an explicit `impl`
selector on each implementation row, and release promotion preserves that selector
when matching a development catalog template. This removes the previous ambiguity
between native and Quetzal implementations that both use vLLM.

Do not alter the native nightly entry. Add Quetzal as a second P300X2 implementation
row only after the CI matrix consumer passes its `impl` value to `run.py`, and after
the pinned development image and immutable package are available to that job. Until
then, nightly enrollment would only create an expected infrastructure failure.
