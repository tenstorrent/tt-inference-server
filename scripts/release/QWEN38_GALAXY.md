# Experimental Qwen3.8-27B Galaxy image

This bundle packages the standard TTIS entrypoint and vLLM TT plugin for one
Blackhole Galaxy: eight TP4 replicas, 16 decode slots per replica, a 262,144-token
per-request context limit and 1,050,592 configured KV tokens per replica. The
128 slots do not imply that 128 simultaneous 256K conversations fit in memory.
Weights are mounted read-only from local host disk. Compilation and tensor caches
are separate, removable host-local data.

This is **not a qualified release**. The current all-BFP8 candidate completed
corrected full GPQA at **176/198 (88.89%)**, with zero output-budget truncations,
in 50m49s on October 9, 2026 UTC. The user accepted GPQA; the original strict
177/198 gate remains recorded as missed. The actual image's TTIS startup handoff
passed after fixing the probe to supply the required `--model` and `--tt-device`
arguments. This probe intercepted the final server launch and opened no devices;
container inference, current-policy agentic evaluation, SJC3 Helm and release CI
still require qualification.

The current BFP8 image manifest is
`sha256:79f7b4469a6ec2bcce5204399b37b2aeced8be7f260dd98f7007ad41f8813055`,
with model source `20619e008a236aaf393937b222a60a5b03e49cdc`. Its startup and
identity receipts are in [the release evidence directory](evidence/qwen38-20261009/).
The historical native-policy image and build example below are separate artifacts.
Use the all-BFP8 preparation instructions and its G0 receipt for the current candidate.

Historical controls: the eight-replica native-recurrence G0 check
passed on October 9, 2026 UTC. Native-control full GPQA finished **170/198
(85.86%)**, including one output-budget cutoff counted incorrect. The bounded
Tau3 pilot completed 3/12 successes. Six attempts hit task, request or step
limits, and three completed unsuccessfully. Partial review found agent and
simulator errors plus a reproduced upstream tool-state bug. All failures remain
in the denominator; the pilot is not a matched published reference setup.
The optimized recurrence's completed full GPQA result was 163/198
(82.32%); 15 output-budget cutoffs were included as incorrect. Neither that
result nor G0 alone meets the accuracy release gate. Container hardware and SJC3
Helm qualification also remain required.

The first experimental image completed compilation and source/import verification
at TTIS source `e0e05bad5361d7c170068b3ad7b4df27de192250`. Its OCI manifest is
`sha256:0b11f045bf089088a62b6e3c1aeb9b64cc72b74a632935f25203609b1e023579`.
The archive is preserved on the build host at
`/home/ttuser/qwen38-release-build-20261009-v5/artifact/qwen38-image.oci.tar`,
with full-file SHA-256
`755626b044cea12f390fd343439dbfe4af920aba6e7d156a26490c478325d648`.
It is not yet registry-published or hardware-qualified. The archive checksum
and OCI manifest digest are different identities. Do not use the archive checksum
as a Helm image digest. The separately queued BFP8/HiFi2-head experiment does
not alter this built image's native BFP4/LoFi-head policy.

## Source identity

| Component | Immutable source |
|---|---|
| Compiled Metal runtime | `a08819ddbe23077f8037d3802303939064868ff6` |
| Model subtree | `0abdc3403f039c46becef335ad02db99237593f8` |
| vLLM TT plugin | `b7e4292e4193cba20abe9c7c68ce489201b2e36b` |
| Checkpoint and tokenizer | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The model subtree pin exactly matches the frozen G0 runtime-source hash set.
Later commits on `anatarajan/qwen38-long-context-throughput-20261007` contain
reports, qualification tooling and three standalone delivery-probe files. Those
probes were not in the frozen serving source; the bundle does not relax the source
check to include them. Common model utilities and sampling come from the compiled
Metal pin. `precision_accurate_decode.json` selects native recurrence, accurate
full-tile decode attention, existing BFP4 weights, BFP8 KV and FP32 recurrent state.

## Prepare and build

Use a clean checkout at the model subtree pin and the passing
`native-g0/receipts/full-model.json` from the persistent native-control run.
The complete G0 receipt must report eight physical replicas and a passing
concurrent-versus-isolated timing comparison. Preparation rejects different
runtime files or precision. The physical chip grouping is host-specific; another
Galaxy needs its own G0 receipt before this configuration is qualified there.
Every qualified runtime file and the selected precision policy must also exist
with identical bytes in the recorded model commit. A locally passing receipt
cannot qualify an untracked or ignored policy for a SHA-pinned image build.

From a clean checkout of this TTIS branch with Python workflow dependencies:

```bash
python scripts/release/prepare_qwen38_galaxy.py \
  --model-source /path/to/pinned-metal-model-checkout \
  --qualification /path/to/native-g0/receipts/full-model.json \
  --weights-host-path /absolute/host/checkpoint-directory \
  --output qwen38-bundle
```

The output directory must be new. It contains an experimental runtime ModelSpec,
G0 receipt, source/version manifest, image verifier and Helm values. The image
recipe builds the pinned native runtime before overlaying only the pinned Qwen
subtree. It installs vLLM 0.26.0 with the plugin's CPU/empty-target installation
script and checks source hashes, precision, package versions and imports in both
image stages. These checks do not open hardware or prove model accuracy.

Pass the full TTIS checkout SHA and the manifest's exact build arguments:

```bash
docker build -f vllm-tt-metal/qwen38-galaxy.Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:6cec3f1fc25931126d0ed90dff4d7e62128c03db85bcb1313f4cb9bc40c0f8cb \
  --build-arg METAL_SHA=a08819ddbe23077f8037d3802303939064868ff6 \
  --build-arg MODEL_SHA=0abdc3403f039c46becef335ad02db99237593f8 \
  --build-arg PLUGIN_SHA=b7e4292e4193cba20abe9c7c68ce489201b2e36b \
  --build-arg TTIS_SHA="$(git rev-parse HEAD)" \
  --build-arg BUILD_JOBS=24 \
  -t qwen38-galaxy:experimental .
```

The Dockerfile preserves the measured model settings through the existing TTIS
runtime-spec path. `EXTRA_MODELS_DIR` is present before plugin discovery.
`DISABLE_METAL_OP_TIMEOUT=1` prevents the wrapper from introducing its default
five-second per-operation watchdog, which was absent from the tested host launch.
The host qualification controller still has bounded startup, evaluation and total
runtime timeouts. Kubernetes uses a two-hour startup budget, then normal health
probes. No device reset or host setup runs during the image build.

## Check an imported OCI image

On Docker's containerd image store, an imported image may use its OCI manifest
digest as `Id`; the legacy image store uses the config digest. Do not assume the
config digest is a runnable image reference. Validate the imported object against
both immutable digests recorded by the build:

```bash
python scripts/release/verify_local_oci_image.py \
  --archive /path/to/image.oci.tar \
  --manifest-digest sha256:ACTUAL_MANIFEST_DIGEST \
  --config-digest sha256:ACTUAL_CONFIG_DIGEST \
  --output /path/to/new-identity-receipt.json
```

This reads pinned metadata blobs and checks their hashes, the manifest-to-config
link, platform, ordered root filesystem layers, and runtime configuration against
Docker inspection. It neither imports nor runs the image. Layer payloads are not
rehashed here; retain the transfer archive checksum and Docker import evidence
separately. The returned `image_id` is usable for subsequent local checks. It is
not a registry publication or inference qualification. This check passed on the
imported native v5 image on Oct 9; its runtime/startup probes remain separate.

## Helm handoff

Before Helm qualification, `scripts/release/run_qwen38_container_smoke.py` can
run the immutable image after an owned hardware queue. It requires the exact
predecessor service invocation and a completed cleanup receipt, validates frozen
API-test sources and the successful image-startup receipt, then obtains the
hardware lock. It resets once, checks all eight workers, and tests streaming,
multi-turn chat, concurrency 128 and a tool round trip. Optional
`--evaluation-command /path/to/argv.json --evaluation-timeout 9000` runs a bounded
evaluation while the same container remains loaded. The command is an explicit
JSON argv list, never shell text. Use a persistent service with task-owned Docker
cleanup as recorded in the [v2 launch receipt](evidence/qwen38-20261009/image-hardware-control-v2/launch.json).
These tests are queued, not yet reported as passed.

Publish the built image to the intended registry, record its **actual digest**,
and retain its manifest, Python-package inventory and build log. No image digest
is claimed by this preparation step. The BusyBox init image is already pinned;
rendering deliberately fails until the inference-image digest is supplied.

```bash
helm template qwen38-galaxy charts/tt-inference-server \
  -f qwen38-bundle/values.yaml \
  --set-string defaults.image.digest=sha256:ACTUAL_BUILT_IMAGE_DIGEST
```

The overlay requests 32 `galaxy-blackhole` DRA boards on one node and keeps the
checkpoint mount read-only. Validate the SJC3 node, DRA visibility/chip numbering,
weights path and available memory before installation. Do not infer that an image
build or successful chart render proves that deployment passed. The normal chart
supports API authentication via `auth.apiKey`. The Qwen overlay sets startup,
readiness and liveness probes to `/health`: this route checks engine health
without a bearer token. The chart's default `/v1/models` liveness route would
return 401 with API authentication enabled and cause repeated restarts.
Regenerate older bundles with the current preparation script, or set
`defaults.probes.liveness.path=/health` when rendering their unchanged values.
This is a Helm configuration correction; it does not change the image or claim
container hardware qualification.

For release promotion, retain all 198 GPQA questions and the original score gate
alongside the explicit user acceptance of 176/198,
report output-budget cutoffs explicitly, review the bounded Tau3 outcomes, and validate
chat, multi-turn tool calls and measured throughput through the built container.
Only publish qualification claims that the final image and configuration actually
reproduce.

## Preparing an alternate precision policy

The BFP8/HiFi2-head experiment completed 166/198 and did not qualify. The full
BFP8 decoder passed eight-replica G0 and completed corrected GPQA at 176/198 on Oct 9.
Its immutable runtime is `20619e008a236aaf393937b222a60a5b03e49cdc`, on
`anatarajan/qwen38-bfp8-control-runtime-20261009`; runtime/config bytes match
the tested source. GPQA is accepted by the user; release qualification remains incomplete.
Prepare an experimental bundle using its own exact G0 receipt:

```bash
python scripts/release/prepare_qwen38_galaxy.py \
  --model-source /path/to/committed-bfp8-control-checkout \
  --qualification /path/to/bfp8-control/native-g0/receipts/full-model.json \
  --precision precision_accurate_decode_bfp8_all.json \
  --weights-host-path /absolute/host/checkpoint-directory \
  --output qwen38-bfp8-bundle
```

Use the new manifest's `MODEL_SHA` when building. The selected policy is preserved
through the runtime ModelSpec, TTIS wrapper and Helm pod environment. Rebuild and
qualify that new image; the existing native-policy image does not inherit later
precision-control results. Runtime ModelSpec, authenticated Helm probes and
startup checks are tested with native, head-only BFP8 and all-BFP8 policies.

The separate head-policy image is built at TTIS
`2485b039be071f75fa29adff6c84ecc87d60359e` and model source
`d3e8d6021f7aadcb28ba3903d79bd04b288a2819`. Its OCI manifest is
`sha256:c8ed7a5a17b4b400c84bbb82a52bd125a56b4daf5fbb150b699af62ffee169b1`;
the preserved archive checksum is
`dd3bd0e91716870d13bd0ad2d67baf35664272cf946132adec77760ccf71ec25`.
It passed source/import checks but is **not accuracy-qualified**. At 08:26 UTC,
the full GPQA run had 166 correct out of 193 completed, with five remaining;
it can no longer reach 177/198. The image remains an experimental artifact,
not a release candidate that passed the score gate.

## Check the packaged startup before opening hardware

`scripts/release/probe_qwen38_startup.py` runs the image's actual TTIS `main()`
through imports, model registration, cache/log setup and argument assembly.
It intercepts only the final call that would start the vLLM server and compares
the resulting arguments and declared environment against the bundled ModelSpec.
This is stronger than `--help`, but it does not load model weights, start HTTP,
or qualify hardware and accuracy. Run it against an already loaded immutable
image ID in a disposable container, without devices or network:

```bash
docker run --rm --network=none --read-only --memory=4g --memory-swap=4g \
  --cpus=4 --pids-limit=256 \
  --tmpfs /tmp:rw,size=512m \
  --tmpfs /home/container_app_user/cache_root:rw,uid=1000,gid=1000,size=512m \
  --mount type=bind,src=/absolute/path/probe_qwen38_startup.py,dst=/startup-probe.py,readonly \
  --mount type=bind,src=/absolute/host/checkpoint-directory,dst=/mnt/hf-cache,readonly \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /home/container_app_user/tt-metal/python_env/bin/python \
  sha256:ACTUAL_LOADED_IMAGE_CONFIG_DIGEST /startup-probe.py
```

Keep stdout and stderr with the image identity. A successful receipt is labeled
`startup_handoff_passed_unqualified`. The probe is mounted separately, so testing
it does not replace or rebuild the image being examined. Its argument-validation
tests pass locally; the actual image startup check remains queued.
