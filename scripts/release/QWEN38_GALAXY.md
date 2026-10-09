# Experimental Qwen3.8-27B Galaxy image

This bundle packages the standard TTIS entrypoint and vLLM TT plugin for one
Blackhole Galaxy: eight TP4 replicas, 16 decode slots per replica, a 262,144-token
per-request context limit and 1,050,592 configured KV tokens per replica. The
128 slots do not imply that 128 simultaneous 256K conversations fit in memory.
Weights are mounted read-only from local host disk. Compilation and tensor caches
are separate, removable host-local data.

This is **not a qualified release**. The eight-replica native-recurrence G0 check
passed on October 9, 2026 UTC. Native-control GPQA and the Tau3 pilot remain in
progress. The optimized recurrence's completed full GPQA result was 163/198
(82.32%); 15 output-budget cutoffs were included as incorrect. Neither that
result nor G0 alone meets the accuracy release gate. Container hardware and SJC3
Helm qualification also remain required.

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

## Helm handoff

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
supports API authentication via `auth.apiKey`.

For release promotion, retain all 198 GPQA questions and the existing score gate,
report output-budget cutoffs explicitly, finish the valid Tau3 pilot, and validate
chat, multi-turn tool calls and measured throughput through the built container.
Only publish qualification claims that the final image and configuration actually
reproduce.
