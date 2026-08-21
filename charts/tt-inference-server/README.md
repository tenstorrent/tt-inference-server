# tt-inference-server Helm Chart

Deploys vLLM, media, and forge inference backends on Tenstorrent hardware.

**Chart version:** 0.2.0 | **App version:** 0.12.0

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Device Acquisition (DRA)](#device-acquisition-dra)
- [Quick Start](#quick-start)
- [Chart Structure](#chart-structure)
- [Configuration System](#configuration-system)
- [Values Reference](#values-reference)
- [Defaults Reference](#defaults-reference)
- [Supported Models](#supported-models)
- [Advanced Usage](#advanced-usage)

---

## Overview

This chart creates a single `Deployment`, `Service`, `ConfigMap`, and `Secret` for one inference model on one Tenstorrent device. The chart ships with pre-validated configurations for every supported model and device combination. You select a model and device at install time; the chart merges your selection against a layered config system to produce the final Kubernetes resources.

Three engines are supported:

- **vllm** — large language models served via the vLLM OpenAI-compatible API
- **media** — image/audio/video/embedding models served via the tt-media-inference-server API
- **forge** — models served via the tt-forge backend

Devices are acquired through **Dynamic Resource Allocation (DRA)**: each Pod
declares a `ResourceClaim` that the [tt-operator](https://docs.tenstorrent.com/tt-operator/latest/)
DRA driver satisfies by allocating Tenstorrent boards and injecting the matching
`/dev/tenstorrent/<N>` node(s). The container runs **unprivileged**, with **no
`hostPath` on `/dev/tenstorrent`**, and multiple Pods can share one Node.

---

## Architecture

tt-operator (the **supply side**) and this chart (the **consumer**) have
separate lifecycles, connected only through DRA:

- **tt-operator** is an umbrella chart of cluster-singleton components (the DRA
  driver, fabric manager, node-feature-discovery, ...). There is no standalone
  `tt-operator` image. A cluster administrator installs it **once per cluster**.
- **This chart** (the consumer) requests devices through a `ResourceClaim` and
  never installs, bundles, or depends on tt-operator as a sub-chart — cluster
  singletons and per-model workloads have different lifecycle granularity, and
  bundling would make each model release co-own the cluster-wide device
  foundation (a `helm uninstall` of one model would tear it down for the others).

---

## Requirements

- **Kubernetes ≥ 1.34** — the chart emits `resource.k8s.io/v1`, which is served
  from 1.34 (DRA GA). Verify the API is present:
  `kubectl api-resources --api-group=resource.k8s.io`.
- **tt-operator installed on the cluster**, providing at least the DRA driver
  ([`tt-dra-driver`](https://docs.tenstorrent.com/tt-dra-driver/latest/)), the
  fabric manager ([`tt-fabric-manager`](https://docs.tenstorrent.com/tt-fabric-manager/latest/),
  which the DRA driver queries to enumerate devices), and
  [node-feature-discovery](https://docs.tenstorrent.com/tt-operator/latest/components/node-feature-discovery.html). See
  [Architecture](#architecture) for how its lifecycle relates to this chart.

The tt-operator chart and its component images are **public on GHCR** — the
`helm install` below needs no authentication on a standard cluster.

Install tt-operator (v0.2 doesn't use its multi-node scheduling, so `jobset` and
`kubepmix` are turned off):

```bash
helm install tt-operator oci://ghcr.io/tenstorrent/helm/tt-operator \
  --version 0.2.0 -n tt-operator-system --create-namespace \
  --set jobset.enabled=false \
  --set kubepmix.enabled=false
# Verify: a DeviceClass and a per-node ResourceSlice appear
kubectl get deviceclass          # tenstorrent.com
kubectl get resourceslices       # <node> / tenstorrent.com, one device per board
```

> The remaining sub-charts — `node-feature-discovery`, `tt-k8s-driver-manager`,
> `tt-fabric-manager`, `tt-dra-driver`, `tt-telemetry` — stay at their defaults.
> See the [tt-operator docs](https://docs.tenstorrent.com/tt-operator/latest/) for
> what each sub-chart does and how to configure it.

---

## Device Acquisition (DRA)

The chart renders a `ResourceClaimTemplate`; every Pod gets its own claim from
it. The number of boards requested is a function of `device` (not the model): it
comes from the top-level `deviceBoardCounts` map in `values.yaml`, which the
generator produces from `workflows/device_utils.py:BOARD_TYPE_COUNT_TO_DEVICE`
(a single source of truth, not a per-model value):

| device | boards requested (`count`) |
|---|---|
| `n150`, `n300`, `p100`, `p150`, `p300` | 1 |
| `p300x2` | 2 |
| `n150x4`, `p150x4`, `t3k` | 4 |
| `p150x8` | 8 |
| `galaxy` | 32 |

**Multiple Pods per Node.** The scheduler places a Pod only once its claim can be
allocated and never assigns the same board twice, so a T3K node can run e.g. four
independent single-`n300` Pods. When all boards are claimed, an extra Pod stays
`Pending` (`FailedScheduling: cannot allocate all claims`) until one frees.

**Not yet supported (future work).** A board count is correct only for a single DRA
device (`n300` = one board = an adjacent on-board chip pair) or a whole dedicated node
(`t3k`). Adjacency-partitioned requests — several separate devices that must be
physically adjacent (QSFP-linked), e.g. `galaxy_t3k` or a Galaxy dual — can't be
expressed (DRA allocates by count, not position), so they are absent from
`deviceBoardCounts` and fail at render.

**Teardown order.** Delete the inference workload *before* uninstalling
tt-operator; removing the DRA driver first leaves Pods unable to release their
claims and stuck `Terminating`.

---

## Quick Start

> **Prerequisite:** tt-operator must already be installed on the cluster (see [Requirements](#requirements)). Without its DRA driver the Pod can't acquire a device and stays `Pending`.

Three values are required at install time:

```bash
helm install my-model ./charts/tt-inference-server \
  --set model="Llama-3.1-8B-Instruct" \
  --set device=galaxy \
  --set hfToken="hf_xxx"
```

The chart will fail at render time if `model` or `device` is missing, or if the combination is not present in `values.yaml`.

---

## Chart Structure

```
charts/tt-inference-server/
├── Chart.yaml               # Chart metadata, name, and versions
├── values.yaml              # All configuration: required values, defaults, per-model overrides
└── templates/
    ├── _helpers.tpl         # Config resolution logic and name-generation helpers
    ├── deployment.yaml      # Deployment with init containers and volume mounts
    ├── configmap.yaml       # Non-secret environment config (model, device, cache paths)
    ├── secret.yaml          # HF_TOKEN secret
    └── service.yaml         # ClusterIP Service exposing the inference API
```

### templates/_helpers.tpl

The chart's named templates (all prefixed `tt-inference-server.`), grouped by role:

**Config resolution** — resolve `model` + `device` into an effective config:

| Helper | Purpose |
|---|---|
| `validateValues` | Fails the render if `model`/`device` is missing, or the resolved combination has no entry in `models`. |
| `resolvedEngine` | Picks the engine: `.Values.engine`, else the sole engine offering the device, else `models.<model>.defaultEngine`. |
| `resolvedImpl` | Picks the impl: `.Values.impl`, else `models.<model>.<engine>.<device>.defaultImpl`. |
| `resolvedConfig` | Deep-merges `defaults` with the resolved impl block into the effective config (see [Configuration System](#configuration-system)). |

**Derived values** — build fields from the resolved config:

| Helper | Purpose |
|---|---|
| `image` | Container image string `repository:tag`. |
| `containerEnv` | Container env, merged from spec env, hf-cache env, and `extraEnv` `valueFrom` entries. |
| `cacheHostPath` | `cache.hostPath` if set, else `/opt/cache/<model>-<device>-<impl>`. |
| `draDeviceCount` | DRA board count for `device` from `deviceBoardCounts` (fails for unsupported shapes). |
| `draBoardName` | DRA `boardName` the `ResourceClaim` selects, from `deviceBoardNames`. |

**Naming** — `name`, `fullname`, `chart`, `labels`, `selectorLabels`, `configmapName`, `secretName`: standard Helm name/label helpers.

### Init Containers

Run before the inference server starts:

**`fix-cache-permissions`** (always)
Runs `chown -R 1000:1000 /cache` on the cache host path volume. The inference server runs as UID 1000 and requires write access to the cache directory, which may be created by root on the host.

**`cleanup-hugepages`** (only when `hugepages.enabled`)
Removes stale hugepage files left by previous runs, **scoped to this Pod's
DRA-allocated board(s)** only. The init container requests the same
`ResourceClaim` as the main container, so the DRA driver injects its
`/dev/tenstorrent/<N>` node(s); the container then deletes only
`/dev/hugepages-1G/device_<N>_*tenstorrent` (and the bare `tenstorrent` file for
board 0) for each allocated `N`. This is required now that multiple Pods can run
on one Node — a global `rm` would delete a co-located running Pod's hugepage
files. Without the cleanup, a replacement Pod reusing the same board may fail to
acquire hugepages if a previous Pod exited uncleanly.

---

## Configuration System

A model is addressed by a `model → engine → device → impl` path. `engine` and `impl` are normally inferred, so most installs set only `model` and `device`:

- **engine** — if exactly one engine offers the requested device it is chosen automatically; when more than one does, `models.<model>.defaultEngine` is used unless you pass `--set engine=`.
- **impl** — defaults to `models.<model>.<engine>.<device>.defaultImpl` unless you pass `--set impl=`.

The effective config is then a two-level merge — the resolved impl block layered on top of `defaults`, where the impl block wins on any conflict:

```
defaults                                              ← baseline for every model
  └── models.<model>.<engine>.<device>.impls.<impl>   ← resolved leaf, wins on conflict
```

**Example:** for `model=Llama-3.1-8B-Instruct` and `device=n300`:

1. Resolve the engine — `n300` is offered by both `vllm` and `forge`, so `models.Llama-3.1-8B-Instruct.defaultEngine` (`vllm`) is used. Pass `--set engine=forge` to pick the other.
2. Resolve the impl — `models.Llama-3.1-8B-Instruct.vllm.n300.defaultImpl` (`tt_transformers`).
3. Start from the full `defaults` block, then deep-merge `models.Llama-3.1-8B-Instruct.vllm.n300.impls.tt_transformers` on top — overriding `image.repository`, `image.tag`, `resources`, `progressDeadlineSeconds`, and `env`.

Any field the resolved impl block does not set falls back to `defaults`.

---

## Values Reference

Set per release, typically via `--set`.

| Value | Required | Default | Description |
|---|---|---|---|
| `model` | yes | `""` | Model name. Must match a key under `models`. |
| `device` | yes | `""` | Device name. Must match a key under `models.<model>.<engine>`. |
| `engine` | no | `""` | Pin the engine (`vllm`, `media`, or `forge`) when a model+device is offered by more than one. Defaults to the only matching engine, or `models.<model>.defaultEngine` when ambiguous. |
| `impl` | no | `""` | Pin the implementation when a device offers more than one. Defaults to `models.<model>.<engine>.<device>.defaultImpl`. |
| `hfToken` | yes* | `""` | HuggingFace token. Injected as `HF_TOKEN`. Required unless weights are pre-downloaded via `hfCacheDir`. |
| `hfCacheDir` | no | `""` | Host path to a pre-downloaded HuggingFace weights directory. Mounted read-only at `/mnt/hf-cache`; skips download at startup. |
| `auth.apiKey` | yes* | `""` | Bearer key clients must send. Stored in the release Secret as `API_KEY` (media/forge) or `VLLM_API_KEY` (vllm). *Required for `media` and `forge` engines unless `auth.disabled=true`: those servers guard their inference routes with a literal bearer key and fall back to a well-known built-in default when unset. |
| `auth.disabled` | no | `false` | Run the server with authentication off (`NO_AUTH=1`). media/forge only — vLLM is already open when `auth.apiKey` is unset. |
| `hugepages.enabled` | no | `true` | Whether Tenstorrent boards need 1Gi hugepages. Set `false` on IOMMU + KMD 1.29.0+ clusters to drop the `hugepages-1Gi` request/limit, the `/dev/hugepages-1G` volume + mounts, and the `cleanup-hugepages` initContainer. |
| `podMonitor.enabled` | no | `false` | Emit a `PodMonitor` scraping the server's `/metrics`. Requires the Prometheus Operator CRDs (`monitoring.coreos.com`); leave `false` on clusters without them. |
| `podMonitor.labels` | no | `{release: prometheus}` | Labels on the `PodMonitor`; must match your Prometheus's `podMonitorSelector` or it won't be scraped. The default matches tt-telemetry's chart and the kube-prometheus-stack convention (`podMonitorSelector` defaults to `release: <the stack's release name>`); override when yours is named differently. |
| `podMonitor.interval` | no | `30s` | Scrape interval. |
| `podMonitor.path` | no | `/metrics` | Metrics HTTP path (scraped on the `http` port). |
| `cache.hostPath` | no | `""` | Override the host path used for the ttnn cache volume. Defaults to `/opt/cache/<model>-<device>-<impl>`. |
| `nameOverride` | no | `""` | Overrides the chart name component in resource names. |
| `fullnameOverride` | no | `""` | Fully overrides the resource name prefix. |
| `models` | — | See [Supported Models](#supported-models) | Per-model catalogue keyed by `<model>.<engine>.<device>.impls.<impl>`. Each impl leaf overrides `defaults`. |

## Defaults Reference

All fields under `defaults` apply to every model/engine/device/impl unless overridden by the resolved impl block.

| Field | Default | Description |
|---|---|---|
| `defaults.replicaCount` | `1` | Number of Deployment replicas. |
| `defaults.progressDeadlineSeconds` | `3600` | Deployment progress deadline. Set high due to long model load times. |
| `defaults.dra.deviceClassName` | `tenstorrent.com` | DRA `DeviceClass` (published by tt-operator) that the `ResourceClaim` selects. The board `count` is not set here — it is derived from `device` by `tt-inference-server.draDeviceCount`. |
| `defaults.updateStrategy` | `RollingUpdate` (maxSurge 0, maxUnavailable 1) | Deployment update strategy. `maxSurge=0` terminates the old Pod before creating its replacement so its board and hugepages are freed first; `maxUnavailable=1` rolls one Pod at a time when `replicaCount > 1`. |
| `defaults.podAnnotations` | `{}` | Annotations applied to the pod template. |
| `defaults.podSecurityContext` | `{}` | Pod-level `securityContext`. |
| `defaults.image.pullPolicy` | `IfNotPresent` | Image pull policy. |
| `defaults.image.pullSecrets` | `[]` | Image pull secrets. |
| `defaults.service.type` | `ClusterIP` | Kubernetes Service type. |
| `defaults.service.port` | `8000` | Service port. |
| `defaults.service.targetPort` | `8000` | Container target port. |
| `defaults.service.annotations` | `{}` | Annotations applied to the Service. |
| `defaults.resources.limits.hugepages-1Gi` | `32Gi` | Hugepage limit. Dropped when `hugepages.enabled=false`. (`cpu` and `memory` limits are unset by default.) |
| `defaults.resources.requests.cpu` | `"6"` | CPU request. |
| `defaults.resources.requests.memory` | `64Gi` | Memory request (often overridden per impl). |
| `defaults.resources.requests.hugepages-1Gi` | `32Gi` | Hugepage request. Dropped when `hugepages.enabled=false`. |
| `defaults.probes.startup.periodSeconds` | `15` | startupProbe poll interval (must be `> 0`). The startupProbe is always rendered — it owns the model compile/warmup window, and while it runs the kubelet suppresses liveness/readiness so a slow compile never triggers a restart. `failureThreshold` is not set here — the Deployment derives it as `ceil(progressDeadlineSeconds / periodSeconds)`, so the startup budget tracks the max compile time and the Pod flips to `1/1` as soon as one poll succeeds. |
| `defaults.probes.liveness.enabled` | `true` | Enable liveness probe. |
| `defaults.probes.liveness.path` | `/v1/models` | Liveness probe HTTP path (`/tt-liveness` for non-vllm engines). |
| `defaults.probes.liveness.initialDelaySeconds` | `0` | Liveness probe initial delay. `0` because the startupProbe gates the compile window. |
| `defaults.probes.readiness.enabled` | `true` | Enable readiness probe. |
| `defaults.probes.readiness.path` | `/health` | Readiness probe HTTP path (also used by the startupProbe). |
| `defaults.probes.readiness.initialDelaySeconds` | `0` | Readiness probe initial delay. `0` because the startupProbe gates the compile window. |
| `defaults.nodeSelector` | `{}` | Node selector applied to the pod. |
| `defaults.tolerations` | `[]` | Tolerations applied to the pod. |
| `defaults.affinity` | `{}` | User-supplied affinity, applied as-is. v0.2 no longer injects a 1-Pod-per-Node `podAntiAffinity` — DRA prevents device collisions by allocation, so multiple Pods may run on one Node. |
| `defaults.extraEnv` | `[]` | Additional environment variables (see [Extra Environment Variables](#extra-environment-variables)). |

---

## Supported Models

### vLLM

| Model | Devices |
|---|---|
| `AFM-4.5B` | n300, t3k |
| `DeepSeek-R1-0528` | galaxy |
| `DeepSeek-R1-Distill-Llama-70B` | galaxy, p150x4, p150x8, p300x2, t3k |
| `Llama-3.1-70B` | galaxy, p150x4, p150x8, p300x2, t3k |
| `Llama-3.1-70B-Instruct` | galaxy, p150x4, p150x8, p300x2, t3k |
| `Llama-3.1-8B` | galaxy, gpu, n150, n300, p100, p150, p150x4, p150x8, p300, p300x2, t3k |
| `Llama-3.1-8B-Instruct` | galaxy, gpu, n150, n300, p100, p150, p150x4, p150x8, p300, p300x2, t3k |
| `Llama-3.2-11B-Vision` | n300, t3k |
| `Llama-3.2-11B-Vision-Instruct` | n300, t3k |
| `Llama-3.2-1B` | n150, n300, t3k |
| `Llama-3.2-1B-Instruct` | n150, n300, t3k |
| `Llama-3.2-3B` | n150, n300, t3k |
| `Llama-3.2-3B-Instruct` | n150, n300, t3k |
| `Llama-3.2-90B-Vision` | t3k |
| `Llama-3.2-90B-Vision-Instruct` | t3k |
| `Llama-3.3-70B-Instruct` | galaxy, p150x4, p150x8, p300x2, t3k |
| `Mistral-7B-Instruct-v0.3` | n150, n300, t3k |
| `QwQ-32B` | galaxy, t3k |
| `Qwen2.5-72B` | galaxy, t3k |
| `Qwen2.5-72B-Instruct` | galaxy, t3k |
| `Qwen2.5-7B` | n150x4, n300 |
| `Qwen2.5-7B-Instruct` | n150x4, n300 |
| `Qwen2.5-Coder-32B-Instruct` | t3k |
| `Qwen2.5-VL-32B-Instruct` | t3k |
| `Qwen2.5-VL-3B-Instruct` | n150, n300 |
| `Qwen2.5-VL-72B-Instruct` | gpu, t3k |
| `Qwen2.5-VL-7B-Instruct` | n150, n300 |
| `Qwen3-32B` | galaxy, p150x8, p300x2, t3k |
| `Qwen3-8B` | galaxy, n150, n300, p300, t3k |
| `Qwen3-VL-32B-Instruct` | t3k |
| `Qwen3.6-27B` | p150x8, p300x2 |
| `gemma-3-1b-it` | n150 |
| `gemma-3-27b-it` | galaxy, p300x2, t3k |
| `gemma-3-4b-it` | n150, n300, p150, t3k |
| `gemma-4-31B-it` | p300x2 |
| `gpt-oss-120b` | galaxy, p300x2, t3k |
| `gpt-oss-20b` | galaxy, t3k |
| `medgemma-27b-it` | galaxy, p300x2, t3k |
| `medgemma-4b-it` | n150, n300, p150, t3k |

### Media

| Model | Devices |
|---|---|
| `FLUX.1-dev` | galaxy, p150x4, p150x8, p300, p300x2, t3k |
| `FLUX.1-schnell` | galaxy, p150x4, p150x8, p300, p300x2, t3k |
| `Llama-3.1-70B` | t3k |
| `Motif-Image-6B-Preview` | galaxy, p150x8, p300x2, t3k |
| `Qwen-Image` | galaxy, t3k |
| `Qwen-Image-2512` | galaxy, t3k |
| `Qwen3-Embedding-8B` | galaxy, n150, n300, t3k |
| `Wan2.2-I2V-A14B-Diffusers` | galaxy, p150x4, p150x8, p300x2, t3k |
| `Wan2.2-T2V-A14B-Diffusers` | galaxy, p150x4, p150x8, p300x2, t3k |
| `Z-Image-Turbo` | p300x2 |
| `bge-large-en-v1.5` | galaxy, n150, n300, t3k |
| `distil-large-v3` | galaxy, n150, n300, p150, p300, p300x2, t3k |
| `mochi-1-preview` | galaxy, p150x4, p150x8, p300x2, t3k |
| `speecht5_tts` | n150, n300, p150, p300, p300x2 |
| `stable-diffusion-3.5-large` | galaxy, t3k |
| `stable-diffusion-xl-1.0-inpainting-0.1` | galaxy, n150, n300, p150, p150x4, p150x8, p300x2, t3k |
| `stable-diffusion-xl-base-1.0` | galaxy, n150, n300, p150, p150x4, p150x8, p300x2, t3k |
| `stable-diffusion-xl-base-1.0-img-2-img` | galaxy, n150, n300, p150, p150x4, p150x8, p300x2, t3k |
| `whisper-large-v3` | galaxy, n150, n300, p150, p300, p300x2, t3k |

### Forge

| Model | Devices |
|---|---|
| `Falcon3-7B-Instruct` | n150, n300, p150 |
| `Llama-3.1-8B-Instruct` | n150, n300, p150 |
| `Llama-3.2-3B` | n150, n300, p150 |
| `Llama-3.2-3B-Instruct` | n150, n300, p150 |
| `Qwen3-4B` | n150, n300, p150 |
| `Qwen3-8B` | n150, n300, p150 |
| `Qwen3-Embedding-0.6B` | p300x2 |
| `Qwen3-Embedding-4B` | galaxy, n150, n300, p300x2, t3k |
| `bge-m3` | p300x2 |
| `efficientnet` | n150, n300 |
| `mobilenetv2` | n150, n300 |
| `resnet-50` | n150, n300 |
| `segformer` | n150, n300 |
| `stable-diffusion-xl-base-1.0` | p150x8, p300x2 |
| `unet` | n150, n300 |
| `vit` | n150, n300 |
| `vovnet` | n150, n300 |
| `yolox_nano` | n150, p150 |

To add a new model, add an entry under `models.<name>.<engine>.<device>` in `values.yaml`, where `<engine>` is one of `vllm`, `media`, or `forge`, and the device block contains an `impls.<impl-id>` entry with `image.repository` and `image.tag`.

---

## Advanced Usage

### Pre-downloaded Weights

If model weights are already present on the node, set `hfCacheDir` to skip the download step:

```bash
helm install my-model ./charts/tt-inference-server \
  --set model="Llama-3.1-8B-Instruct" \
  --set device=galaxy \
  --set hfCacheDir="/data/weights/Llama-3.1-8B-Instruct"
```

The host path is mounted read-only at `/mnt/hf-cache` inside the container. The chart sets `MODEL_WEIGHTS_DIR` (vLLM) or `MODEL_WEIGHTS_PATH` + `DOWNLOAD_WEIGHTS_FROM_SERVICE=false` (media) accordingly.

### Authentication

The media and forge servers check a literal `Authorization: Bearer $API_KEY` on
their inference routes (`/health` and `/v1/models` stay open) and fall back to a
well-known built-in key when `API_KEY` is unset — authenticated-looking, but not
authenticated. The chart therefore fails the install for those engines until you
choose:

```bash
# authenticate with your own key (lands in the release Secret as API_KEY)
helm install my-model ./charts/tt-inference-server \
  --set model="Qwen3-Embedding-4B" --set device=n150 \
  --set auth.apiKey="$(openssl rand -hex 16)"

# or run without auth, e.g. behind your own gateway
helm install my-model ./charts/tt-inference-server \
  --set model="Qwen3-Embedding-4B" --set device=n150 \
  --set auth.disabled=true
```

vLLM engines need no such gate: they are unauthenticated unless `VLLM_API_KEY`
is set, which is where `auth.apiKey` lands for them.

### Extra Environment Variables

Inject arbitrary environment variables via `defaults.extraEnv`. Each entry supports either a literal `value` or a `valueFrom` reference:

```yaml
# values override file
defaults:
  extraEnv:
    - name: VLLM_WORKER_MULTIPROC_METHOD
      value: "spawn"
    - name: MY_SECRET
      valueFrom:
        secretKeyRef:
          name: my-secret
          key: my-key
```

Literal values are written into the ConfigMap. `valueFrom` entries are injected directly into the container spec and are not stored in the ConfigMap.

For `media` and `forge` engines the chart also sets
`HF_HOME=/home/container_app_user/cache_root/huggingface`, so downloaded weights
land on the cache volume instead of the container's ephemeral layer and survive a
Pod restart. vLLM images manage their own weight location under `CACHE_ROOT`.

### Custom Node Scheduling

Pin inference pods to specific nodes using `defaults.nodeSelector`, `defaults.tolerations`, or `defaults.affinity`:

```yaml
defaults:
  nodeSelector:
    kubernetes.io/hostname: galaxy-node-01
  tolerations:
    - key: "tenstorrent.com/device"
      operator: "Exists"
      effect: "NoSchedule"
```

### Overriding the Cache Path

By default, the cache volume is mounted from `/opt/cache/<model>-<device>` on the host. Override with:

```bash
helm install my-model ./charts/tt-inference-server \
  --set model="Llama-3.1-8B-Instruct" \
  --set device=galaxy \
  --set hfToken="hf_xxx" \
  --set cache.hostPath="/mnt/fast-nvme/cache"
```

### Monitoring

Metrics come from two independent sources, each enabled separately:

| Source | Measures | Owned by | Enable with |
|---|---|---|---|
| App metrics | Inference server: request rate, latency, tokens/s, queue depth | this chart (consumer) | `podMonitor.enabled=true` |
| Device telemetry | Tenstorrent boards: temperature, power, chip utilization | tt-operator (supply side) | `tt-telemetry.enabled=true` on tt-operator |

**App metrics** — vLLM and media-server expose `/metrics`. Set `podMonitor.enabled=true` to emit a `PodMonitor` that Prometheus scrapes:

```bash
helm install my-model ./charts/tt-inference-server \
  --set model="Llama-3.1-8B-Instruct" \
  --set device=galaxy \
  --set hfToken="hf_xxx" \
  --set podMonitor.enabled=true
```

`podMonitor.labels` defaults to `release: prometheus`, matching tt-telemetry's chart and a kube-prometheus-stack installed under the release name `prometheus`. If your Prometheus release is named differently, override it (`--set podMonitor.labels.release=<release-name>`) or the `PodMonitor` won't be scraped; see the [Values Reference](#values-reference) for the other `podMonitor.*` options and the Prometheus Operator CRD requirement.

The chart ships no Grafana dashboard (metrics differ per engine). For vLLM, import the [official vLLM dashboard](https://docs.vllm.ai/en/stable/examples/online_serving/prometheus_grafana/) ([JSON](https://github.com/vllm-project/vllm/tree/main/examples/observability/prometheus_grafana)) via Grafana → Import, pointed at your Prometheus. For media-server / forge, build panels in Grafana Explore from the scraped metrics.

**Device telemetry** is not part of this chart. It is a tt-operator component (`tt-telemetry`) — a cluster-wide singleton with a separate lifecycle. The cluster admin enables it by setting the `tt-telemetry.enabled=true` subchart value when installing the tt-operator umbrella chart (`oci://ghcr.io/tenstorrent/helm/tt-operator`); see the [tt-telemetry documentation](https://docs.tenstorrent.com/tt-telemetry/latest/) for details.

Correlating the two (device × app) is future work — it needs a supply-side bridge exporter that maps devices to pods.
