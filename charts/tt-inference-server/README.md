# tt-inference-server Helm Chart

Deploys vLLM and media inference backends on Tenstorrent Galaxy hardware.

**Chart version:** 0.2.0 | **App version:** 0.12.0

---

## Overview

This chart creates a single `Deployment`, `Service`, `ConfigMap`, and `Secret` for one inference model on one Tenstorrent device. The chart ships with pre-validated configurations for every supported model and device combination. You select a model and device at install time; the chart merges your selection against a layered config system to produce the final Kubernetes resources.

Two server types are supported:

- **vllm** — large language models served via the vLLM OpenAI-compatible API
- **media** — image/audio/video models served via the tt-media-inference-server API

---

## Quick Start

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

Contains the core logic for this chart:

- **`tt-inference-server.validateValues`** — fails the render if `model` or `device` is missing, or if the `model`+`device` combination has no entry in the `models` map.
- **`tt-inference-server.resolvedConfig`** — deep-merges `defaults` with the per-model/device config block and returns the final effective config as YAML (see [Configuration System](#configuration-system)).
- **`tt-inference-server.image`** — assembles the container image string (`repository:tag`) from the resolved config.
- **`tt-inference-server.cacheHostPath`** — returns `cache.hostPath` if set, otherwise generates `/opt/cache/<model>-<device>`.
- Standard Helm name helpers: `tt-inference-server.name`, `tt-inference-server.fullname`, `tt-inference-server.labels`, `tt-inference-server.selectorLabels`.

---

## Configuration System

Values are resolved through a three-level merge. Later levels win on any conflict.

```
defaults                          ← baseline for all models and devices
  └── models.<name>               ← model-level fields (serverType)
        └── models.<name>.<device> ← deepest override, wins on all conflicts
```

**Example:** for `model=Llama-3.1-8B-Instruct` and `device=galaxy`, the effective config is:

1. Start with the full `defaults` block.
2. Apply the `serverType: vllm` field from `models.Llama-3.1-8B-Instruct`.
3. Deep-merge the `models.Llama-3.1-8B-Instruct.galaxy` block on top — overriding `image.repository`, `image.tag`, `resources`, and probe delays.

Any field not overridden at the model/device level falls back to `defaults`.

---

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| cache.hostPath | string | `""` | Override the host path used for the cache volume. Defaults to `/opt/cache/<model>-<device>` when empty. |
| defaults.affinity | object | `{}` | Pod affinity rules. When non-empty, replaces the auto-generated `podAntiAffinity` below. |
| defaults.extraEnv | list | `[]` | Additional environment variables. Items take the form `{name, value}` or `{name, valueFrom}`. |
| defaults.image.pullPolicy | string | `"IfNotPresent"` | Container image pull policy. |
| defaults.image.pullSecrets | list | `[]` | Image pull secrets for private registries. |
| defaults.nodeSelector | object | `{}` | Node selector applied to the pod. |
| defaults.podAnnotations | object | `{}` | Extra annotations applied to the pod template. |
| defaults.podAntiAffinity.enabled | bool | `true` | Enable the default anti-affinity rule that enforces 1:1 Pod:Node allocation. |
| defaults.podAntiAffinity.topologyKey | string | `"kubernetes.io/hostname"` | Topology key for anti-affinity. `kubernetes.io/hostname` enforces one pod per node. |
| defaults.podSecurityContext | object | `{}` | Pod-level `securityContext`. |
| defaults.probes.liveness.enabled | bool | `true` | Enable the liveness probe. |
| defaults.probes.liveness.initialDelaySeconds | int | `2400` | Seconds before liveness checks start. Tuned per model for warmup time. |
| defaults.probes.liveness.path | string | `"/v1/models"` | HTTP path for the liveness probe. vLLM uses `/v1/models`, media-server uses `/tt-liveness`. |
| defaults.probes.readiness.enabled | bool | `true` | Enable the readiness probe. |
| defaults.probes.readiness.initialDelaySeconds | int | `2400` | Seconds before readiness checks start. Tuned per model for warmup time. |
| defaults.probes.readiness.path | string | `"/health"` | HTTP path for the readiness probe. |
| defaults.progressDeadlineSeconds | int | `3600` | Deployment `progressDeadlineSeconds`. Raised to accommodate long model warmup times. |
| defaults.replicaCount | int | `1` | Number of Deployment replicas. Effective maximum is constrained to the node count by `podAntiAffinity`. |
| defaults.resources.limits.cpu | string | `"8"` | CPU limit. |
| defaults.resources.limits.hugepages-1Gi | string | `"32Gi"` | Hugepage (1Gi) limit. |
| defaults.resources.limits.memory | string | `"128Gi"` | Memory limit. Overridden per model. |
| defaults.resources.requests.cpu | string | `"6"` | CPU request. |
| defaults.resources.requests.hugepages-1Gi | string | `"32Gi"` | Hugepage (1Gi) request. |
| defaults.resources.requests.memory | string | `"64Gi"` | Memory request. Overridden per model. |
| defaults.serverType | string | `"vllm"` | Server backend. `vllm` for LLMs, `media` for tt-media-server. Overridden per model. |
| defaults.service.annotations | object | `{}` | Annotations applied to the Service. |
| defaults.service.port | int | `8000` | Service port exposed inside the cluster. |
| defaults.service.targetPort | int | `8000` | Container target port; must match the inference server's listen port. |
| defaults.service.type | string | `"ClusterIP"` | Kubernetes Service type. |
| defaults.tolerations | list | `[]` | Tolerations applied to the pod. |
| defaults.updateStrategy | object | `{"rollingUpdate":{"maxSurge":0,"maxUnavailable":1},"type":"RollingUpdate"}` | Deployment update strategy. `maxSurge: 0` keeps single-node upgrades from hanging under `podAntiAffinity`. |
| device | string | `""` | Device name. Must match a key under `models.<model>`. Required at install time. |
| fullnameOverride | string | `""` | Fully overrides the resource name prefix. |
| hfToken | string | `""` | HuggingFace token. Required for gated or private model weights. Injected as `HF_TOKEN`. |
| model | string | `""` | Model name. Must match a key under `models`. Required at install time. |
| models | object | See the Supported Models section below. | Per-model catalogue keyed by `<model-name>.<device-name>`. Each leaf overrides `defaults:` for image, resources, and probes. See the Supported Models section in the README for the full list. |
| nameOverride | string | `""` | Overrides the chart-name component of generated resource names. |

---

## Supported Models

### vLLM

| Model | Device |
|---|---|
| `Llama-3.1-8B-Instruct` | galaxy |
| `Llama-3.1-8B` | galaxy |
| `Llama-3.1-70B-Instruct` | galaxy |
| `Llama-3.1-70B` | galaxy |
| `Llama-3.3-70B-Instruct` | galaxy |
| `DeepSeek-R1-Distill-Llama-70B` | galaxy |
| `Qwen3-8B` | galaxy |
| `Qwen3-32B` | galaxy |
| `gpt-oss-120b` | galaxy |

### Media

| Model | Device |
|---|---|
| `whisper-large-v3` | galaxy |
| `distil-large-v3` | galaxy |
| `stable-diffusion-xl-base-1.0` | galaxy |
| `stable-diffusion-xl-base-1.0-img-2-img` | galaxy |
| `FLUX.1-dev` | galaxy |
| `FLUX.1-schnell` | galaxy |
| `Wan2.2-T2V-A14B-Diffusers` | galaxy |
| `mochi-1-preview` | galaxy |

To add a new model, add an entry under `models` in `values.yaml` with at least one device sub-key containing `image.repository` and `image.tag`.

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

---

## Init Containers

Each pod runs two init containers before the inference server starts:

**`fix-cache-permissions`**
Runs `chown -R 1000:1000 /cache` on the cache host path volume. The inference server runs as UID 1000 and requires write access to the cache directory, which may be created by root on the host.

**`cleanup-hugepages`**
Removes stale hugepage files left by previous runs (`/dev/hugepages-1G/device_*_tenstorrent` and `/dev/hugepages-1G/tenstorrent`). Runs privileged. Without this cleanup, the inference server may fail to acquire hugepages if a previous pod exited uncleanly.
