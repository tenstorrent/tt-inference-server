# tt-inference-server Helm Chart

Deploys vLLM and media inference backends on Tenstorrent Galaxy hardware.

**Chart version:** 0.1.0 | **App version:** 0.12.0

---

## Overview

This chart creates a single `Deployment`, `Service`, `ConfigMap`, and `Secret` for one inference model on one Tenstorrent device. The chart ships with pre-validated configurations for every supported model and device combination. You select a model and device at install time; the chart merges your selection against a layered config system to produce the final Kubernetes resources.

Two server types are supported:

- **vllm** — large language models served via the vLLM OpenAI-compatible API
- **media** — image/audio/video/embedding models served via the tt-media-inference-server API

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
| cache.hostPath | string | `""` | Override the host path used for the ttnn cache volume. Defaults to `/opt/cache/<model>-<device>`. |
| defaults.affinity | object | `{}` | Affinity rules applied to the pod. |
| defaults.extraEnv | list | `[]` | Additional environment variables (see [Extra Environment Variables](#extra-environment-variables)). |
| defaults.image.pullPolicy | string | `"IfNotPresent"` | Image pull policy. |
| defaults.image.pullSecrets | list | `[]` | Image pull secrets. |
| defaults.nodeSelector | object | `{}` | Node selector applied to the pod. |
| defaults.podAnnotations | object | `{}` | Annotations applied to the pod template. |
| defaults.podSecurityContext | object | `{}` | Pod-level `securityContext`. |
| defaults.probes.liveness.enabled | bool | `true` | Enable liveness probe. |
| defaults.probes.liveness.initialDelaySeconds | int | `2400` | Liveness probe initial delay. Set high due to model load times. |
| defaults.probes.liveness.path | string | `"/v1/models"` | Liveness probe HTTP path. |
| defaults.probes.readiness.enabled | bool | `true` | Enable readiness probe. |
| defaults.probes.readiness.initialDelaySeconds | int | `2400` | Readiness probe initial delay. |
| defaults.probes.readiness.path | string | `"/health"` | Readiness probe HTTP path. |
| defaults.progressDeadlineSeconds | int | `3600` | Deployment progress deadline. Set high due to long model load times. |
| defaults.replicaCount | int | `1` | Number of Deployment replicas. |
| defaults.resources.limits.cpu | string | `"8"` | CPU limit. |
| defaults.resources.limits.hugepages-1Gi | string | `"32Gi"` | Hugepage limit. |
| defaults.resources.limits.memory | string | `"128Gi"` | Memory limit (overridden per model). |
| defaults.resources.requests.cpu | string | `"6"` | CPU request. |
| defaults.resources.requests.hugepages-1Gi | string | `"32Gi"` | Hugepage request. |
| defaults.resources.requests.memory | string | `"64Gi"` | Memory request (overridden per model). |
| defaults.serverType | string | `"vllm"` | Server backend: `vllm` or `media`. |
| defaults.service.annotations | object | `{}` | Annotations applied to the Service. |
| defaults.service.port | int | `8000` | Service port. |
| defaults.service.targetPort | int | `8000` | Container target port. |
| defaults.service.type | string | `"ClusterIP"` | Kubernetes Service type. |
| defaults.tolerations | list | `[]` | Tolerations applied to the pod. |
| device | string | `""` | Device name. Must match a key under `models.<model>`. |
| fullnameOverride | string | `""` | Fully overrides the resource name prefix. |
| hfToken | string | `""` | HuggingFace token. Injected as `HF_TOKEN`. Required unless weights are pre-downloaded. |
| model | string | `""` | Model name. Must match a key in `models`. |
| models | object | See the Supported Models section below. | Per-model catalogue keyed by `<model-name>.<device-name>`. Each leaf overrides `defaults:` for image, resources, and probes. See the Supported Models section below for the full list. |
| nameOverride | string | `""` | Overrides the chart name component in resource names. |

---

## Supported Models

### vLLM

| Model | Device |
|---|---|
| `AFM-4.5B` | n300 |
| `AFM-4.5B` | t3k |
| `DeepSeek-R1-0528` | galaxy |
| `DeepSeek-R1-Distill-Llama-70B` | galaxy |
| `DeepSeek-R1-Distill-Llama-70B` | galaxy_t3k |
| `DeepSeek-R1-Distill-Llama-70B` | p150x4 |
| `DeepSeek-R1-Distill-Llama-70B` | p150x8 |
| `DeepSeek-R1-Distill-Llama-70B` | p300x2 |
| `DeepSeek-R1-Distill-Llama-70B` | t3k |
| `Llama-3.1-70B` | galaxy |
| `Llama-3.1-70B` | galaxy_t3k |
| `Llama-3.1-70B` | p150x4 |
| `Llama-3.1-70B` | p150x8 |
| `Llama-3.1-70B` | p300x2 |
| `Llama-3.1-70B` | t3k |
| `Llama-3.1-70B-Instruct` | galaxy |
| `Llama-3.1-70B-Instruct` | galaxy_t3k |
| `Llama-3.1-70B-Instruct` | p150x4 |
| `Llama-3.1-70B-Instruct` | p150x8 |
| `Llama-3.1-70B-Instruct` | p300x2 |
| `Llama-3.1-70B-Instruct` | t3k |
| `Llama-3.1-8B` | galaxy |
| `Llama-3.1-8B` | galaxy_t3k |
| `Llama-3.1-8B` | gpu |
| `Llama-3.1-8B` | n150 |
| `Llama-3.1-8B` | n300 |
| `Llama-3.1-8B` | p100 |
| `Llama-3.1-8B` | p150 |
| `Llama-3.1-8B` | p150x4 |
| `Llama-3.1-8B` | p150x8 |
| `Llama-3.1-8B` | p300 |
| `Llama-3.1-8B` | p300x2 |
| `Llama-3.1-8B` | t3k |
| `Llama-3.1-8B-Instruct` | galaxy |
| `Llama-3.1-8B-Instruct` | galaxy_t3k |
| `Llama-3.1-8B-Instruct` | gpu |
| `Llama-3.1-8B-Instruct` | n150 |
| `Llama-3.1-8B-Instruct` | n300 |
| `Llama-3.1-8B-Instruct` | p100 |
| `Llama-3.1-8B-Instruct` | p150 |
| `Llama-3.1-8B-Instruct` | p150x4 |
| `Llama-3.1-8B-Instruct` | p150x8 |
| `Llama-3.1-8B-Instruct` | p300 |
| `Llama-3.1-8B-Instruct` | p300x2 |
| `Llama-3.1-8B-Instruct` | t3k |
| `Llama-3.2-11B-Vision` | n300 |
| `Llama-3.2-11B-Vision` | t3k |
| `Llama-3.2-11B-Vision-Instruct` | n300 |
| `Llama-3.2-11B-Vision-Instruct` | t3k |
| `Llama-3.2-1B` | n150 |
| `Llama-3.2-1B` | n300 |
| `Llama-3.2-1B` | t3k |
| `Llama-3.2-1B-Instruct` | n150 |
| `Llama-3.2-1B-Instruct` | n300 |
| `Llama-3.2-1B-Instruct` | t3k |
| `Llama-3.2-3B` | n150 |
| `Llama-3.2-3B` | n300 |
| `Llama-3.2-3B` | t3k |
| `Llama-3.2-3B-Instruct` | n150 |
| `Llama-3.2-3B-Instruct` | n300 |
| `Llama-3.2-3B-Instruct` | t3k |
| `Llama-3.2-90B-Vision` | t3k |
| `Llama-3.2-90B-Vision-Instruct` | t3k |
| `Llama-3.3-70B-Instruct` | galaxy |
| `Llama-3.3-70B-Instruct` | galaxy_t3k |
| `Llama-3.3-70B-Instruct` | p150x4 |
| `Llama-3.3-70B-Instruct` | p150x8 |
| `Llama-3.3-70B-Instruct` | p300x2 |
| `Llama-3.3-70B-Instruct` | t3k |
| `Mistral-7B-Instruct-v0.3` | n150 |
| `Mistral-7B-Instruct-v0.3` | n300 |
| `Mistral-7B-Instruct-v0.3` | t3k |
| `Mistral-Small-3.1-24B-Instruct-2503` | t3k |
| `QwQ-32B` | galaxy |
| `QwQ-32B` | galaxy_t3k |
| `QwQ-32B` | t3k |
| `Qwen2.5-72B` | galaxy |
| `Qwen2.5-72B` | galaxy_t3k |
| `Qwen2.5-72B` | t3k |
| `Qwen2.5-72B-Instruct` | galaxy |
| `Qwen2.5-72B-Instruct` | galaxy_t3k |
| `Qwen2.5-72B-Instruct` | t3k |
| `Qwen2.5-7B` | n150x4 |
| `Qwen2.5-7B` | n300 |
| `Qwen2.5-7B-Instruct` | n150x4 |
| `Qwen2.5-7B-Instruct` | n300 |
| `Qwen2.5-Coder-32B-Instruct` | galaxy_t3k |
| `Qwen2.5-Coder-32B-Instruct` | t3k |
| `Qwen2.5-VL-32B-Instruct` | t3k |
| `Qwen2.5-VL-3B-Instruct` | n150 |
| `Qwen2.5-VL-3B-Instruct` | n300 |
| `Qwen2.5-VL-72B-Instruct` | t3k |
| `Qwen2.5-VL-7B-Instruct` | n150 |
| `Qwen2.5-VL-7B-Instruct` | n300 |
| `Qwen3-32B` | galaxy |
| `Qwen3-32B` | galaxy_t3k |
| `Qwen3-32B` | p150x8 |
| `Qwen3-32B` | p300x2 |
| `Qwen3-32B` | t3k |
| `Qwen3-8B` | galaxy |
| `Qwen3-8B` | galaxy_t3k |
| `Qwen3-8B` | n150 |
| `Qwen3-8B` | n300 |
| `Qwen3-8B` | p300 |
| `Qwen3-8B` | t3k |
| `Qwen3-VL-32B-Instruct` | t3k |
| `gemma-3-1b-it` | n150 |
| `gemma-3-27b-it` | galaxy |
| `gemma-3-27b-it` | galaxy_t3k |
| `gemma-3-27b-it` | t3k |
| `gemma-3-4b-it` | n150 |
| `gemma-3-4b-it` | n300 |
| `gpt-oss-120b` | galaxy |
| `gpt-oss-120b` | t3k |
| `gpt-oss-20b` | galaxy |
| `gpt-oss-20b` | galaxy_t3k |
| `gpt-oss-20b` | t3k |
| `medgemma-27b-it` | galaxy |
| `medgemma-27b-it` | galaxy_t3k |
| `medgemma-27b-it` | t3k |
| `medgemma-4b-it` | n150 |
| `medgemma-4b-it` | n300 |

### Media

| Model | Device |
|---|---|
| `FLUX.1-dev` | galaxy |
| `FLUX.1-dev` | p150x4 |
| `FLUX.1-dev` | p150x8 |
| `FLUX.1-dev` | p300 |
| `FLUX.1-dev` | p300x2 |
| `FLUX.1-dev` | t3k |
| `FLUX.1-schnell` | galaxy |
| `FLUX.1-schnell` | p150x4 |
| `FLUX.1-schnell` | p150x8 |
| `FLUX.1-schnell` | p300 |
| `FLUX.1-schnell` | p300x2 |
| `FLUX.1-schnell` | t3k |
| `Llama-3.1-70B` | t3k |
| `Motif-Image-6B-Preview` | galaxy |
| `Motif-Image-6B-Preview` | p150x8 |
| `Motif-Image-6B-Preview` | p300x2 |
| `Motif-Image-6B-Preview` | t3k |
| `Qwen-Image` | galaxy |
| `Qwen-Image` | t3k |
| `Qwen-Image-2512` | galaxy |
| `Qwen-Image-2512` | t3k |
| `Qwen3-Embedding-8B` | galaxy |
| `Qwen3-Embedding-8B` | n150 |
| `Qwen3-Embedding-8B` | n300 |
| `Qwen3-Embedding-8B` | t3k |
| `Wan2.2-T2V-A14B-Diffusers` | galaxy |
| `Wan2.2-T2V-A14B-Diffusers` | p150x4 |
| `Wan2.2-T2V-A14B-Diffusers` | p150x8 |
| `Wan2.2-T2V-A14B-Diffusers` | p300x2 |
| `Wan2.2-T2V-A14B-Diffusers` | t3k |
| `bge-large-en-v1.5` | galaxy |
| `bge-large-en-v1.5` | n150 |
| `bge-large-en-v1.5` | n300 |
| `bge-large-en-v1.5` | t3k |
| `bge-m3` | galaxy |
| `bge-m3` | n150 |
| `bge-m3` | n300 |
| `bge-m3` | t3k |
| `distil-large-v3` | galaxy |
| `distil-large-v3` | n150 |
| `distil-large-v3` | n300 |
| `distil-large-v3` | p150 |
| `distil-large-v3` | p300 |
| `distil-large-v3` | p300x2 |
| `distil-large-v3` | t3k |
| `mochi-1-preview` | galaxy |
| `mochi-1-preview` | p150x4 |
| `mochi-1-preview` | p150x8 |
| `mochi-1-preview` | p300x2 |
| `mochi-1-preview` | t3k |
| `speecht5_tts` | n150 |
| `speecht5_tts` | n300 |
| `speecht5_tts` | p150 |
| `speecht5_tts` | p300 |
| `speecht5_tts` | p300x2 |
| `stable-diffusion-3.5-large` | galaxy |
| `stable-diffusion-3.5-large` | t3k |
| `stable-diffusion-xl-1.0-inpainting-0.1` | galaxy |
| `stable-diffusion-xl-1.0-inpainting-0.1` | n150 |
| `stable-diffusion-xl-1.0-inpainting-0.1` | n300 |
| `stable-diffusion-xl-1.0-inpainting-0.1` | t3k |
| `stable-diffusion-xl-base-1.0` | galaxy |
| `stable-diffusion-xl-base-1.0` | n150 |
| `stable-diffusion-xl-base-1.0` | n300 |
| `stable-diffusion-xl-base-1.0` | p150x4 |
| `stable-diffusion-xl-base-1.0` | p150x8 |
| `stable-diffusion-xl-base-1.0` | p300x2 |
| `stable-diffusion-xl-base-1.0` | t3k |
| `stable-diffusion-xl-base-1.0-img-2-img` | galaxy |
| `stable-diffusion-xl-base-1.0-img-2-img` | n150 |
| `stable-diffusion-xl-base-1.0-img-2-img` | n300 |
| `stable-diffusion-xl-base-1.0-img-2-img` | p150x4 |
| `stable-diffusion-xl-base-1.0-img-2-img` | p150x8 |
| `stable-diffusion-xl-base-1.0-img-2-img` | p300x2 |
| `stable-diffusion-xl-base-1.0-img-2-img` | t3k |
| `whisper-large-v3` | galaxy |
| `whisper-large-v3` | n150 |
| `whisper-large-v3` | n300 |
| `whisper-large-v3` | p150 |
| `whisper-large-v3` | p300 |
| `whisper-large-v3` | p300x2 |
| `whisper-large-v3` | t3k |

### Forge

| Model | Device |
|---|---|
| `Llama-3.2-3B` | n150 |
| `Llama-3.2-3B` | n300 |
| `Qwen3-4B` | n150 |
| `Qwen3-4B` | n300 |
| `Qwen3-Embedding-4B` | galaxy |
| `Qwen3-Embedding-4B` | n150 |
| `Qwen3-Embedding-4B` | n300 |
| `Qwen3-Embedding-4B` | t3k |
| `efficientnet` | n150 |
| `efficientnet` | n300 |
| `mobilenetv2` | n150 |
| `mobilenetv2` | n300 |
| `resnet-50` | n150 |
| `resnet-50` | n300 |
| `segformer` | n150 |
| `segformer` | n300 |
| `unet` | n150 |
| `unet` | n300 |
| `vit` | n150 |
| `vit` | n300 |
| `vovnet` | n150 |
| `vovnet` | n300 |

To add a new model, add an entry under `models` in `values.yaml` with at least one engine sub-key and at least one device under that engine.

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