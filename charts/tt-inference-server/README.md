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
| cache.hostPath | string | `""` |  |
| defaults | object | `{"affinity":{},"extraEnv":[],"image":{"pullPolicy":"IfNotPresent","pullSecrets":[]},"nodeSelector":{},"podAnnotations":{},"podSecurityContext":{},"probes":{"liveness":{"enabled":true,"initialDelaySeconds":2400,"path":"/v1/models"},"readiness":{"enabled":true,"initialDelaySeconds":2400,"path":"/health"}},"progressDeadlineSeconds":3600,"replicaCount":1,"resources":{"limits":{"cpu":"8","hugepages-1Gi":"32Gi","memory":"128Gi"},"requests":{"cpu":"6","hugepages-1Gi":"32Gi","memory":"64Gi"}},"serverType":"vllm","service":{"annotations":{},"port":8000,"targetPort":8000,"type":"ClusterIP"},"tolerations":[]}` | ------------------------------------------------------------------------- |
| device | string | `""` |  |
| fullnameOverride | string | `""` |  |
| hfToken | string | `""` |  |
| model | string | `""` |  |
| models | object | `{"DeepSeek-R1-Distill-Llama-70B":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.10.0-e867533-8f36910"},"probes":{"liveness":{"initialDelaySeconds":5400},"readiness":{"initialDelaySeconds":5400}},"progressDeadlineSeconds":6000,"resources":{"limits":{"memory":"256Gi"},"requests":{"memory":"175Gi"}}},"serverType":"vllm"},"FLUX.1-dev":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.10.0-555f240"},"probes":{"liveness":{"initialDelaySeconds":600,"path":"/tt-liveness"},"readiness":{"initialDelaySeconds":600}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"6Gi"}}},"serverType":"media"},"FLUX.1-schnell":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.10.0-555f240"},"probes":{"liveness":{"initialDelaySeconds":600,"path":"/tt-liveness"},"readiness":{"initialDelaySeconds":600}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"6Gi"}}},"serverType":"media"},"Llama-3.1-70B":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.10.0-e867533-8f36910"},"probes":{"liveness":{"initialDelaySeconds":5400},"readiness":{"initialDelaySeconds":5400}},"progressDeadlineSeconds":6000,"resources":{"limits":{"memory":"256Gi"},"requests":{"memory":"175Gi"}}},"serverType":"vllm"},"Llama-3.1-70B-Instruct":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.10.0-e867533-8f36910"},"probes":{"liveness":{"initialDelaySeconds":5400},"readiness":{"initialDelaySeconds":5400}},"progressDeadlineSeconds":6000,"resources":{"limits":{"memory":"256Gi"},"requests":{"memory":"175Gi"}}},"serverType":"vllm"},"Llama-3.1-8B":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.11.1-bac8b34-7c6685a"},"probes":{"liveness":{"initialDelaySeconds":2400},"readiness":{"initialDelaySeconds":2400}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"20Gi"}}},"serverType":"vllm"},"Llama-3.1-8B-Instruct":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.11.1-bac8b34-7c6685a"},"probes":{"liveness":{"initialDelaySeconds":2400},"readiness":{"initialDelaySeconds":2400}},"resources":{"limits":{"memory":"150Gi"},"requests":{"memory":"50Gi"}}},"serverType":"vllm"},"Llama-3.3-70B-Instruct":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.10.0-e867533-8f36910"},"probes":{"liveness":{"initialDelaySeconds":5400},"readiness":{"initialDelaySeconds":5400}},"progressDeadlineSeconds":6000,"resources":{"limits":{"memory":"256Gi"},"requests":{"memory":"175Gi"}}},"serverType":"vllm"},"Qwen3-32B":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.11.1-bac8b34-7c6685a"},"probes":{"liveness":{"initialDelaySeconds":3600},"readiness":{"initialDelaySeconds":3600}},"progressDeadlineSeconds":4200,"resources":{"limits":{"memory":"256Gi"},"requests":{"memory":"128Gi"}}},"serverType":"vllm"},"Qwen3-8B":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.10.0-e0e0500-409b1cd"},"probes":{"liveness":{"initialDelaySeconds":2400},"readiness":{"initialDelaySeconds":2400}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"20Gi"}}},"serverType":"vllm"},"Wan2.2-T2V-A14B-Diffusers":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.10.0-555f240"},"probes":{"liveness":{"enabled":false},"readiness":{"enabled":false}},"resources":{"limits":{"memory":"200Gi"},"requests":{"memory":"100Gi"}}},"serverType":"media"},"distil-large-v3":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.11.1-bac8b34"},"probes":{"liveness":{"initialDelaySeconds":600,"path":"/tt-liveness"},"readiness":{"initialDelaySeconds":600}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"6Gi"}}},"serverType":"media"},"gpt-oss-120b":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64","tag":"0.11.1-bac8b34-7c6685a"},"probes":{"liveness":{"initialDelaySeconds":7200},"readiness":{"initialDelaySeconds":7200}},"progressDeadlineSeconds":7800,"resources":{"limits":{"memory":"512Gi"},"requests":{"memory":"300Gi"}}},"serverType":"vllm"},"mochi-1-preview":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.10.0-555f240"},"probes":{"liveness":{"initialDelaySeconds":600,"path":"/tt-liveness"},"readiness":{"initialDelaySeconds":600}},"resources":{"limits":{"memory":"64Gi"},"requests":{"memory":"32Gi"}}},"serverType":"media"},"stable-diffusion-xl-base-1.0":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.10.0-555f240"},"probes":{"liveness":{"enabled":false},"readiness":{"enabled":false}},"resources":{"limits":{"memory":"480Gi"},"requests":{"memory":"200Gi"}}},"serverType":"media"},"stable-diffusion-xl-base-1.0-img-2-img":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.11.1-bac8b34"},"probes":{"liveness":{"initialDelaySeconds":600,"path":"/tt-liveness"},"readiness":{"initialDelaySeconds":600}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"6Gi"}}},"serverType":"media"},"whisper-large-v3":{"galaxy":{"image":{"repository":"ghcr.io/tenstorrent/tt-media-inference-server","tag":"0.11.1-bac8b34"},"probes":{"liveness":{"initialDelaySeconds":600,"path":"/tt-liveness"},"readiness":{"initialDelaySeconds":600}},"resources":{"limits":{"memory":"32Gi"},"requests":{"memory":"6Gi"}}},"serverType":"media"}}` | ------------------------------------------------------------------------- |
| nameOverride | string | `""` |  |

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
