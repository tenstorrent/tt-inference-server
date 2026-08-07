{{/*
Validate required values and that model/engine/device/impl resolves.
*/}}
{{- define "tt-inference-server.validateValues" -}}
{{- if not .Values.model }}
  {{- fail "model is required. Pass --set model=<model-name>" }}
{{- end }}

{{- if not .Values.device }}
  {{- fail "device is required. Pass --set device=<device-name>" }}
{{- end }}

{{- if not (hasKey .Values.models .Values.model) }}
  {{- $available := keys .Values.models | sortAlpha | join ", " }}
  {{- fail (printf "Unknown model '%s'. Available: %s" .Values.model $available) }}
{{- end }}

{{- $modelEntry := index .Values.models .Values.model }}
{{- $engine := include "tt-inference-server.resolvedEngine" . }}
{{- $engineEntry := index $modelEntry $engine }}

{{- if not (hasKey $engineEntry .Values.device) }}
  {{- $available := keys $engineEntry | sortAlpha | join ", " }}
  {{- fail (printf "No config for model '%s' on engine '%s' device '%s'. Available devices: %s" .Values.model $engine .Values.device $available) }}
{{- end }}

{{- $deviceEntry := index $engineEntry .Values.device }}
{{- $impl := include "tt-inference-server.resolvedImpl" . }}

{{- if not (hasKey $deviceEntry.impls $impl) }}
  {{- $available := keys $deviceEntry.impls | sortAlpha | join ", " }}
  {{- fail (printf "No impl '%s' for model '%s' on engine '%s' device '%s'. Available impls: %s" $impl .Values.model $engine .Values.device $available) }}
{{- end }}
{{- end }}

{{/*
Resolve the engine to use:
  - If .Values.engine is set, use it.
  - Else find all engine keys under models[model] that contain device.
    - If exactly one candidate, use it.
    - If multiple, use models[model].defaultEngine (must be set).
    - If zero, fail.
*/}}
{{- define "tt-inference-server.resolvedEngine" -}}
{{- $modelEntry := index .Values.models .Values.model }}
{{- if .Values.engine }}
{{- .Values.engine }}
{{- else }}
{{- $candidates := list }}
{{- range $engineKey, $engineEntry := $modelEntry }}
{{- if ne $engineKey "defaultEngine" }}
{{- if and (kindIs "map" $engineEntry) (hasKey $engineEntry $.Values.device) }}
{{- $candidates = append $candidates $engineKey }}
{{- end }}
{{- end }}
{{- end }}
{{- if eq (len $candidates) 1 }}
{{- index $candidates 0 }}
{{- else if gt (len $candidates) 1 }}
{{- if not $modelEntry.defaultEngine }}
  {{- fail (printf "Model '%s' device '%s' is available under multiple engines (%s) and 'defaultEngine' is not set. Pass --set engine=..." $.Values.model $.Values.device (join ", " $candidates)) }}
{{- end }}
{{- $modelEntry.defaultEngine }}
{{- else }}
  {{- fail (printf "Model '%s' has no engine that provides device '%s'." $.Values.model $.Values.device) }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Resolve the impl to use:
  - If .Values.impl is set, use it.
  - Else use models[model][engine][device].defaultImpl (must be set).
*/}}
{{- define "tt-inference-server.resolvedImpl" -}}
{{- if .Values.impl }}
{{- .Values.impl }}
{{- else }}
{{- $modelEntry := index .Values.models .Values.model }}
{{- $engine := include "tt-inference-server.resolvedEngine" . }}
{{- $deviceEntry := index (index $modelEntry $engine) .Values.device }}
{{- if not $deviceEntry.defaultImpl }}
  {{- fail (printf "No defaultImpl set for '%s' / %s / %s. Pass --set impl=..." .Values.model $engine .Values.device) }}
{{- end }}
{{- $deviceEntry.defaultImpl }}
{{- end }}
{{- end }}

{{/*
Resolve the effective config: deep-merge defaults with the impl block.

Engine is not stamped onto this config — callers that need to branch on it
should invoke "tt-inference-server.resolvedEngine" directly.

Usage (in a template):
  {{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml }}
*/}}
{{- define "tt-inference-server.resolvedConfig" -}}
{{- $modelEntry := index .Values.models .Values.model }}
{{- $engine := include "tt-inference-server.resolvedEngine" . }}
{{- $deviceEntry := index (index $modelEntry $engine) .Values.device }}
{{- $impl := include "tt-inference-server.resolvedImpl" . }}
{{- $implCfg := index $deviceEntry.impls $impl }}
{{- $cfg := mergeOverwrite (deepCopy .Values.defaults) $implCfg }}
{{- $cfg | toYaml }}
{{- end }}

{{/*
Container image string built from resolved config.
*/}}
{{- define "tt-inference-server.image" -}}
{{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml }}
{{- printf "%s:%s" $cfg.image.repository $cfg.image.tag }}
{{- end }}

{{/*
Container env list, merged from three independent sources:
  1. spec env  — $cfg.env (name/value pairs from the ModelSpec)
  2. hf-cache env  — MODEL_WEIGHTS_* + DOWNLOAD_WEIGHTS_FROM_SERVICE when
                     .Values.hfCacheDir is set
  3. extra valueFrom — $cfg.extraEnv entries that carry secret references etc.
Yields an empty string if nothing applies, so the caller can short-circuit
with `with (include … | trim)`.
*/}}
{{- define "tt-inference-server.containerEnv" -}}
{{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml -}}
{{- range $cfg.env }}
- name: {{ .name }}
  value: {{ .value | quote }}
{{- end }}
{{- if .Values.hfCacheDir }}
- name: MODEL_WEIGHTS_DIR
  value: "/mnt/hf-cache"
- name: MODEL_WEIGHTS_PATH
  value: "/mnt/hf-cache"
- name: DOWNLOAD_WEIGHTS_FROM_SERVICE
  value: "false"
{{- end }}
{{- range $cfg.extraEnv }}
{{- if .valueFrom }}
- name: {{ .name }}
  valueFrom:
    {{- toYaml .valueFrom | nindent 4 }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Cache hostPath — defaults to /opt/cache/<model>-<device>-<impl>. Includes impl
so two impls on the same device don't share a cache directory.
*/}}
{{- define "tt-inference-server.cacheHostPath" -}}
{{- if .Values.cache.hostPath }}
{{- .Values.cache.hostPath }}
{{- else }}
{{- $model := .Values.model | replace "/" "-" | replace "." "-" }}
{{- $device := .Values.device | lower }}
{{- $impl := include "tt-inference-server.resolvedImpl" . | replace "/" "-" | replace "." "-" }}
{{- printf "/opt/cache/%s-%s-%s" $model $device $impl }}
{{- end }}
{{- end }}

{{/*
DRA device count — how many Tenstorrent boards a ResourceClaim must request for
the selected device shape.

The count is purely a function of .Values.device (not the model), so it comes
from the top-level `.Values.deviceBoardCounts` map — a single source of truth
generated from workflows/device_utils.py:BOARD_TYPE_COUNT_TO_DEVICE, not hand-
maintained here.

A plain board count is only correct when the claim takes a whole dedicated
device (no wrong subset to pick). Topology-sensitive partitions — e.g.
galaxy_t3k, which carves a T3K-shaped mesh out of a Galaxy and must be pinned to
adjacent chips in one tray — are absent from the map and fail loudly; expressing
them needs DRA constraints matching topology attributes the tt-dra-driver does
not yet publish (future work).

Behaviour:
  - device in deviceBoardCounts -> returns the board count
  - non-Tenstorrent device       -> returns "" (no ResourceClaim; e.g. gpu/cpu)
  - anything else                -> fails the render (fail closed, rather than
    silently deploying a device-less Pod)
*/}}
{{- define "tt-inference-server.draDeviceCount" -}}
{{- $counts := .Values.deviceBoardCounts | default dict -}}
{{- $nonTT := list "gpu" "cpu" -}}
{{- $d := .Values.device | lower -}}
{{- if hasKey $counts $d -}}
{{- index $counts $d -}}
{{- else if has $d $nonTT -}}
{{- /* non-Tenstorrent device: intentionally no ResourceClaim */ -}}
{{- else -}}
{{- fail (printf "device '%s' is not supported via DRA in this chart. Topology-sensitive partitions (e.g. galaxy_t3k) are future work and need the tt-dra-driver to publish topology attributes. Supported: %s." $d (keys $counts | sortAlpha | join ", ")) -}}
{{- end -}}
{{- end -}}

{{/*
Chart name helpers
*/}}
{{- define "tt-inference-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tt-inference-server.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "tt-inference-server.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tt-inference-server.labels" -}}
helm.sh/chart: {{ include "tt-inference-server.chart" . }}
{{ include "tt-inference-server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "tt-inference-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tt-inference-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "tt-inference-server.configmapName" -}}
{{- printf "%s-config" (include "tt-inference-server.fullname" .) }}
{{- end }}

{{- define "tt-inference-server.secretName" -}}
{{- printf "%s-secret" (include "tt-inference-server.fullname" .) }}
{{- end }}

{{/*
Compose the final affinity object — user-supplied affinity, passed through as-is.

Under DRA the scheduler prevents device collisions by allocation (two Pods can
never be assigned the same board), so multiple Pods may run on one Node (e.g. 4
single-n300 Pods on a T3K). Operators can supply their own nodeAffinity /
podAffinity / podAntiAffinity via .Values.affinity.
*/}}
{{- define "tt-inference-server.affinity" -}}
{{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml -}}
{{- with $cfg.affinity -}}
{{- toYaml . -}}
{{- end -}}
{{- end -}}
