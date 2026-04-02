{{/*
Validate required values and that model+device exists in the models map.
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
{{- $deviceMap := index .Values.models .Values.model }}
{{- if not (hasKey $deviceMap .Values.device) }}
  {{- $available := keys $deviceMap | sortAlpha | join ", " }}
  {{- fail (printf "No config for model '%s' on device '%s'. Available devices: %s" .Values.model .Values.device $available) }}
{{- end }}
{{- end }}

{{/*
Resolve the effective config for the selected model+device by deep-merging
defaults with the per-model overrides. Model config wins on any conflict.

Usage (in a template):
  {{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml }}
*/}}
{{- define "tt-inference-server.resolvedConfig" -}}
{{- $modelCfg := index (index .Values.models .Values.model) .Values.device }}
{{- mergeOverwrite (deepCopy .Values.defaults) $modelCfg | toYaml }}
{{- end }}

{{/*
Container image string built from resolved config.
*/}}
{{- define "tt-inference-server.image" -}}
{{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml }}
{{- printf "%s:%s" $cfg.image.repository $cfg.image.tag }}
{{- end }}

{{/*
Cache hostPath — defaults to /opt/cache/<model>-<device>.
*/}}
{{- define "tt-inference-server.cacheHostPath" -}}
{{- if .Values.cache.hostPath }}
{{- .Values.cache.hostPath }}
{{- else }}
{{- $model := .Values.model | replace "/" "-" | replace "." "-" }}
{{- $device := .Values.device | lower }}
{{- printf "/opt/cache/%s-%s" $model $device }}
{{- end }}
{{- end }}

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
