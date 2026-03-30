{{/*
Expand the name of the chart.
*/}}
{{- define "tt-inference-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
Truncated to 63 chars — Kubernetes DNS label limit.
*/}}
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

{{/*
Chart label
*/}}
{{- define "tt-inference-server.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tt-inference-server.labels" -}}
helm.sh/chart: {{ include "tt-inference-server.chart" . }}
{{ include "tt-inference-server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tt-inference-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tt-inference-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ConfigMap name
*/}}
{{- define "tt-inference-server.configmapName" -}}
{{- printf "%s-config" (include "tt-inference-server.fullname" .) }}
{{- end }}

{{/*
Secret name
*/}}
{{- define "tt-inference-server.secretName" -}}
{{- printf "%s-secret" (include "tt-inference-server.fullname" .) }}
{{- end }}

{{/*
Container image — tag defaults to chart appVersion.
*/}}
{{- define "tt-inference-server.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{/*
Cache hostPath — defaults to /opt/cache/<model>-<device> if not explicitly set.
Sanitises model name: slashes and dots become dashes, lowercase.
*/}}
{{- define "tt-inference-server.cacheHostPath" -}}
{{- if .Values.cache.hostPath }}
{{- .Values.cache.hostPath }}
{{- else }}
{{- $model := .Values.model | lower | replace "/" "-" | replace "." "-" }}
{{- $device := .Values.device | lower }}
{{- printf "/opt/cache/%s-%s" $model $device }}
{{- end }}
{{- end }}

{{/*
Validate that required values are present.
Include this template in any resource that requires model and device to be set.
*/}}
{{- define "tt-inference-server.validateValues" -}}
{{- if not .Values.model }}
{{- fail "A model name is required. Pass --set model=<model-name> at helm install time." }}
{{- end }}
{{- if not .Values.device }}
{{- fail "A device is required. Pass --set device=<device-name> at helm install time." }}
{{- end }}
{{- if and (ne .Values.serverType "vllm") (ne .Values.serverType "media") }}
{{- fail "serverType must be either 'vllm' or 'media'." }}
{{- end }}
{{- end }}
