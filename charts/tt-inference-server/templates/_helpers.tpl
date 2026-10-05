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

{{/*
media/forge fall back to a well-known built-in key when API_KEY is unset, which
looks authenticated but is not, so make the operator pick. vLLM needs no gate:
it is open unless VLLM_API_KEY is set, which is what auth.apiKey sets.
*/}}
{{- $auth := .Values.auth | default dict }}
{{- if and (dig "apiKey" "" $auth) (dig "disabled" false $auth) }}
  {{- fail "auth.apiKey and auth.disabled are contradictory: the server would honour NO_AUTH and serve unauthenticated while the key sits in the release Secret. Set one." }}
{{- end }}
{{- if or (eq $engine "media") (eq $engine "forge") }}
  {{- if and (not (dig "apiKey" "" $auth)) (not (dig "disabled" false $auth)) }}
    {{- fail (printf "engine '%s' authenticates its inference routes with a bearer key, and leaving it unset would silently use the server image's built-in default. Set auth.apiKey=<key> (stored in the release Secret as API_KEY), or auth.disabled=true to run without auth." $engine) }}
  {{- end }}
{{- end }}

{{- include "tt-inference-server.validateImageFloor" . }}
{{- end }}

{{/*
The chart drives the server through interfaces that only exist in images from a
certain release onwards, so a row pinned below that release cannot serve and is
refused here rather than at runtime:

  - vllm: the chart passes --model/--tt-device as container args. Images before
    0.11.0 start the server from CMD, which those args replace, and they leave
    CACHE_ROOT unset, which their entrypoint requires — verified on hardware:
    the container crash-loops in its entrypoint, and with CACHE_ROOT supplied it
    then fails on `exec: "--model"`.
  - media/forge: auth.disabled maps to NO_AUTH, which the server only honours
    from 0.15.0. Below that the flag does nothing: the server still demands a
    bearer key and falls back to a built-in default, so clients that send no
    header are rejected — not the unauthenticated serving that was asked for.

Rows whose tag carries no release prefix (forge commit-hash builds) can't be
placed against a floor, so they pass.
*/}}
{{- define "tt-inference-server.validateImageFloor" -}}
{{- if dig "enforce" true (.Values.imageFloor | default dict) }}
{{- $engine := include "tt-inference-server.resolvedEngine" . }}
{{- $tag := include "tt-inference-server.imageTag" . }}
{{- $release := regexFind "^[0-9]+\\.[0-9]+\\.[0-9]+" $tag }}
{{- if $release }}
{{- $floors := .Values.imageFloor | default dict }}
{{- if eq $engine "vllm" }}
  {{- $floor := dig "vllm" "" $floors }}
  {{- if and $floor (not (semverCompare (printf ">=%s" $floor) $release)) }}
    {{- fail (printf "model '%s' on device '%s' is pinned to image tag '%s' (release %s), below this chart's floor of %s for the vllm engine. The chart passes --model/--tt-device as container args, but images before 0.11.0 use an ENTRYPOINT that does not forward them (and do not set CACHE_ROOT, which that same entrypoint requires), so the container dies in its entrypoint and the server never starts. Choose a model/device whose pinned image is 0.11.0 or newer, or have the pin refreshed in the ModelSpec catalogue. Set imageFloor.enforce=false only to reproduce this failure deliberately." .Values.model .Values.device $tag $release $floor) }}
  {{- end }}
{{- else if dig "disabled" false (.Values.auth | default dict) }}
  {{- $floor := dig "authDisabled" "" $floors }}
  {{- if and $floor (not (semverCompare (printf ">=%s" $floor) $release)) }}
    {{- fail (printf "auth.disabled=true requires NO_AUTH, which the '%s' server only honours from release %s, but model '%s' on device '%s' is pinned to image tag '%s' (release %s). That server ignores NO_AUTH and still requires Authorization: Bearer <API_KEY>, falling back to a key published in its own source when API_KEY is unset, so the release would not serve unauthenticated as asked: clients sending no header are rejected. Set auth.apiKey=<key> instead, or use a newer pin. Set imageFloor.enforce=false only to reproduce this failure deliberately." $engine $floor .Values.model .Values.device $tag $release) }}
  {{- end }}
{{- end }}
{{- end }}
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
{{- printf "%s:%s" $cfg.image.repository (include "tt-inference-server.imageTag" .) }}
{{- end }}

{{/*
Resolved image tag, and the value of app.kubernetes.io/version. One release runs
one pinned image, so the tag is the only value that names what is actually
serving; the chart itself has no single app version to report, since the
catalogue's rows span many product releases.
*/}}
{{- define "tt-inference-server.imageTag" -}}
{{- $cfg := include "tt-inference-server.resolvedConfig" . | fromYaml }}
{{- $cfg.image.tag | toString }}
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
hugepages-1Gi request/limit: hugepages.size when set, else one 1Gi page per ASIC
of the device (deviceChipCounts). Empty for a device with no chip count (gpu/cpu,
or a shape the map does not cover), which drops the request entirely.
*/}}
{{- define "tt-inference-server.hugepagesSize" -}}
{{- $override := dig "size" "" (.Values.hugepages | default dict) -}}
{{- if $override -}}
{{- $override -}}
{{- else -}}
{{- $chips := index (.Values.deviceChipCounts | default dict) (.Values.device | lower) -}}
{{- if $chips -}}{{- printf "%dGi" (int $chips) -}}{{- end -}}
{{- end -}}
{{- end -}}

{{- define "tt-inference-server.cacheRoot" -}}
/home/container_app_user/cache_root
{{- end -}}

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
DRA board count for .Values.device, read from the generated
.Values.deviceBoardCounts map (single source of truth). Returns:
  - device in the map  -> its board count
  - gpu/cpu (non-TT)   -> "" (no ResourceClaim)
  - anything else      -> fail (e.g. galaxy_t3k; unsupported, fail closed)
*/}}
{{- define "tt-inference-server.draDeviceCount" -}}
{{- $counts := .Values.deviceBoardCounts | default dict -}}
{{- $nonTT := list "gpu" "cpu" -}}
{{- $d := .Values.device | lower -}}
{{- if hasKey $counts $d -}}
{{- index $counts $d -}}
{{- else if has $d $nonTT -}}
{{- else -}}
{{- fail (printf "device '%s' is not supported via DRA in this chart. Topology-sensitive partitions (e.g. galaxy_t3k) are future work and need the tt-dra-driver to publish topology attributes. Supported: %s." $d (keys $counts | sortAlpha | join ", ")) -}}
{{- end -}}
{{- end -}}

{{/*
DRA boardName for .Values.device, from the generated .Values.deviceBoardNames map.
The ResourceClaim selects boards by this attribute (CEL). Returns "" for non-TT
devices (gpu/cpu); only used when draDeviceCount is non-empty.
*/}}
{{- define "tt-inference-server.draBoardName" -}}
{{- $names := .Values.deviceBoardNames | default dict -}}
{{- index $names (.Values.device | lower) | default "" -}}
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
app.kubernetes.io/version: {{ include "tt-inference-server.imageTag" . | quote }}
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
