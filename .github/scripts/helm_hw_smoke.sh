#!/usr/bin/env bash
# Runs the chart on a Tenstorrent board (nightly / on-demand, see
# .github/workflows/helm-hw-smoke.yml). Needs a cluster where tt-operator is
# already publishing devices: hw_smoke_bootstrap.sh makes one out of a bare
# runner, or point KUBECONFIG at a prepared cluster and run this alone.
#
#   MODEL=Qwen3-Embedding-4B DEVICE=n150 bash .github/scripts/helm_hw_smoke.sh
#
# Knobs (all optional):
#   MODEL / DEVICE / ENGINE / IMPL  chart selection (ENGINE+IMPL are required for
#                                   IMAGE_TAG and the *_REQUEST overrides, which
#                                   address the values row by path)
#   NAMESPACE / RELEASE             where to install
#   IMAGE_TAG                       override the row's pinned tag, to tell "the
#                                   chart is broken" from "the pin is stale".
#                                   Never the steady state: the smoke guards
#                                   that pin.
#   HUGEPAGES                       true|false (default: probe the node)
#   HF_TOKEN                        gated weights only
#   HF_CACHE_DIR                    host dir with pre-downloaded weights
#   PULL_SECRET                     existing docker-registry secret to use
#   API_PROBE                       chat|embeddings|models (default: from image)
#   API_KEY                         bearer key to install via auth.apiKey and
#                                   then send. Default: fresh random per run.
#   INFER_MODEL_ID                  `model` for the inference call (default: the
#                                   id from /v1/models, HF cache path converted)
#   MAX_WAIT_SECONDS                Ready timeout (default 2700). Independent of
#                                   the chart's budget; both get reported.
#   CPU_REQUEST / MEMORY_REQUEST    shrink the row's requests on a small runner
set -euo pipefail

CHART="${CHART:-charts/tt-inference-server}"
MODEL="${MODEL:-Qwen3-Embedding-4B}"
DEVICE="${DEVICE:-n150}"
ENGINE="${ENGINE:-}"
IMPL="${IMPL:-}"
NAMESPACE="${NAMESPACE:-ttis-hw-smoke}"
RELEASE="${RELEASE:-ttis-hw}"
IMAGE_TAG="${IMAGE_TAG:-}"
HUGEPAGES="${HUGEPAGES:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
PULL_SECRET="${PULL_SECRET:-}"
API_PROBE="${API_PROBE:-}"
API_KEY="${API_KEY:-}"
INFER_MODEL_ID="${INFER_MODEL_ID:-}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-2700}"
CPU_REQUEST="${CPU_REQUEST:-}"
MEMORY_REQUEST="${MEMORY_REQUEST:-}"
LOCAL_PORT="${LOCAL_PORT:-18000}"
DEVICE_CLASS="${DEVICE_CLASS:-tenstorrent.com}"

log()  { printf '\n=== %s ===\n' "$*"; }
info() { printf '     %s\n' "$*"; }
ok()   { printf 'OK   %s\n' "$*"; }
warn() { printf '::warning::%s\n' "$*"; }
fail() { printf '::error::%s\n' "$*" >&2; diagnostics; exit 1; }

PF_PID=""
OVERRIDES=""
cleanup() {
  if [ -n "$PF_PID" ]; then
    kill "$PF_PID" 2>/dev/null || true
    PF_PID=""
  fi
  [ -n "$OVERRIDES" ] && rm -f "$OVERRIDES"
  return 0
}
trap cleanup EXIT

diagnostics() {
  printf '\n----- diagnostics -----\n'
  kubectl -n "$NAMESPACE" get pods -o wide 2>/dev/null || true
  kubectl -n "$NAMESPACE" describe pods 2>/dev/null | tail -80 || true
  kubectl -n "$NAMESPACE" get resourceclaim,resourceclaimtemplate -o yaml 2>/dev/null | tail -60 || true
  kubectl get resourceslices -o yaml 2>/dev/null | tail -80 || true
  kubectl -n "$NAMESPACE" logs -l app.kubernetes.io/instance="$RELEASE" \
    --tail=150 --all-containers --prefix 2>/dev/null || true
  kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp 2>/dev/null | tail -30 || true
}

secs_between() { echo $(( $(date -d "$2" +%s) - $(date -d "$1" +%s) )); }

log "chart selection"
HELM_SET=(--set "model=$MODEL" --set "device=$DEVICE")
[ -n "$ENGINE" ]       && HELM_SET+=(--set "engine=$ENGINE")
[ -n "$IMPL" ]         && HELM_SET+=(--set "impl=$IMPL")
[ -n "$HF_TOKEN" ]     && HELM_SET+=(--set "hfToken=$HF_TOKEN")
[ -n "$HF_CACHE_DIR" ] && HELM_SET+=(--set "hfCacheDir=$HF_CACHE_DIR")
[ -n "$PULL_SECRET" ]  && HELM_SET+=(--set "defaults.image.pullSecrets[0].name=$PULL_SECRET")

# A runner with no 1Gi pages would leave the Pod Pending unrelated to DRA.
if [ -z "$HUGEPAGES" ]; then
  hp_alloc="$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.hugepages-1Gi}' 2>/dev/null || true)"
  case "${hp_alloc:-0}" in ""|0|0Gi|0Ki) HUGEPAGES=false ;; *) HUGEPAGES=true ;; esac
  info "HUGEPAGES not set -> $HUGEPAGES (node allocatable hugepages-1Gi=${hp_alloc:-0})"
fi
HELM_SET+=(--set "hugepages.enabled=$HUGEPAGES")

# Random rather than fixed: a fixed key would still pass if the chart stopped
# wiring it and the server fell back to its built-in default.
if [ -z "$API_KEY" ]; then
  API_KEY="$(openssl rand -hex 16 2>/dev/null || head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  [ -n "${GITHUB_ACTIONS:-}" ] && printf '::add-mask::%s\n' "$API_KEY"
fi
HELM_SET+=(--set "auth.apiKey=$API_KEY")

# A values file, not --set: the path contains the model name, and model names
# contain dots that --set reads as separators.
if [ -n "$IMAGE_TAG" ] || [ -n "$CPU_REQUEST" ] || [ -n "$MEMORY_REQUEST" ]; then
  { [ -n "$ENGINE" ] && [ -n "$IMPL" ]; } \
    || fail "IMAGE_TAG / CPU_REQUEST / MEMORY_REQUEST address models.<model>.<engine>.<device>.impls.<impl>, so ENGINE and IMPL must be set explicitly"
  OVERRIDES="$(mktemp /tmp/hw-smoke-overrides.XXXXXX)"
  {
    echo "models:"
    echo "  \"$MODEL\":"
    echo "    \"$ENGINE\":"
    echo "      \"$DEVICE\":"
    echo "        impls:"
    echo "          \"$IMPL\":"
    if [ -n "$IMAGE_TAG" ]; then
      echo "            image:"
      echo "              tag: \"$IMAGE_TAG\""
    fi
    if [ -n "$CPU_REQUEST" ] || [ -n "$MEMORY_REQUEST" ]; then
      echo "            resources:"
      echo "              requests:"
      [ -n "$CPU_REQUEST" ]    && echo "                cpu: \"$CPU_REQUEST\""
      [ -n "$MEMORY_REQUEST" ] && echo "                memory: \"$MEMORY_REQUEST\""
    fi
  } > "$OVERRIDES"
  info "row overrides:"
  sed 's/^/       /' "$OVERRIDES"
  HELM_SET+=(-f "$OVERRIDES")
fi

RENDERED="$(helm template "$RELEASE" "$CHART" --namespace "$NAMESPACE" "${HELM_SET[@]}")" \
  || fail "helm template failed for model=$MODEL device=$DEVICE — does that model/engine/device/impl row exist?"
IMAGE_REF="$(printf '%s\n' "$RENDERED" | awk '$1=="image:" && $2 ~ /ghcr.io|docker.io|\//{print $2}' | grep -v busybox | head -1)"
WANT_BOARD="$(printf '%s\n' "$RENDERED" | awk -F'"' '/boardName ==/{print $(NF-1)}')"
WANT_COUNT="$(printf '%s\n' "$RENDERED" | awk '$1=="count:"{print $2; exit}')"
info "model=$MODEL device=$DEVICE image=$IMAGE_REF hugepages=$HUGEPAGES"

log "precondition: tt-operator's DRA layer is publishing devices"
kubectl get deviceclass "$DEVICE_CLASS" >/dev/null 2>&1 \
  || fail "DeviceClass $DEVICE_CLASS not found — tt-operator (tt-dra-driver) is not installed on this cluster"
[ -n "$WANT_COUNT" ] \
  || fail "the chart rendered no ResourceClaimTemplate for device=$DEVICE (a non-TT device?) — nothing for this smoke to assert"
info "the claim asks for $WANT_COUNT x boardName=$WANT_BOARD"
HAVE="$(kubectl get resourceslices \
  -o jsonpath='{range .items[*]}{range .spec.devices[*]}{.attributes.boardName.string}{"\n"}{end}{end}' 2>/dev/null \
  | grep -c "^${WANT_BOARD}$" || true)"
[ "${HAVE:-0}" -ge "$WANT_COUNT" ] \
  || fail "the DRA layer publishes ${HAVE:-0} board(s) of type $WANT_BOARD, the claim needs $WANT_COUNT — precondition unmet (fabric topology / tt-dra-driver), not a chart failure"
ok "$HAVE $WANT_BOARD board(s) published in ResourceSlices"

log "install $RELEASE into namespace $NAMESPACE"
helm install "$RELEASE" "$CHART" --namespace "$NAMESPACE" --create-namespace "${HELM_SET[@]}"
INSTALLED_AT="$(date -u +%s)"

POD=""
for _ in $(seq 1 60); do
  POD="$(kubectl -n "$NAMESPACE" get pod -l app.kubernetes.io/instance="$RELEASE" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$POD" ] && break
  sleep 2
done
[ -n "$POD" ] || fail "the Deployment never created a Pod"
info "pod=$POD"

log "the DRA claim allocates a board"
CLAIM=""
for _ in $(seq 1 60); do
  CLAIM="$(kubectl -n "$NAMESPACE" get resourceclaim -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$CLAIM" ] && break
  sleep 2
done
[ -n "$CLAIM" ] || fail "no ResourceClaim was generated from the chart's ResourceClaimTemplate"
DEVICES=""
for _ in $(seq 1 60); do
  DEVICES="$(kubectl -n "$NAMESPACE" get resourceclaim "$CLAIM" \
    -o jsonpath='{range .status.allocation.devices.results[*]}{.device}{" "}{end}' 2>/dev/null || true)"
  [ -n "${DEVICES// /}" ] && break
  sleep 5
done
[ -n "${DEVICES// /}" ] \
  || fail "ResourceClaim $CLAIM never allocated — the scheduler found no free $WANT_BOARD board for this Pod"
ok "claim $CLAIM allocated: $DEVICES"

priv="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.containers[0].securityContext.privileged}')"
[ "${priv:-false}" != "true" ] || fail "the container runs privileged — the DRA path should not need it"
hostpaths="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{range .spec.volumes[*]}{.hostPath.path}{"\n"}{end}' | grep -c '^/dev/tenstorrent' || true)"
[ "${hostpaths:-0}" -eq 0 ] || fail "the Pod mounts /dev/tenstorrent by hostPath — DRA should be injecting it instead"
ok "no privileged container, no /dev/tenstorrent hostPath"

log "the container starts and sees its board"
state=""
for _ in $(seq 1 180); do
  state="$(kubectl -n "$NAMESPACE" get pod "$POD" \
    -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null || true)"
  case "$state" in *running*) break ;; esac
  [ "$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.phase}')" = "Failed" ] \
    && fail "the Pod reached phase Failed before the container started"
  sleep 10
done
case "$state" in
  *running*) ;;
  *) fail "the container never started within 30m (image pull or init container stuck): ${state:-<none>}" ;;
esac
STARTED_AT="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.status.containerStatuses[0].state.running.startedAt}')"
info "container started at $STARTED_AT"
injected="$(kubectl -n "$NAMESPACE" exec "$POD" -c inference-server -- \
  sh -c 'find /dev/tenstorrent -maxdepth 1 -type c | wc -l' 2>/dev/null | tr -d '[:space:]')"
[ "${injected:-0}" -eq "$WANT_COUNT" ] \
  || fail "the container sees ${injected:-0} /dev/tenstorrent device node(s), the claim allocated $WANT_COUNT"
kubectl -n "$NAMESPACE" exec "$POD" -c inference-server -- ls -l /dev/tenstorrent || true
ok "$injected /dev/tenstorrent device node(s) injected into the real inference container"

log "Ready inside the compile window"
ft="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.containers[0].startupProbe.failureThreshold}')"
ps="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.containers[0].startupProbe.periodSeconds}')"
{ [ -n "$ft" ] && [ -n "$ps" ] && [ "$ft" -gt 0 ] && [ "$ps" -gt 0 ]; } \
  || fail "no usable startupProbe on the container — the compile window is unguarded"
BUDGET=$(( ft * ps ))
info "chart budget = failureThreshold($ft) x periodSeconds($ps) = ${BUDGET}s; this smoke waits up to ${MAX_WAIT_SECONDS}s"
deadline=$(( $(date -u +%s) + MAX_WAIT_SECONDS ))
ready=""
while [ "$(date -u +%s)" -lt "$deadline" ]; do
  ready="$(kubectl -n "$NAMESPACE" get pod "$POD" \
    -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || true)"
  [ "$ready" = "True" ] && break
  restarts="$(kubectl -n "$NAMESPACE" get pod "$POD" \
    -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null || echo 0)"
  [ "${restarts:-0}" -gt 0 ] \
    && fail "the container restarted ${restarts}x before becoming Ready — the startupProbe budget (${BUDGET}s) expired mid-compile, or the server crashed"
  sleep 15
done
[ "$ready" = "True" ] \
  || fail "the Pod did not become Ready within ${MAX_WAIT_SECONDS}s (the chart's own budget is ${BUDGET}s) — compile is slower than either, or /health never answered"
READY_AT="$(kubectl -n "$NAMESPACE" get pod "$POD" \
  -o jsonpath='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}')"
COMPILE="$(secs_between "$STARTED_AT" "$READY_AT")"
TOTAL=$(( $(date -d "$READY_AT" +%s) - INSTALLED_AT ))
restarts="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].restartCount}')"
[ "${restarts:-0}" -eq 0 ] || fail "the Pod is Ready but its container restarted ${restarts}x on the way"
PCT=$(( COMPILE * 100 / BUDGET ))
ok "Ready after ${COMPILE}s of compile/warmup (${TOTAL}s from helm install), 0 restarts — ${PCT}% of the ${BUDGET}s budget"
[ "$PCT" -lt 80 ] \
  || warn "compile used ${PCT}% of the startupProbe budget (${COMPILE}s of ${BUDGET}s) — the window is close to too small for this row"

log "the served API answers"
SVC="$(kubectl -n "$NAMESPACE" get svc -l app.kubernetes.io/instance="$RELEASE" \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$NAMESPACE" port-forward "svc/$SVC" "${LOCAL_PORT}:8000" >/tmp/hw-smoke-pf.log 2>&1 &
PF_PID=$!
for _ in $(seq 1 30); do
  curl -sf -o /dev/null "http://127.0.0.1:${LOCAL_PORT}/health" && break
  sleep 2
done

code="$(curl -s -o /tmp/hw-smoke-health.json -w '%{http_code}' "http://127.0.0.1:${LOCAL_PORT}/health")"
[ "$code" = "200" ] \
  || fail "/health returned $code (port-forward: $(tr '\n' ' ' < /tmp/hw-smoke-pf.log | tail -c 200))"
ok "/health 200"

AUTH=()
[ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")

# vLLM guards /v1/models once VLLM_API_KEY is set, so send the key here too.
code="$(curl -s -o /tmp/hw-smoke-models.json -w '%{http_code}' \
  "${AUTH[@]+"${AUTH[@]}"}" "http://127.0.0.1:${LOCAL_PORT}/v1/models")"
[ "$code" = "200" ] || fail "/v1/models returned $code"
SERVED_ID="$(python3 -c 'import json;d=json.load(open("/tmp/hw-smoke-models.json"));print((d.get("data") or [{}])[0].get("id",""))')"
[ -n "$SERVED_ID" ] \
  || fail "/v1/models returned 200 but listed no model: $(head -c 300 /tmp/hw-smoke-models.json)"
ok "/v1/models 200, serving id=$SERVED_ID"

# The media server advertises settings.model_weights_path (an HF snapshot dir)
# but its runner only accepts the HF repo id, so posting the advertised id back
# gets "Model <path> is not supported by <Runner>". Convert the path.
if [ -z "$INFER_MODEL_ID" ]; then
  case "$SERVED_ID" in
    *models--*/snapshots/*)
      repo="${SERVED_ID#*models--}"
      repo="${repo%%/snapshots/*}"
      INFER_MODEL_ID="$(printf '%s' "$repo" | sed 's/--/\//')"
      info "the served id is an HF cache path -> using repo id $INFER_MODEL_ID for the inference call"
      ;;
    *)
      INFER_MODEL_ID="$SERVED_ID"
      ;;
  esac
fi

if [ -z "$API_PROBE" ]; then
  case "$IMAGE_REF" in
    *vllm-tt-metal-src*) API_PROBE=chat ;;
    *)                   API_PROBE=embeddings ;;
  esac
  info "API_PROBE not set -> $API_PROBE (from the resolved image)"
fi
case "$API_PROBE" in
  chat)
    INFER_PATH=/v1/chat/completions
    PAYLOAD="{\"model\":\"${INFER_MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":16}"
    ;;
  embeddings)
    INFER_PATH=/v1/embeddings
    PAYLOAD="{\"model\":\"${INFER_MODEL_ID}\",\"input\":\"hello from the hardware smoke\"}"
    ;;
  models)
    INFER_PATH=""
    info "API_PROBE=models — inference call skipped by request"
    ;;
  *)
    fail "unknown API_PROBE=$API_PROBE (want chat|embeddings|models)"
    ;;
esac

if [ -n "$INFER_PATH" ]; then
  # Negative first: without this, "200 with the key" would also pass on a
  # server that ignores the key entirely.
  if [ -n "$API_KEY" ]; then
    code="$(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
      -d "$PAYLOAD" "http://127.0.0.1:${LOCAL_PORT}${INFER_PATH}")"
    [ "$code" = "401" ] \
      || fail "$INFER_PATH answered $code without a key while auth.apiKey was installed — the key is not being enforced (expected 401)"
    ok "$INFER_PATH without a key -> 401 (auth.apiKey is enforced)"
  fi

  code="$(curl -s -o /tmp/hw-smoke-infer.json -w '%{http_code}' \
    -H 'Content-Type: application/json' "${AUTH[@]+"${AUTH[@]}"}" \
    -d "$PAYLOAD" "http://127.0.0.1:${LOCAL_PORT}${INFER_PATH}")"
  [ "$code" = "200" ] \
    || fail "$INFER_PATH returned $code: $(head -c 300 /tmp/hw-smoke-infer.json)"

  case "$API_PROBE" in
    chat)
      python3 -c 'import json,sys;d=json.load(open("/tmp/hw-smoke-infer.json"));sys.exit(0 if (d["choices"][0]["message"].get("content") or "").strip() else 1)' \
        || fail "$INFER_PATH returned 200 but generated no content: $(head -c 300 /tmp/hw-smoke-infer.json)"
      ok "$INFER_PATH 200 with generated content"
      ;;
    embeddings)
      dims="$(python3 -c 'import json;d=json.load(open("/tmp/hw-smoke-infer.json"));print(len((d.get("data") or [{}])[0].get("embedding") or []))')"
      [ "${dims:-0}" -gt 0 ] \
        || fail "$INFER_PATH returned 200 but an empty vector: $(head -c 300 /tmp/hw-smoke-infer.json)"
      ok "$INFER_PATH 200 with a ${dims}-dimension vector"
      ;;
  esac
fi
kill "$PF_PID" 2>/dev/null || true
PF_PID=""

# 4. Uninstall releases the board. `helm uninstall --wait` returning is not
# proof: if the DRA plugin is wedged the kubelet cannot finish
# NodeUnprepareResources and the Pod sits in Terminating still holding it.
log "uninstall releases the board"
helm uninstall "$RELEASE" --namespace "$NAMESPACE" --wait
gone=""
for _ in $(seq 1 60); do
  left="$(kubectl -n "$NAMESPACE" get pod -l app.kubernetes.io/instance="$RELEASE" -o name 2>/dev/null | wc -l | tr -d ' ')"
  [ "$left" = "0" ] && { gone=yes; break; }
  sleep 5
done
[ -n "$gone" ] \
  || fail "the Pod was still present 300s after uninstall — stuck Terminating means the board is still claimed: $(kubectl -n "$NAMESPACE" get pod -l app.kubernetes.io/instance="$RELEASE" -o wide | tail -2)"
for kind in resourceclaim resourceclaimtemplate; do
  left=""
  for _ in $(seq 1 30); do
    left="$(kubectl -n "$NAMESPACE" get "$kind" -o name 2>/dev/null | wc -l | tr -d ' ')"
    [ "$left" = "0" ] && break
    sleep 2
  done
  [ "$left" = "0" ] || fail "$left $kind left in $NAMESPACE after uninstall"
done
ok "Pod and claim gone, board released"

kubectl delete namespace "$NAMESPACE" --wait=false >/dev/null 2>&1 || true

log "hardware smoke PASSED"
printf 'model=%s device=%s image=%s compile=%ss budget=%ss (%s%%)\n' \
  "$MODEL" "$DEVICE" "$IMAGE_REF" "$COMPILE" "$BUDGET" "$PCT"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Hardware smoke passed"
    echo ""
    echo "| | |"
    echo "|---|---|"
    echo "| model / device | \`$MODEL\` / \`$DEVICE\` |"
    echo "| image | \`$IMAGE_REF\` |"
    echo "| boards allocated | \`$DEVICES\` (asked for $WANT_COUNT x $WANT_BOARD) |"
    echo "| hugepages | \`$HUGEPAGES\` |"
    echo "| compile to Ready | ${COMPILE}s of the chart's ${BUDGET}s startupProbe budget (${PCT}%) |"
    echo "| served id | \`$SERVED_ID\` |"
    echo "| inference call | \`$API_PROBE\` as \`$INFER_MODEL_ID\`$([ -n "$API_KEY" ] && echo ", API key sent") |"
  } >> "$GITHUB_STEP_SUMMARY"
fi
