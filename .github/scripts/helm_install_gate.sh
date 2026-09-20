#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install gate — hardware-free live-install checks for the tt-inference-server
# chart on a kind cluster. Sibling of the helm-chart-validate render job: that
# job proves the manifests render, this one proves what render cannot — a *real*
# API server accepts the DRA ResourceClaimTemplate (CEL selector and all), the
# Pod becomes schedulable and stays Pending *only* because no Tenstorrent device
# exists, and the CRD-gated / toggled resources actually apply against a live
# control plane.
#
# NOT re-tested here: that the DRA driver can hand a Pod /dev/tenstorrent — that
# is tt-operator's dra-smoke, on real silicon. We assume a working DRA layer and
# stub the one piece of it the scheduler needs (the DeviceClass); the absence of
# real devices (ResourceSlices) is exactly what keeps the Pod Pending.
#
# Preconditions (set up by the caller / workflow):
#   - kubectl, helm, and jq on PATH; KUBECONFIG pointing at a kind cluster whose
#     API server serves resource.k8s.io/v1 (Kubernetes >= 1.34, where DRA is GA).
#   - network access to fetch the Prometheus Operator PodMonitor CRD.
# ---------------------------------------------------------------------------
set -euo pipefail

CHART="${CHART:-charts/tt-inference-server}"
MODEL="${MODEL:-gemma-3-1b-it}"     # small single-board model — the shape is what matters, not the weights
DEVICE="${DEVICE:-n150}"            # 1 board => DRA count 1
DEVICE_CLASS="${DEVICE_CLASS:-tenstorrent.com}"
# Pinned so the gate can't drift when upstream retags. Only the PodMonitor CRD
# schema is needed, nothing else from the Prometheus Operator.
PM_CRD_URL="${PM_CRD_URL:-https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.76.0/example/prometheus-operator-crd/monitoring.coreos.com_podmonitors.yaml}"

log()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf '::error::%s\n' "$*" >&2; exit 1; }

# helm install with the common required values; extra --set args passed through.
install() {
  local release="$1" ns="$2"; shift 2
  helm install "$release" "$CHART" \
    --namespace "$ns" --create-namespace \
    --set model="$MODEL" --set device="$DEVICE" --set hfToken=hf_dummy \
    "$@"
}

# Poll until the Deployment's Pod exists; echo its name (fail after ~60s).
wait_for_pod() {
  local ns="$1" release="$2" pod=""
  for _ in $(seq 1 30); do
    pod="$(kubectl -n "$ns" get pod -l app.kubernetes.io/instance="$release" -o name 2>/dev/null | head -1)"
    [ -n "$pod" ] && { echo "$pod"; return 0; }
    sleep 2
  done
  return 1
}

# ---------------------------------------------------------------------------
log "precheck: API server serves resource.k8s.io/v1 (DRA GA, k8s >= 1.34)"
command -v jq >/dev/null || fail "jq is required but not on PATH"
kubectl api-resources --api-group=resource.k8s.io 2>/dev/null | grep -q resourceclaimtemplates \
  || fail "resource.k8s.io/v1 not served — need a kindest/node:v1.34+ cluster"
echo "cluster server version: $(kubectl version -o json 2>/dev/null | jq -r '.serverVersion.gitVersion // "unknown"')"

# A DeviceClass is the one piece of the DRA layer the scheduler resolves before
# it will even create the claim. tt-dra-driver installs the real one; we stub it
# (no devices) so the claim path runs and stalls purely on device absence.
log "setup: stub DeviceClass '$DEVICE_CLASS' (working DRA layer, minus silicon)"
kubectl apply -f - <<EOF
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: ${DEVICE_CLASS}
spec: {}
EOF

# ---------------------------------------------------------------------------
# Core check — the DRA claim path only a live install can exercise. Install with
# hugepages OFF so DRA is the sole scheduling constraint (a kind node has no 1Gi
# hugepages; leaving them on would mask the DRA gate). This install therefore
# doubles as the hugepages-off toggle check.
# ---------------------------------------------------------------------------
CORE_NS=ttis-core
log "install (hugepages off): DRA claim accepted, Pod schedulable-but-Pending"
install ttis "$CORE_NS" --set hugepages.enabled=false

# The real API server accepted the ResourceClaimTemplate (schema + CEL selector).
rct="$(kubectl -n "$CORE_NS" get resourceclaimtemplate \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[ -n "$rct" ] || fail "API server did not accept a ResourceClaimTemplate"
cls="$(kubectl -n "$CORE_NS" get resourceclaimtemplate "$rct" \
  -o jsonpath='{.spec.spec.devices.requests[0].exactly.deviceClassName}')"
cnt="$(kubectl -n "$CORE_NS" get resourceclaimtemplate "$rct" \
  -o jsonpath='{.spec.spec.devices.requests[0].exactly.count}')"
[ "$cls" = "$DEVICE_CLASS" ] || fail "ResourceClaimTemplate deviceClassName=$cls, want $DEVICE_CLASS"
[ "$cnt" = "1" ]             || fail "ResourceClaimTemplate count=$cnt, want 1 (n150 = 1 board)"
echo "OK  ResourceClaimTemplate $rct accepted (deviceClassName=$cls count=$cnt)"

# The control plane auto-creates a ResourceClaim from the template and the
# scheduler leaves it unallocated (no ResourceSlice => no device); the Pod stays
# Pending. Unallocated-claim + Pending is exactly "no device present is fine".
POD="$(wait_for_pod "$CORE_NS" ttis)" || fail "Deployment never created a Pod"
alloc=""
for _ in $(seq 1 30); do
  rc="$(kubectl -n "$CORE_NS" get resourceclaim -o name 2>/dev/null | head -1)"
  [ -n "$rc" ] && { alloc="$(kubectl -n "$CORE_NS" get "$rc" -o jsonpath='{.status.allocation}')"; break; }
  sleep 2
done
[ -n "${rc:-}" ]   || fail "no ResourceClaim was created from the template"
[ -z "$alloc" ]    || fail "ResourceClaim unexpectedly allocated on a device-less cluster"
phase="$(kubectl -n "$CORE_NS" get "$POD" -o jsonpath='{.status.phase}')"
[ "$phase" = "Pending" ] || fail "Pod phase=$phase, want Pending (accepted, awaiting device)"
echo "OK  Pod Pending, ResourceClaim ${rc##*/} created and unallocated (awaiting device)"

hp_res="$(kubectl -n "$CORE_NS" get "$POD" -o jsonpath='{.spec.containers[0].resources}' | grep -c hugepages || true)"
hp_vol="$(kubectl -n "$CORE_NS" get "$POD" -o json | jq '[.spec.volumes[].name] | index("hugepages-1g")')"
hp_ini="$(kubectl -n "$CORE_NS" get "$POD" -o json | jq '(.spec.initContainers // []) | map(.name) | index("cleanup-hugepages")')"
if [ "$hp_res" != "0" ] || [ "$hp_vol" != "null" ] || [ "$hp_ini" != "null" ]; then
  fail "hugepages.enabled=false but a hugepages request/volume/initContainer remains"
fi
echo "OK  hugepages off: no request, no /dev/hugepages-1G volume, no cleanup initContainer"

# ---------------------------------------------------------------------------
# hugepages ON (the chart default) wires the request, the volume, and the
# cleanup initContainer. The Pod won't schedule on kind (no hugepages), which is
# fine: we assert the rendered-and-accepted spec, not readiness.
# ---------------------------------------------------------------------------
HP_NS=ttis-hugepages
log "install (hugepages on, default): request + volume + cleanup initContainer wired"
install ttis-hp "$HP_NS"
POD_HP="$(wait_for_pod "$HP_NS" ttis-hp)" || fail "hugepages-on install created no Pod"
hp_res="$(kubectl -n "$HP_NS" get "$POD_HP" -o jsonpath='{.spec.containers[0].resources}' | grep -c hugepages || true)"
hp_vol="$(kubectl -n "$HP_NS" get "$POD_HP" -o json | jq '[.spec.volumes[].name] | index("hugepages-1g")')"
hp_ini="$(kubectl -n "$HP_NS" get "$POD_HP" -o json | jq '(.spec.initContainers // []) | map(.name) | index("cleanup-hugepages")')"
if [ "$hp_res" -lt 1 ] || [ "$hp_vol" = "null" ] || [ "$hp_ini" = "null" ]; then
  fail "hugepages.enabled=true (default) but request/volume/initContainer missing"
fi
echo "OK  hugepages on: request + volume + cleanup initContainer present"
helm uninstall ttis-hp --namespace "$HP_NS" >/dev/null

# ---------------------------------------------------------------------------
# PodMonitor is CRD-gated. The chart documents: disabled by default because a
# PodMonitor needs the Prometheus Operator CRD, and installing one without that
# CRD fails. Only a live API server enforces "unknown kind => reject", which is
# why render cannot cover any of the three cases below.
# ---------------------------------------------------------------------------
PM_NS=ttis-podmonitor

log "podMonitor disabled (default): install succeeds, no PodMonitor object"
install ttis-pm "$PM_NS" --set hugepages.enabled=false >/dev/null
[ -z "$(kubectl -n "$PM_NS" get podmonitors.monitoring.coreos.com -o name 2>/dev/null)" ] \
  || fail "podMonitor disabled but a PodMonitor object exists"
helm uninstall ttis-pm --namespace "$PM_NS" >/dev/null
echo "OK  podMonitor disabled: no PodMonitor object"

log "podMonitor enabled without the CRD: install must fail"
if install ttis-pm "$PM_NS" --set hugepages.enabled=false --set podMonitor.enabled=true >/dev/null 2>&1; then
  helm uninstall ttis-pm --namespace "$PM_NS" >/dev/null 2>&1 || true
  fail "podMonitor enabled without the CRD but install succeeded (should be rejected)"
fi
helm uninstall ttis-pm --namespace "$PM_NS" >/dev/null 2>&1 || true
echo "OK  podMonitor enabled without CRD: install rejected as expected"

log "podMonitor enabled with the CRD: install succeeds, PodMonitor object created"
kubectl apply --server-side -f "$PM_CRD_URL" >/dev/null
kubectl wait --for=condition=established --timeout=60s crd/podmonitors.monitoring.coreos.com >/dev/null
install ttis-pm "$PM_NS" --set hugepages.enabled=false --set podMonitor.enabled=true >/dev/null
[ -n "$(kubectl -n "$PM_NS" get podmonitors.monitoring.coreos.com -o name 2>/dev/null)" ] \
  || fail "podMonitor enabled with the CRD but no PodMonitor object was created"
echo "OK  podMonitor enabled with CRD: PodMonitor object created"

# ---------------------------------------------------------------------------
# --wait blocks on the release's own objects, but the auto-created ResourceClaim
# is owned by the Pod and garbage-collected only after the Pod is gone — so poll
# each kind to drain rather than assume a fixed delay suffices on a slow runner.
# ---------------------------------------------------------------------------
log "uninstall is clean (no leftover ResourceClaimTemplate / ResourceClaim / Pod)"
helm uninstall ttis-pm --namespace "$PM_NS" --wait >/dev/null
helm uninstall ttis --namespace "$CORE_NS" --wait >/dev/null
for ns in "$CORE_NS" "$PM_NS"; do
  for kind in resourceclaimtemplate resourceclaim pod; do
    left=""
    for _ in $(seq 1 30); do
      left="$(kubectl -n "$ns" get "$kind" -o name 2>/dev/null | wc -l | tr -d ' ')"
      [ "$left" = "0" ] && break
      sleep 2
    done
    [ "$left" = "0" ] || fail "$left $kind left in $ns after uninstall"
  done
done
echo "OK  uninstall clean: no leftover ResourceClaimTemplate / ResourceClaim / Pod"

log "install gate PASSED"
