#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bootstrap a Tenstorrent hardware runner into a cluster the chart can run on.
#
# The hardware smoke (.github/scripts/helm_hw_smoke.sh) assumes "tt-operator is
# installed and publishing devices". On the shared TT runner pool that is not a
# given: the runners are ephemeral KVM VMs with a card passed through and tt-kmd
# preloaded, and nothing else. This script turns one into that assumption:
#
#   RKE2 single-node cluster (DRA is GA in the pinned k8s) ->
#   tt-operator umbrella, NFD + tt-fabric-manager + tt-dra-driver only ->
#   devices published in a ResourceSlice
#
# tt-k8s-driver-manager is deliberately OFF: the runner already has tt-kmd
# loaded, and letting the manager reinstall it would add two DKMS builds and a
# device churn window to every run for nothing this smoke asserts. Same for
# tt-telemetry / jobset / kubepmix (kubepmix would also drag in cert-manager).
#
# THE PLUGIN RESTART, and why it is here rather than in the smoke: the DRA
# kubelet plugin discovers boards through tt-fabric-manager's GetTopology and
# publishes a ResourceSlice. If it comes up before TTFM has topology, it
# publishes a slice with no devices and does not refresh it for the life of the
# pod — claims then never allocate. In tt-operator's own integration-rke2 runs
# this is visible as `dra-smoke` failing on 40 of 42 job instances; the two that
# allocated are exactly the two where the plugin pod happened to (re)start ~45s
# after the driver was up. Restarting the DaemonSet once TTFM is Ready makes
# that ordering deliberate instead of lucky.
#
# Usage:  bash .github/scripts/hw_smoke_bootstrap.sh
#
# Knobs (all optional):
#   RKE2_VERSION           pinned RKE2 release (k8s >= 1.34 needed: DRA GA)
#   TT_OPERATOR_VERSION    umbrella chart version to install
#   OPERATOR_NS            namespace for tt-operator (default tt-operator-system)
#   GHCR_USERNAME / GHCR_TOKEN
#                          create a pull secret and wire it into the subcharts.
#                          Only needed where the container runtime cannot pull
#                          ghcr.io anonymously.
#   SKIP_CLUSTER=true      cluster + tt-operator are already there; only run the
#                          preflight and the device-publication wait
# ---------------------------------------------------------------------------
set -euo pipefail

RKE2_VERSION="${RKE2_VERSION:-v1.36.1+rke2r2}"
TT_OPERATOR_CHART="${TT_OPERATOR_CHART:-oci://ghcr.io/tenstorrent/helm/tt-operator}"
TT_OPERATOR_VERSION="${TT_OPERATOR_VERSION:-0.2.0}"
OPERATOR_NS="${OPERATOR_NS:-tt-operator-system}"
RELEASE="${TT_OPERATOR_RELEASE:-tt-operator}"
GHCR_USERNAME="${GHCR_USERNAME:-}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
PULL_SECRET="${PULL_SECRET:-tt-operator-image-pulltoken}"
SKIP_CLUSTER="${SKIP_CLUSTER:-false}"
SLICE_TIMEOUT="${SLICE_TIMEOUT:-300}"

log()  { printf '\n=== %s ===\n' "$*"; }
info() { printf '     %s\n' "$*"; }
ok()   { printf 'OK   %s\n' "$*"; }
warn() { printf '::warning::%s\n' "$*"; }
fail() { printf '::error::%s\n' "$*" >&2; exit 1; }

# Export a value to later workflow steps when running under Actions.
export_env() {
  info "$1=$2"
  [ -n "${GITHUB_ENV:-}" ] && echo "$1=$2" >> "$GITHUB_ENV"
  return 0
}

# ---------------------------------------------------------------------------
# Preflight — what this host actually is. Cheap, and it turns "the Pod stayed
# Pending" into a one-line answer later.
# ---------------------------------------------------------------------------
log "preflight: the runner"
[ -d /dev/tenstorrent ] || fail "no /dev/tenstorrent — this is not a Tenstorrent runner (or tt-kmd is not loaded)"
DEV_COUNT="$(find /dev/tenstorrent -maxdepth 1 -type c | wc -l | tr -d ' ')"
[ "$DEV_COUNT" -ge 1 ] || fail "/dev/tenstorrent exists but holds no device node"
info "device nodes: $DEV_COUNT"
if [ -r /sys/module/tenstorrent/version ]; then
  info "tt-kmd: $(cat /sys/module/tenstorrent/version)"
else
  warn "tt-kmd version unreadable (/sys/module/tenstorrent/version) — module not loaded via DKMS?"
fi

# IOMMU mode decides whether the boards need 1Gi hugepages at all: a device in a
# translating domain (DMA / DMA-FQ) does not, one in identity/passthrough does.
HUGEPAGES_NEEDED=true
for d in /sys/bus/pci/devices/*; do
  [ -r "$d/vendor" ] || continue
  [ "$(cat "$d/vendor")" = "0x1e52" ] || continue
  itype="$(cat "$d/iommu_group/type" 2>/dev/null || echo unknown)"
  info "$(basename "$d") iommu_group type=$itype"
  case "$itype" in DMA*) HUGEPAGES_NEEDED=false ;; esac
  break
done
NR_HP="$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo 0)"
FREE_HP="$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages 2>/dev/null || echo 0)"
info "1Gi hugepages: nr=$NR_HP free=$FREE_HP"
if [ "$HUGEPAGES_NEEDED" = "true" ] && [ "$NR_HP" = "0" ]; then
  warn "the boards are in an identity IOMMU domain (hugepages required) but the host has no 1Gi hugepages — the chart will be installed with hugepages off and the device may fail to open"
fi
# The smoke asks the node, not this script, but exporting it keeps the decision
# visible in the log and lets a caller override it per run.
if [ "$HUGEPAGES_NEEDED" = "true" ] && [ "$NR_HP" != "0" ]; then
  export_env HW_SMOKE_HUGEPAGES true
else
  export_env HW_SMOKE_HUGEPAGES false
fi

info "cpu(s)=$(nproc) memory=$(awk '/MemTotal/{printf "%.0fGi", $2/1048576}' /proc/meminfo)"
info "disk: $(df -h --output=avail / | tail -1 | tr -d ' ') available on /"
AVAIL_GB="$(df --output=avail -BG / | tail -1 | tr -dc '0-9')"
[ "${AVAIL_GB:-0}" -ge 60 ] \
  || warn "only ${AVAIL_GB}Gi free on / — an inference image plus weights and a compile cache usually needs more; a pull failure here is capacity, not the chart"

if [ "$SKIP_CLUSTER" = "true" ]; then
  info "SKIP_CLUSTER=true — leaving the cluster and tt-operator alone"
else

# ---------------------------------------------------------------------------
# RKE2. Mirrors tt-operator's setup-rke2-cluster action (same runner pool, same
# proxy traversal problem), minus cert-manager: nothing we install needs it.
# ---------------------------------------------------------------------------
log "install RKE2 $RKE2_VERSION"
curl -sfL https://get.rke2.io -o /tmp/rke2-install.sh
sudo -E INSTALL_RKE2_VERSION="$RKE2_VERSION" bash /tmp/rke2-install.sh

# Containerd inside RKE2 needs the proxy on its own process tree, or every
# in-cluster image pull (ghcr.io, registry.k8s.io) dies. The runner's default
# no_proxy covers none of the cluster CIDRs, so spell them out.
NPROXY="127.0.0.1,localhost,::1,10.0.0.0/8,10.42.0.0/16,10.43.0.0/16,.svc,.svc.cluster.local,${no_proxy:-}"
sudo mkdir -p /etc/systemd/system/rke2-server.service.d
sudo tee /etc/systemd/system/rke2-server.service.d/proxy.conf >/dev/null <<EOF
[Service]
Environment="HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy:-}}"
Environment="HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-}}"
Environment="NO_PROXY=${NPROXY}"
Environment="https_proxy=${https_proxy:-${HTTPS_PROXY:-}}"
Environment="http_proxy=${http_proxy:-${HTTP_PROXY:-}}"
Environment="no_proxy=${NPROXY}"
EOF
sudo systemctl daemon-reload
sudo systemctl enable rke2-server
# --no-block: systemd's Type=notify deadline is tighter than rke2's first start
# (etcd init + image load). Wait on the real ready signal, rke2.yaml on disk.
sudo systemctl start --no-block rke2-server
sudo timeout 600 bash -c 'until [ -s /etc/rancher/rke2/rke2.yaml ]; do sleep 5; done' \
  || fail "rke2-server never wrote a kubeconfig — check journalctl -u rke2-server"
mkdir -p "$HOME/.kube"
sudo install -m 0600 -o "$USER" -g "$USER" /etc/rancher/rke2/rke2.yaml "$HOME/.kube/config"
export KUBECONFIG="$HOME/.kube/config"
export_env KUBECONFIG "$HOME/.kube/config"
export NO_PROXY="$NPROXY" no_proxy="$NPROXY"
export_env NO_PROXY "$NPROXY"
export_env no_proxy "$NPROXY"

if ! command -v kubectl >/dev/null; then
  KCTL_VER="$(curl -sfL https://dl.k8s.io/release/stable.txt)"
  curl -sfL -o /tmp/kubectl "https://dl.k8s.io/release/${KCTL_VER}/bin/linux/amd64/kubectl"
  sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl
fi
command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
kubectl version --client
helm version --short

log "wait for the API server and the node"
timeout 300 bash -c 'until kubectl get --raw=/readyz >/dev/null 2>&1; do sleep 5; done' \
  || fail "API server never became ready"
# The node object can lag /readyz; `kubectl wait --all` errors with "no matching
# resources" in that window, so block until at least one Node exists first.
# shellcheck disable=SC2016  # the poll must evaluate inside the timeout'd shell
timeout 180 bash -c 'until [ "$(kubectl get nodes --no-headers 2>/dev/null | wc -l)" -gt 0 ]; do sleep 3; done'
kubectl wait --for=condition=Ready node --all --timeout=5m
kubectl get nodes -o wide
kubectl api-resources --api-group=resource.k8s.io 2>/dev/null | grep -q resourceslices \
  || fail "this cluster does not serve resource.k8s.io — DRA is not available (need k8s >= 1.34)"
ok "cluster up, resource.k8s.io served"

# ---------------------------------------------------------------------------
# tt-operator: the DRA layer the chart consumes.
# ---------------------------------------------------------------------------
log "install tt-operator $TT_OPERATOR_VERSION (NFD + tt-fabric-manager + tt-dra-driver)"
kubectl create namespace "$OPERATOR_NS" --dry-run=client -o yaml | kubectl apply -f -
OP_SET=(
  --set tt-k8s-driver-manager.enabled=false
  --set tt-telemetry.enabled=false
  --set jobset.enabled=false
  --set kubepmix.enabled=false
)
if [ -n "$GHCR_TOKEN" ]; then
  echo "$GHCR_TOKEN" | helm registry login ghcr.io --username "${GHCR_USERNAME:-x}" --password-stdin
  kubectl create secret docker-registry "$PULL_SECRET" -n "$OPERATOR_NS" \
    --docker-server=ghcr.io --docker-username="${GHCR_USERNAME:-x}" --docker-password="$GHCR_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -
  for sub in tt-fabric-manager tt-dra-driver; do
    OP_SET+=(--set "${sub}.imagePullSecrets[0].name=$PULL_SECRET")
  done
  info "pull secret $PULL_SECRET staged and wired into the subcharts"
else
  info "no GHCR_TOKEN given — relying on anonymous ghcr.io pulls"
fi
# No --wait: the DRA kubelet plugin crash-loops until the TTFM agent answers
# GetTopology, so "all resources Ready" is not a meaningful install gate here.
# The rollout waits below are, and they say which component is late.
helm install "$RELEASE" "$TT_OPERATOR_CHART" --version "$TT_OPERATOR_VERSION" \
  -n "$OPERATOR_NS" "${OP_SET[@]}"

fi  # SKIP_CLUSTER

# ---------------------------------------------------------------------------
# Devices published. See the header for why the plugin gets restarted.
# ---------------------------------------------------------------------------
log "wait for tt-fabric-manager, then republish the DRA slice"
# The DaemonSets are created by the install above, but NFD has to label the node
# before their pods are scheduled — poll for the objects rather than assume.
FM_DS=""; DRA_DS=""
for _ in $(seq 1 60); do
  FM_DS="$(kubectl -n "$OPERATOR_NS" get ds -o name 2>/dev/null | grep -m1 'fabric-manager-agent' || true)"
  DRA_DS="$(kubectl -n "$OPERATOR_NS" get ds -l app.kubernetes.io/name=tt-dra-driver -o name 2>/dev/null | head -1)"
  { [ -n "$FM_DS" ] && [ -n "$DRA_DS" ]; } && break
  sleep 5
done
[ -n "$FM_DS" ] || fail "no tt-fabric-manager agent DaemonSet in $OPERATOR_NS — tt-dra-driver would have no topology source"
[ -n "$DRA_DS" ] || fail "no tt-dra-driver DaemonSet in $OPERATOR_NS"
kubectl -n "$OPERATOR_NS" rollout status "$FM_DS" --timeout=10m
kubectl -n "$OPERATOR_NS" rollout status "$DRA_DS" --timeout=10m

info "restarting $DRA_DS so discovery runs with topology already up"
kubectl -n "$OPERATOR_NS" rollout restart "$DRA_DS"
kubectl -n "$OPERATOR_NS" rollout status "$DRA_DS" --timeout=10m

log "wait for boards in a ResourceSlice"
published=""
end=$(( $(date -u +%s) + SLICE_TIMEOUT ))
while [ "$(date -u +%s)" -lt "$end" ]; do
  published="$(kubectl get resourceslices \
    -o jsonpath='{range .items[*]}{range .spec.devices[*]}{.name}{"="}{.attributes.boardName.string}{" "}{end}{end}' 2>/dev/null || true)"
  [ -n "${published// /}" ] && break
  sleep 5
done
if [ -z "${published// /}" ]; then
  kubectl get resourceslices -o yaml || true
  kubectl -n "$OPERATOR_NS" logs "$DRA_DS" --tail=80 || true
  kubectl -n "$OPERATOR_NS" logs "$FM_DS" --tail=80 || true
  fail "no devices published within ${SLICE_TIMEOUT}s — tt-fabric-manager reported no topology on this runner, so no claim can allocate (environment, not the chart)"
fi
ok "boards published: $published"
kubectl get deviceclass
