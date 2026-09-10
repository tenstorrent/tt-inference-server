#!/bin/bash
# quad3_deploy.sh -- one-command MiniMax-H3 deployment on OM Quad3 (racks F1/F2, 172.16.104.120-123) with the C12 recipe.
# Runs ON the launch node 172.16.104.120 (OM-F1-GBH01) as zni; every step is ../c12_quad/h3_deploy.sh with the H3_*
# overrides from env_om_quad3.sh (sourced here, so nothing has to be exported by hand) plus the OM-only preparation.
#
#   quad3_deploy.sh all <t2va|fl2va|ref2va> [n]   preflight -> prep -> setup -> replicate -> reset (once) -> start -> probe n (default 3)
#   quad3_deploy.sh preflight                     read-only: launch node, zni, ssh trust, chips, k8s gate, media venv, ffmpeg
#   quad3_deploy.sh prep                          om_quad3_prep.sh (dirs, ffmpeg, DiT overlay, clones, env files, rankfile)
#   quad3_deploy.sh setup | replicate | reset | start <task> | wait-ready [s] | probe <task> [n] | status | check | logs [n] | stop
#                                                 -> h3_deploy.sh with the Quad3 overrides
#   quad3_deploy.sh k8s-status | k8s-cordon | k8s-uncordon   rke2 kubectl on this node for the four F nodes (never run by `all`)
#
# Gates (the script cannot decide these for you):
#   H3_K8S_OK=1        required by reset/start/all: you cordoned the four F nodes (k8s-cordon) or have the k8s owner's OK.
#                      rke2 + the tt-operator DRA driver are live on the F hosts and can hand chips to a job at any time.
#   H3_SKIP_RESET=1    `all` skips the one-time tt-smi -glx_reset_auto (only if the chips were reset since the last k8s use).
#   H3_ALLOW_HOLDERS=1 `all`/reset proceed although /dev/tenstorrent handles are held on some host (default: stop).
# One deployment per quad at a time. Never tt-smi -r on these Galaxy hosts. Logs: $H3_WT/deploy_logs/quad3_deploy.<ts>.log
set -u
D=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$D/env_om_quad3.sh"
H3D=$D/../c12_quad/h3_deploy.sh
PREP=$D/om_quad3_prep.sh
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
KUBECTL="sudo -n /var/lib/rancher/rke2/bin/kubectl --kubeconfig /etc/rancher/rke2/rke2.yaml"
NODE_RE='om-f[12]-gbh0[12]'                      # rke2 node names are the lower-cased hostnames (kube-apiserver-om-f1-gbh01 ...)

say() { echo "[quad3 $(date -u +%H:%M:%S)] $*"; }
die() { say "ERROR: $*"; exit 2; }
hosts_list() { echo "$H3_HOSTS" | tr ',' ' '; }
step() { say "==== $*"; }

on_launch_node() { case " $(hostname -I 2>/dev/null) " in *" $H3_RANK0 "*) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------------------------------------------- preflight
preflight() {  # read-only; returns 1 on any FAIL. Chip-touching actions also need H3_K8S_OK=1 (checked by the caller).
  local rc=0 h out
  ok()  { echo "  OK    $*"; }
  bad() { echo "  FAIL  $*"; rc=1; }
  wrn() { echo "  WARN  $*"; }
  on_launch_node && ok "on the launch node $H3_RANK0 ($(hostname))" || bad "not on the launch node $H3_RANK0 -- this is $(hostname) [$(hostname -I 2>/dev/null)]"
  [ "$(id -un)" = zni ] && ok "running as zni" || bad "running as $(id -un), must be zni (scripts/provision-zni.sh --quad3 --trust from the quad-agent box)"
  [ -f "$H3D" ] && ok "h3_deploy.sh at $H3D" || bad "missing $H3D (checkout must carry tests/deploy/c12_quad)"
  grep -q 'H3_BASE_ENV' "$H3D" 2>/dev/null && ok "h3_deploy.sh has the H3_BASE_ENV/H3_RESET_MODE/H3_SHARED_FS hooks" \
    || bad "this h3_deploy.sh predates the OM hooks (no H3_BASE_ENV) -- env_om_quad3.sh relies on them"
  for h in $(hosts_list); do
    out=$($SSH -n "$h" 'echo "$(id -un)@$(hostname) chips=$(lspci 2>/dev/null | grep -ci "1e52\|tenstorrent") held=$(for f in /dev/tenstorrent/[0-9]*; do sudo -n fuser $f 2>/dev/null; done | wc -w) rke2=$( (systemctl is-active --quiet rke2-server || systemctl is-active --quiet rke2-agent) 2>/dev/null && echo active || echo inactive) ffmpeg=$(command -v ffmpeg >/dev/null && echo yes || echo no) overlay=$(mountpoint -q '"$TT_DIT_CACHE_DIR"' && echo mounted || echo no)"' 2>&1) \
      || { bad "$h: ssh as zni failed ($out) -- intra-quad trust: provision-zni.sh --quad3 --trust"; continue; }
    case "$out" in
      zni@*chips=32*) ok "$h: $out" ;;
      zni@*) bad "$h: $out (expected chips=32; <32 after a reset = FRB2/FRB3 tray hang -> BMC power cycle, do not re-reset)" ;;
      *) bad "$h: landed as the wrong user: $out" ;;
    esac
    case "$out" in *"held=0"*) ;; *) wrn "$h holds /dev/tenstorrent handles -- a k8s job or a previous deployment is on the chips" ;; esac
    case "$out" in *"rke2=active"*) wrn "$h: rke2 ACTIVE -- k8s gate: cordon the four F nodes (k8s-cordon) or get the owner's OK; export H3_K8S_OK=1 when done" ;; esac
  done
  if [ -x "$H3_MEDIA_ENV/bin/tt-run" ]; then ok "media venv at $H3_MEDIA_ENV"
  else bad "media venv missing at $H3_MEDIA_ENV -- on gbh-e4-02 as zni: tests/deploy/om_quad3/copy_media_env_from_quad1.sh"; fi
  [ -r /data_bh/h3/model_index.json ] && ok "weights readable at /data_bh/h3" || bad "/data_bh/h3/model_index.json not readable (NFS /data_bh mounted?)"
  [ "${H3_K8S_OK:-0}" = 1 ] && ok "H3_K8S_OK=1 (k8s gate acknowledged)" || wrn "H3_K8S_OK not set: reset/start/all will refuse until you export H3_K8S_OK=1"
  return $rc
}

require_k8s_ok() { [ "${H3_K8S_OK:-0}" = 1 ] || die "k8s gate: cordon the four F nodes (quad3_deploy.sh k8s-cordon) or get the k8s owner's OK, then export H3_K8S_OK=1"; }

holders_total() { local n=0 h c; for h in $(hosts_list); do c=$($SSH -n "$h" 'for f in /dev/tenstorrent/[0-9]*; do sudo -n fuser $f 2>/dev/null; done | wc -w' 2>/dev/null || echo 0); n=$((n + c)); done; echo "$n"; }

# ---------------------------------------------------------------------------------------------------- k8s helpers (explicit only)
k8s_nodes() { $KUBECTL get nodes -o name 2>/dev/null | sed 's|^node/||' | grep -E "$NODE_RE"; }
k8s_status()  { on_launch_node || die "kubectl is only tried on the rke2 server node $H3_RANK0"; $KUBECTL get nodes -o wide || die "kubectl failed here (the .118 join server is down; ask the k8s owner)"; }
k8s_cordon()  {
  on_launch_node || die "run on $H3_RANK0"
  local nodes; nodes=$(k8s_nodes); [ -n "$nodes" ] || die "no nodes matching $NODE_RE (kubectl unavailable or names differ: quad3_deploy.sh k8s-status)"
  [ "$(echo "$nodes" | wc -l)" = 4 ] || die "expected 4 F nodes, got: $(echo $nodes) -- cordon by hand after checking k8s-status"
  say "cordoning: $(echo $nodes)   (blocks NEW scheduling only; daemonsets such as the DRA driver keep running -- whether that keeps the chips free is unverified)"
  # shellcheck disable=SC2086
  $KUBECTL cordon $nodes || die "cordon failed"
  $KUBECTL get nodes | grep -E "$NODE_RE"
  say "export H3_K8S_OK=1 to proceed; remember k8s-uncordon when handing Quad3 back"
}
k8s_uncordon() { on_launch_node || die "run on $H3_RANK0"; local nodes; nodes=$(k8s_nodes); [ -n "$nodes" ] || die "no F nodes found"; $KUBECTL uncordon $nodes && $KUBECTL get nodes | grep -E "$NODE_RE"; }

# ---------------------------------------------------------------------------------------------------- delegation
h3() { "$H3D" "$@"; }   # env_om_quad3.sh is already sourced: H3_VM/H3_HOSTS/H3_RESET_MODE/... are in the environment

run_all() {
  local task=${1:?task} n=${2:-3} ts log
  case "$task" in t2va|fl2va|ref2va) ;; *) die "task must be t2va|fl2va|ref2va" ;; esac
  mkdir -p "$H3_WT/deploy_logs" || die "cannot create $H3_WT/deploy_logs"
  ts=$(date -u +%Y%m%d_%H%M%S); log=$H3_WT/deploy_logs/quad3_deploy.$ts.log
  exec > >(tee -a "$log") 2>&1
  say "quad3_deploy.sh all $task $n   log: $log"
  step "preflight";  preflight || die "preflight FAILED -- fix the FAIL lines (media venv / zni / ssh trust / chips) and re-run"
  require_k8s_ok
  step "prep (om_quad3_prep.sh)"; bash "$PREP" || die "prep failed"
  step "setup (worktrees at the pins, device-tree build on the first run, env file, import smoke test)"; h3 setup || die "setup failed"
  step "replicate ($H3_VM -> the other three hosts)"; h3 replicate || die "replicate failed"
  if [ "${H3_SKIP_RESET:-0}" = 1 ]; then say "H3_SKIP_RESET=1: skipping the one-time reset"
  else
    local held; held=$(holders_total)
    if [ "$held" != 0 ] && [ "${H3_ALLOW_HOLDERS:-0}" != 1 ]; then
      die "$held /dev/tenstorrent handle(s) held across the quad (k8s job or old deployment) -- stop it first, or H3_ALLOW_HOLDERS=1 to reset anyway"
    fi
    step "reset once (k8s residue on the chips): tt-smi -glx_reset_auto on all four, then 60 s settle"
    h3 reset || die "reset failed -- rc != 0 usually means an FRB2/FRB3 tray hang: check chips per host, BMC power cycle that host, do NOT re-reset"
  fi
  step "start $task"; h3 start "$task" || die "start $task failed -- $H3D logs 80"
  step "probe $task x$n"; h3 probe "$task" "$n" || die "probe reported a failure -- see $H3_WT/deploy_logs/probe/"
  step "done"; h3 status
  say "ALL DONE: $task serving at http://$H3_RANK0:8000   (stop: $0 stop; hand back: $0 stop && $0 k8s-uncordon)"
}

case ${1:-} in
  all)          run_all "${2:?task}" "${3:-3}" ;;
  preflight)    preflight ;;
  prep)         bash "$PREP" ;;
  setup)        h3 setup ;;
  replicate)    h3 replicate ;;
  reset)        require_k8s_ok; h3 reset ;;
  start)        require_k8s_ok; h3 start "${2:?task}" "${3:-3600}" ;;
  wait-ready)   h3 wait-ready "${2:-3600}" ;;
  probe)        h3 probe "${2:?task}" "${3:-3}" ;;
  status|check|stop) h3 "$1" ;;
  logs)         h3 logs "${2:-40}" ;;
  k8s-status)   k8s_status ;;
  k8s-cordon)   k8s_cordon ;;
  k8s-uncordon) k8s_uncordon ;;
  *) sed -n 2,19p "$0"; exit 1 ;;
esac
