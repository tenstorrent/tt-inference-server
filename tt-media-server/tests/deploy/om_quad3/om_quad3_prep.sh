#!/bin/bash
# om_quad3_prep.sh -- make OM Quad3 (racks F1/F2, 172.16.104.120-123) ready for tests/deploy/c12_quad/h3_deploy.sh.
# Run ON the launch node (172.16.104.120 = OM-F1-GBH01) as zni, after quad-agent scripts/provision-zni.sh --quad3 --trust
# gave zni an account, passwordless sudo and intra-quad ssh trust on all four hosts. Idempotent: every step checks before
# it acts, re-run freely. Touches no chip.
#
#   1. every host   $H3_VM + cache/video dirs; ffmpeg (absent on the F hosts); the DiT cache overlay (same command as
#                   quad-agent h3ctl.sh setup-cache); chips=32, no /dev/tenstorrent holders, weights readable, /dev/shm
#                   and / headroom, tt-smi + fuser on PATH; a reminder while rke2 is still active (k8s STOP gate)
#   2. this host    clone tt-metal + tt-inference-server into $H3_VM if absent (public https, remote 'origin'), copy
#                   env_om_quad3.sh + base_env_om.sh into $H3_VM, write the rankfile, check the media venv copied from
#                   gbh-e4-02 by copy_media_env_from_quad1.sh
#   3. print the h3_deploy.sh commands that come next
set -u
D=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$D/env_om_quad3.sh"
source "$D/base_env_om.sh"      # TT_DIT_CACHE_DIR / H3_DIT_CACHE_LOWER / TT_VIDEO_OUTPUT_DIR: the mount must be what the ranks use
METAL_URL=${H3_METAL_URL:-https://github.com/tenstorrent/tt-metal.git}
TIS_URL=${H3_TIS_URL:-https://github.com/tenstorrent/tt-inference-server.git}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

say() { echo "[h3 $(date -u +%H:%M:%S)] $*"; }
die() { say "ERROR: $*"; exit 2; }
hosts_list() { echo "$H3_HOSTS" | tr ',' ' '; }

case " $(hostname -I 2>/dev/null) " in
  *" $H3_RANK0 "*) ;;
  *) die "run this on the launch node $H3_RANK0 (OM-F1-GBH01); this is $(hostname) [$(hostname -I 2>/dev/null)]" ;;
esac
[ "$(id -un)" = zni ] || die "run as zni (this is $(id -un)); $H3_VM is zni's tree on every host"
fail=0

# ---------------------------------------------------------------------------------------------------- every host
# The payload travels base64 in the command string and runs under `bash -s`, so anything in it that might read stdin
# (apt-get) gets </dev/null -- otherwise it would swallow the rest of the script. No single quotes inside.
HOST_PAYLOAD='
set -u
VM=$1; CACHE=$2; LOWER=$3; VIDEOS=$4; WEIGHTS=$5
rc=0
ok()   { echo "  OK    $*"; }
bad()  { echo "  FAIL  $*"; rc=1; }
warn() { echo "  WARN  $*"; }
if mkdir -p "$VM/cache-upper" "$VM/cache-work" "$CACHE" "$VIDEOS" 2>/dev/null; then ok "$VM dirs (owner $(stat -c %U "$VM"))"
else bad "cannot create dirs under $VM (owner $(stat -c %U "$VM" 2>/dev/null || echo none)) -- sudo chown zni:zni $VM (the dir only; never chown -R across a mounted $CACHE)"; fi
# ffmpeg: not installed on the F hosts (Quad1/2 have /usr/bin/ffmpeg); probe needs ffmpeg + ffprobe, the rank-0 export may too
if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then ok "ffmpeg at $(command -v ffmpeg)"
else
  echo "  installing ffmpeg (apt)"
  if sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q ffmpeg </dev/null >/tmp/om_prep_apt.log 2>&1; then ok "ffmpeg installed"
  else bad "apt-get install ffmpeg failed -- see /tmp/om_prep_apt.log here; try sudo apt-get update first"; fi
fi
# overlayfs: prebuilt NFS cache below, a local writable layer on top. Same command on every host (h3ctl.sh setup-cache).
if mountpoint -q "$CACHE"; then ok "DiT cache overlay already mounted at $CACHE"
elif [ ! -d "$LOWER" ]; then bad "prebuilt DiT cache missing: $LOWER (NFS /data_bh mounted?)"
elif sudo -n mount -t overlay overlay -o "lowerdir=$LOWER,upperdir=$VM/cache-upper,workdir=$VM/cache-work" "$CACHE"; then ok "DiT cache overlay mounted at $CACHE"
else bad "overlay mount failed (needs passwordless sudo here)"; fi
# chips by PCI id (no pci.ids entry for the vendor, so grep the id, not the name)
chips=$(lspci 2>/dev/null | grep -ci "1e52\|tenstorrent")
[ "$chips" = 32 ] && ok "chips=32" || bad "chips=$chips, expected 32 (after a reset: FRB2/FRB3 tray hang -> BMC power cycle, do not re-reset)"
# root view (sudo): the holders this check is for (k8s/DRA/telemetry pods) are not owned by zni, and fuser silently skips fds it cannot read
holders=$(for f in /dev/tenstorrent/[0-9]*; do sudo -n fuser $f 2>/dev/null; done)
held=$(echo $holders | wc -w)
[ "$held" = 0 ] && ok "no /dev/tenstorrent holders" || bad "$held /dev/tenstorrent handle(s) held by $(ps -o comm= -p $(echo $holders | tr " " ,) 2>/dev/null | sort -u | tr "\n" " ")-- a k8s job (DRA) or a telemetry pod is on the chips"
[ -r "$WEIGHTS/model_index.json" ] && ok "weights readable at $WEIGHTS" || bad "$WEIGHTS/model_index.json not readable (NFS /data_bh mounted?)"
rootfree=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); shmfree=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
[ "${rootfree:-0}" -ge 150 ] && ok "/ free ${rootfree} GB" || warn "/ free ${rootfree:-?} GB (trees + build, media venv 3.6 GB, cache-upper up to ~80 GB on a cache miss, videos)"
[ "${shmfree:-0}" -ge 4 ] && ok "/dev/shm free ${shmfree} GB" || bad "/dev/shm free ${shmfree:-?} GB (SHM rings + side-files)"
command -v tt-smi >/dev/null && ok "tt-smi on PATH" || bad "tt-smi not on PATH for zni over ssh (h3_deploy.sh reset runs tt-smi -glx_reset_auto this way)"
command -v fuser >/dev/null || bad "fuser missing (h3_deploy.sh check/stop count device holders with it)"
# rke2 with the tt-operator DRA driver can hand these chips to a k8s job at any time: cordon the node or get the owner OK
if systemctl is-active --quiet rke2-server 2>/dev/null || systemctl is-active --quiet rke2-agent 2>/dev/null; then
  warn "rke2 ACTIVE on $(hostname) -- STOP gate: cordon the four F nodes or get the k8s owner OK before start/reset"
fi
exit $rc
'
b64=$(printf '%s' "$HOST_PAYLOAD" | base64 -w0)
for h in $(hosts_list); do
  say "--- $h"
  $SSH -n "$h" true 2>/dev/null || die "cannot ssh to $h as zni -- run quad-agent scripts/provision-zni.sh --quad3 --trust first"
  $SSH -n "$h" "echo $b64 | base64 -d | bash -s -- '$H3_VM' '$TT_DIT_CACHE_DIR' '$H3_DIT_CACHE_LOWER' '$TT_VIDEO_OUTPUT_DIR' '$H3_WEIGHTS'" || fail=1
done

# ---------------------------------------------------------------------------------------------------- launch node
clone_if_absent() {  # <url> <dir>: h3_deploy.sh needs "any clone with the tenstorrent remote as origin"; setup fetches the pinned refs itself
  local url=$1 dir=$2
  if [ -d "$dir/.git" ]; then
    git -C "$dir" remote get-url origin >/dev/null 2>&1 || die "$dir has no remote named origin (h3_deploy.sh fetches from origin)"
    say "ok: $dir exists (origin $(git -C "$dir" remote get-url origin))"; return
  fi
  say "cloning $url -> $dir (public https; takes a while)"
  git clone -q "$url" "$dir" || die "clone of $url failed"
}
say "--- launch node $H3_RANK0"
mkdir -p "$H3_VM" "$H3_WT" || die "cannot create $H3_WT"
clone_if_absent "$METAL_URL" "$H3_METAL_REPO"
clone_if_absent "$TIS_URL" "$H3_TIS_REPO"

for f in env_om_quad3.sh base_env_om.sh; do   # the env file sources $H3_VM/base_env_om.sh by absolute path on every rank
  if cmp -s "$D/$f" "$H3_VM/$f"; then say "ok: $H3_VM/$f up to date"
  else cp -f "$D/$f" "$H3_VM/$f" || die "cannot write $H3_VM/$f"; say "copied $f -> $H3_VM/"; fi
done

# rankfile: rank i = host i of H3_HOSTS, "slot=0:*" (the C12 format); verify_setup would generate the same when it is missing
want=$(i=0; for h in $(hosts_list); do echo "rank $i=$h slot=0:*"; i=$((i + 1)); done)
if [ -f "$H3_RANKFILE" ] && [ "$(cat "$H3_RANKFILE")" = "$want" ]; then say "ok: rankfile $H3_RANKFILE"
else mkdir -p "$(dirname "$H3_RANKFILE")"; printf '%s\n' "$want" > "$H3_RANKFILE" || die "cannot write $H3_RANKFILE"; say "wrote $H3_RANKFILE"; sed 's/^/    /' "$H3_RANKFILE"; fi

# media venv: copied from gbh-e4-02 (its zni has a different uid, so the copy script chowns; check that it did)
if [ -x "$H3_MEDIA_ENV/bin/tt-run" ] && [ -f "$H3_MEDIA_ENV/bin/activate" ]; then
  owner=$(stat -c %U "$H3_MEDIA_ENV")
  [ "$owner" = zni ] || die "$H3_MEDIA_ENV is owned by $owner -- run: sudo chown -R zni:zni $H3_MEDIA_ENV (that dir only; never chown -R across the mounted $TT_DIT_CACHE_DIR)"
  pyv=$("$H3_MEDIA_ENV/bin/python" -c 'import sys; print(sys.version.split()[0])' 2>&1) \
    || die "$H3_MEDIA_ENV/bin/python does not run ($pyv) -- it must be the relocatable venv from gbh-e4-02 (interpreter inside at _python/)"
  say "ok: media venv $H3_MEDIA_ENV (python $pyv, tt-run present)"
else
  say "MISSING: $H3_MEDIA_ENV/bin/tt-run -- on gbh-e4-02 as zni run tests/deploy/om_quad3/copy_media_env_from_quad1.sh [--with-cache], then re-run this"
  fail=1
fi

[ "$fail" = 0 ] || { say "PREP INCOMPLETE -- fix the FAIL/MISSING lines above and re-run"; exit 1; }
H3D=$(readlink -f "$D/../c12_quad/h3_deploy.sh" 2>/dev/null)
[ -n "$H3D" ] && [ -f "$H3D" ] || H3D='<checkout>/tt-media-server/tests/deploy/c12_quad/h3_deploy.sh'
say "PREP OK -- next, on this host as zni (one deployment per quad at a time; clear the k8s STOP gate first):"
cat <<EOT
  source $H3_VM/env_om_quad3.sh
  $H3D setup          # worktrees at the pins, device-tree build of 162a86b008a on the first run, env file, import smoke test
  $H3D replicate      # $H3_VM -> the three peers (no shared fs here); again after every setup/edit
  $H3D reset          # once, for the k8s residue on the chips (tt-smi -glx_reset_auto per host, then a 60 s settle)
  $H3D start fl2va    # then: probe fl2va 3 | status | check | logs | stop
EOT
