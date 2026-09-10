#!/bin/bash
# copy_media_env_from_quad1.sh -- seed OM Quad3's launch node with the media-server venv (and, opt-in, the converted DiT
# cache upper layer) from OM Quad1's gbh-e4-02, over the 100 GbE rack network. Run ON gbh-e4-02 (172.16.104.119) as zni:
#   copy_media_env_from_quad1.sh               media venv (3.6 GB) -> OM-F1-GBH01:$H3_MEDIA_ENV
#   copy_media_env_from_quad1.sh --with-cache  ...plus cache-upper (202 GB) -> OM-F1-GBH01:$H3_VM/cache-upper
#   H3_COPY_TARGET=<name|ip> ...               another destination (default OM-F1-GBH01 = 172.16.104.120)
#
# Why this shape (quad-agent h3-deploy/copy_to_quad2.sh): zni keys are per quad, so zni@gbh-e4-02 cannot ssh to a Quad3
# host. ubuntu's rack-wide ~/Resources/maas.pem can; it lives on root-squashed NFS, so root cannot read it -- the sender
# runs as ubuntu (sudo -u ubuntu) and needs /home/zni traversable for the duration (750 -> 751, restored at the end). The
# receiver runs rsync under sudo so /home/zni there is writable, and what was copied is chowned to the TARGET's zni afterwards
# (its uid differs from ours, never hardcode one). Only what was copied: a chown -R across $H3_VM would descend into a mounted
# cache-merged overlay, and overlayfs copies every lower file it touches up into cache-upper -- the lower is the 635 GB NFS cache.
# Why gbh-e4-02, not the Quad1 launch node gbh-e4-01: e4-01 has been DOWN since 2026-09-10 (UBB0 tray). e4-01/e4-02 are
# Quad1 machines either way; nothing here touches Quad3's chips or ours.
# What is copied: a relocatable uv venv (pyvenv.cfg relocatable = true, interpreter inside at _python/bin/python3.10) with
# bin/tt-run, bin/uvicorn, bin/ffmpeg. Its .pth files still point at /home/zni/tt-metal-h3 -- harmless on Quad3, the C12
# env file puts the pinned tree's ttnn first on PYTHONPATH. The cache upper was converted for metal 94cff3e426a; whether the
# pinned 34260b25483 pipeline reuses those entries is unverified -- a miss just re-converts, so --with-cache is optional.
set -u
D=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$D/env_om_quad3.sh"                      # H3_VM / H3_MEDIA_ENV = where things land on Quad3
TARGET=${H3_COPY_TARGET:-OM-F1-GBH01}            # MAAS DNS name of 172.16.104.120; ubuntu's ~/.ssh/config on Quad1 hosts lists it.
                                                 # H3_-prefixed knob: a stray TARGET in the shell must not redirect a root rsync + chown
SRC_ENV=/home/zni/h3-deploy/tt-inference-server/tt-media-server/python_env
SRC_CACHE=/home/zni/h3-deploy/cache-upper
WITH_CACHE=0
for a in "$@"; do case "$a" in --with-cache) WITH_CACHE=1 ;; *) echo "usage: $0 [--with-cache]"; exit 1 ;; esac; done
SSH="ssh -i /home/ubuntu/Resources/maas.pem -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -l ubuntu"

say() { echo "[h3 $(date -u +%H:%M:%S)] $*"; }
die() { say "ERROR: $*"; exit 2; }
as_ubuntu() { sudo -n -u ubuntu -H "$@"; }
remote() { as_ubuntu $SSH "$TARGET" "$@"; }       # a command on the target, as ubuntu

[ "$(id -un)" = zni ] || die "run as zni on gbh-e4-02 (this is $(id -un))"
[ -x "$SRC_ENV/bin/tt-run" ] || die "no media venv at $SRC_ENV -- is this gbh-e4-02?"
as_ubuntu test -r /home/ubuntu/Resources/maas.pem || die "ubuntu cannot read /home/ubuntu/Resources/maas.pem (NFS mounted? passwordless sudo?)"
remote true >/dev/null 2>&1 || die "cannot reach $TARGET as ubuntu with maas.pem"
remote id zni >/dev/null 2>&1 || die "zni does not exist on $TARGET -- run quad-agent scripts/provision-zni.sh --quad3 --trust first"
if [ "$WITH_CACHE" = 1 ]; then
  [ -d "$SRC_CACHE" ] || die "no $SRC_CACHE here"
  # never write into the upperdir of a mounted overlay
  remote "mountpoint -q '$H3_VM/cache-merged'" 2>/dev/null && die "$TARGET has the DiT overlay mounted at $H3_VM/cache-merged -- 'sudo umount' it there before seeding cache-upper"
fi

orig_mode=$(stat -c %a /home/zni)
chmod 751 /home/zni
trap 'chmod "$orig_mode" /home/zni' EXIT

copy() {  # <label> <src> <dst> [rsync extra...]: sender = ubuntu with maas.pem, receiver = sudo rsync, then chown at the end
  local label=$1 src=$2 dst=$3; shift 3
  say "$label: $src/ -> $TARGET:$dst/"
  remote "sudo mkdir -p '$dst'" || die "cannot create $dst on $TARGET"
  as_ubuntu rsync -a -x --info=stats2 "$@" -e "$SSH" --rsync-path="sudo rsync" "$src/" "$TARGET:$dst/" 2>&1 \
    | grep -E "Number of files|Total transferred file size|speedup|rsync error|failed" | sed "s/^/    /"
  [ "${PIPESTATUS[0]}" = 0 ] || die "rsync failed for: $label"
}
copy "media venv (relocatable uv venv, 3.6 GB)" "$SRC_ENV" "$H3_MEDIA_ENV"
[ "$WITH_CACHE" = 1 ] && copy "DiT cache upper layer (converted weights, 202 GB, optional seed)" "$SRC_CACHE" "$H3_VM/cache-upper"

say "chown to zni:zni on $TARGET what was copied (its zni uid differs from ours; never -R across $H3_VM, see header)"
remote "sudo chown zni:zni '$H3_VM' && sudo chown -R zni:zni '$H3_MEDIA_ENV'" || die "chown failed on $TARGET"
[ "$WITH_CACHE" = 1 ] && { remote "sudo chown -R zni:zni '$H3_VM/cache-upper'" || die "chown of cache-upper failed on $TARGET"; }
say "on $TARGET now: $(remote "ls -ld '$H3_VM' '$H3_MEDIA_ENV/bin/tt-run'" 2>&1 | tr '\n' ' ')"
say "COPY_DONE -- next: om_quad3_prep.sh on $TARGET as zni"
