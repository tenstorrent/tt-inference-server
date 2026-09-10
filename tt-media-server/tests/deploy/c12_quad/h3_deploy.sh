#!/bin/bash
# h3_deploy.sh -- deploy MiniMax-H3 (t2va | fl2va | ref2va) on the C12 4x32 quad with the configuration verified clean
# on 2026-09-10 (see H3_fl2va_C12_deploy_guide.md and H3_C12_findings_2026-09-10.md next to this script).
#
#   h3_deploy.sh setup                 create/verify the three pinned worktrees, build the device tree if needed,
#                                      wire symlinks, cherry-pick the SP fix, patch dit_runners, write the env file
#   h3_deploy.sh start <task>          stop whatever runs, launch the 4 ranks + the SP frontend, wait until ready
#   h3_deploy.sh wait-ready [secs]     block until the deployment answers (default 3600 s)
#   h3_deploy.sh probe <task> [n]      send n (default 3) identical requests and judge the mp4s (needs ffmpeg)
#   h3_deploy.sh status | check        liveness / which hosts still hold the devices
#   h3_deploy.sh stop                  stop frontend + ranks and wait for the devices to be released
#   h3_deploy.sh reset                 stop, then recover.sh (tt-smi reset + link validation, up to 3 attempts)
#   h3_deploy.sh logs                  tail the merged rank log
#
# Pinned versions (do not change without re-validating):
#   device side  tt-metal 162a86b008a   (.so, kernels, firmware, runtime)  -- newer builds carry LLK 1b17275b8df (noise)
#   python side  tt-metal 34260b25483   (models/tt_dit, ttnn python)       -- 333841bbeb0+ corrupts warm requests
#   server       tt-inference-server 78d516584 + 70a756282 (SP side-file fix) + create_pipeline knob patch
# Env overrides: H3_VM, H3_WT (worktree root), H3_METAL_REPO, H3_TIS_REPO, H3_MEDIA_ENV, H3_API_KEY, H3_SKIP_BUILD=1,
#                H3_FORCE_CHECKOUT=1, H3_FORCE_ENV=1, H3_HOSTS, H3_RANK0, H3_RANKFILE, H3_DESC_DIR, H3_DESC_PREFIX (another quad)
set -u

VM=${H3_VM:-/data/DC-deploy/vision-models}
WT=${H3_WT:-$VM/zni_worktrees}
METAL_SHARED=${H3_METAL_REPO:-$VM/tt-metal}                 # any tt-metal clone with the tenstorrent remote as 'origin'
TIS_SHARED=${H3_TIS_REPO:-$VM/tt-inference-server}          # any tt-inference-server clone
DEVICE_COMMIT=162a86b008a
# 162a86b008a is the PRE-rebase "add H3 bucketing" (amended 2026-09-04 17:22 UTC on the C12 checkout, never on the
# branch that was pushed). It is published as tenstorrent/tt-metal branch zni/h3-c12-clean-device-tree; the public
# post-rebase commit with the same message (193c08b2944) is NOT equivalent -- it sits on main 09-04 incl. LLK 1b17275b8df.
DEVICE_COMMIT_REF=zni/h3-c12-clean-device-tree
PY_COMMIT_REF=sadesoye/H3_rebase_merge_optimizations
SERVER_COMMIT_REF=sadesoye/add_h3_fl2va_ref2va
PY_COMMIT=34260b25483
SERVER_COMMIT=78d516584
SP_FIX_COMMIT=70a756282
DEV=$WT/tt-metal-old
PY=$WT/tt-metal-0909
TMS_ROOT=$WT/tms-0909
TMS=$TMS_ROOT/tt-media-server
MEDIA_ENV=${H3_MEDIA_ENV:-$TIS_SHARED/tt-media-server/python_env}   # media-server venv (uvicorn, tt-run, mpi4py, torch, PIL)
ENVF=$WT/env_c12_0909.sh
HOSTS=${H3_HOSTS:-bh-glx-EXP-c01u21,bh-glx-EXP-c01u14,bh-glx-EXP-c02u07,bh-glx-EXP-c01u07}
RANK0=${H3_RANK0:-bh-glx-EXP-c01u21}
RANKFILE=${H3_RANKFILE:-$VM/test_C12_rankfile}          # rank i = host i of HOSTS, "slot=0:*"
RANKBIND=$DEV/tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml
DESC_DIR=${H3_DESC_DIR:-/data/scaleout_configs/bh_glx_exabox}   # recover.sh cabling/deployment descriptors
DESC_PREFIX=${H3_DESC_PREFIX:-C12}
URL=http://$RANK0:8000
KEY=${H3_API_KEY:-your-secret-key}
LOGDIR=$WT/deploy_logs
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
export PATH="$VM/ffmpeg:/opt/openmpi-v5.0.7-ulfm/bin:$PATH"
mkdir -p "$LOGDIR"

say() { echo "[h3 $(date -u +%H:%M:%S)] $*"; }
die() { say "ERROR: $*"; exit 2; }
hosts_list() { echo "$HOSTS" | tr ',' ' '; }

# ---------------------------------------------------------------------------------------------------- setup
ensure_commit() {  # <repo> <commit> <remote ref>: fetch the ref over https (gh credential) if the commit is not local
  local repo=$1 commit=$2 ref=$3
  git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null && return 0
  say "$commit not in $repo -- fetching origin/$ref"
  git -C "$repo" -c url."https://github.com/".insteadOf="git@github.com:" fetch origin "$ref" >/dev/null 2>&1 || die "fetch of $ref failed in $repo"
  git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null || die "$commit still missing after fetching $ref"
}

worktree_at() {  # <repo> <dir> <commit> [<remote ref containing it>]
  local repo=$1 dir=$2 commit=$3 ref=${4:-}
  [ -n "$ref" ] && ensure_commit "$repo" "$commit" "$ref"
  if [ ! -d "$dir/.git" ] && [ ! -f "$dir/.git" ]; then
    say "creating worktree $dir @ $commit"
    git -C "$repo" worktree add --detach "$dir" "$commit" >/dev/null || die "worktree add failed for $dir"
  fi
  local head; head=$(git -C "$dir" rev-parse --short=11 HEAD)
  if [ "$head" != "$(git -C "$repo" rev-parse --short=11 "$commit")" ]; then
    if [ "${H3_FORCE_CHECKOUT:-0}" = 1 ]; then
      say "moving $dir from $head to $commit (H3_FORCE_CHECKOUT=1)"
      git -C "$dir" stash push -q -m "h3_deploy: parked $(date -u +%FT%TZ)" 2>/dev/null || true
      git -C "$dir" checkout -q --detach "$commit" || die "checkout $commit failed in $dir"
    else
      die "$dir is at $head, expected $commit -- fix by hand or rerun with H3_FORCE_CHECKOUT=1"
    fi
  fi
  say "ok: $dir @ $(git -C "$dir" log -1 --format='%h %s' | cut -c1-70)"
}

setup_device_tree() {
  worktree_at "$METAL_SHARED" "$DEV" "$DEVICE_COMMIT" "$DEVICE_COMMIT_REF"
  [ -e "$DEV/.cpmcache" ] || ln -s "$METAL_SHARED/.cpmcache" "$DEV/.cpmcache"
  [ -e "$DEV/python_env" ] || ln -s "$METAL_SHARED/python_env" "$DEV/python_env"
  for s in umd tracy tt-cluster-descriptors; do
    [ -n "$(ls -A "$DEV/tt_metal/third_party/$s" 2>/dev/null)" ] || (cd "$DEV" && git submodule update --init "tt_metal/third_party/$s" >/dev/null 2>&1)
  done
  if [ ! -f "$DEV/build_Release/lib/libtt_metal.so" ] || [ ! -f "$DEV/ttnn/ttnn/_ttnn.so" ] || [ ! -d "$DEV/runtime/sfpi" ]; then
    [ "${H3_SKIP_BUILD:-0}" = 1 ] && die "device tree not built ($DEV/build_Release) and H3_SKIP_BUILD=1"
    say "building the device tree at $DEVICE_COMMIT (takes a while; log: $LOGDIR/build_device_tree.log)"
    (cd "$DEV" && ./build_metal.sh > "$LOGDIR/build_device_tree.log" 2>&1) || die "build_metal.sh failed, see $LOGDIR/build_device_tree.log"
  fi
  [ -e "$DEV/build" ] || ln -s build_Release "$DEV/build"
  say "ok: device libs $(ls -la --time-style=+%F_%T "$DEV/build_Release/lib/libtt_metal.so" | awk '{print $6}'), sfpi $(grep -oE '7\.[0-9]+\.[0-9]+' "$DEV/runtime/sfpi-version.cmake" | head -1)"
}

setup_python_tree() {
  worktree_at "$METAL_SHARED" "$PY" "$PY_COMMIT" "$PY_COMMIT_REF"
  cd "$PY" || die "no $PY"
  ln -sfn "$DEV/build_Release" build
  ln -sfn "$METAL_SHARED/runtime" runtime
  ln -sfn "$METAL_SHARED/internal-prodia" internal-prodia
  ln -sfn "$METAL_SHARED/python_env" python_env
  ln -sfn "$DEV/ttnn/ttnn/_ttnn.so" ttnn/ttnn/_ttnn.so
  for s in umd tracy tt-cluster-descriptors; do
    [ -L "tt_metal/third_party/$s" ] || { rmdir "tt_metal/third_party/$s" 2>/dev/null; ln -sfn "$DEV/tt_metal/third_party/$s" "tt_metal/third_party/$s"; }
  done
  say "ok: python tree wired (build -> $(readlink build), _ttnn.so -> $(readlink ttnn/ttnn/_ttnn.so))"
}

setup_server_tree() {
  ensure_commit "$TIS_SHARED" "$SERVER_COMMIT" "$SERVER_COMMIT_REF"; ensure_commit "$TIS_SHARED" "$SP_FIX_COMMIT" "$SERVER_COMMIT_REF"
  if [ ! -d "$TMS_ROOT/.git" ] && [ ! -f "$TMS_ROOT/.git" ]; then
    say "creating worktree $TMS_ROOT @ $SERVER_COMMIT"
    git -C "$TIS_SHARED" worktree add --detach "$TMS_ROOT" "$SERVER_COMMIT" >/dev/null || die "worktree add failed for $TMS_ROOT"
  fi
  cd "$TMS_ROOT" || die "no $TMS_ROOT"
  local base fix; base=$(git -C "$TIS_SHARED" rev-parse "$SERVER_COMMIT"); fix=$(git -C "$TIS_SHARED" rev-parse "$SP_FIX_COMMIT")
  # accepted states: exactly SERVER_COMMIT (then cherry-pick), or SERVER_COMMIT + the cherry-picked SP fix on top
  if [ "$(git rev-parse HEAD)" = "$base" ]; then
    say "cherry-picking $SP_FIX_COMMIT (carry duration/aspect through the SP side file)"
    git cherry-pick -x "$SP_FIX_COMMIT" >/dev/null || die "cherry-pick $SP_FIX_COMMIT failed (resolve in $TMS_ROOT)"
  elif [ "$(git rev-parse HEAD~1)" = "$base" ] && git log -1 --format=%b HEAD | grep -q "cherry picked from commit $fix"; then
    :
  elif [ "${H3_FORCE_CHECKOUT:-0}" = 1 ]; then
    say "moving $TMS_ROOT to $SERVER_COMMIT (H3_FORCE_CHECKOUT=1)"
    git stash push -q -m "h3_deploy: parked $(date -u +%FT%TZ)" 2>/dev/null || true
    git checkout -q --detach "$SERVER_COMMIT" || die "checkout failed"; git cherry-pick -x "$SP_FIX_COMMIT" >/dev/null || die "cherry-pick failed"
  else
    die "$TMS_ROOT is at $(git rev-parse --short HEAD), expected $SERVER_COMMIT (+ cherry-picked $SP_FIX_COMMIT) -- rerun with H3_FORCE_CHECKOUT=1"
  fi
  ln -sfn "$MEDIA_ENV" "$TMS/python_env"
  "$MEDIA_ENV/bin/python" - "$TMS/tt_model_runners/dit_runners.py" <<'PY' || die "dit_runners patch failed"
import sys
p = sys.argv[1]; s = open(p).read()
if "accepted = inspect.signature(MiniMaxH3Pipeline.create_pipeline).parameters" in s:
    print("ok: dit_runners already patched"); sys.exit(0)
old = '''            return MiniMaxH3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                weights_dir=self._weights_dir(),
                task=self.pipeline_task,
                dit_fsdp=self.dit_fsdp,
                trace_denoise=_minimax_h3_env_bool("MINIMAX_H3_TRACE_DENOISE"),
                bucket_denoise=_minimax_h3_env_bool("MINIMAX_H3_BUCKET_DENOISE"),
            )'''
new = '''            import inspect

            # h3_deploy: pass newer pipeline knobs only when this pipeline version accepts them.
            accepted = inspect.signature(MiniMaxH3Pipeline.create_pipeline).parameters
            extra = {}
            if "warmup" in accepted:
                # construction-time warmup (metal 56cdeeb9095) corrupts the first served request: keep it off
                extra["warmup"] = os.environ.get("MINIMAX_H3_CONSTRUCTION_WARMUP", "1") != "0"
            if "vae_output_type" in accepted:
                # yuv420 = device-stitched decode (333841bbeb0+, corrupts warm requests); float = host stitch
                extra["vae_output_type"] = os.environ.get("MINIMAX_H3_VAE_OUTPUT", "yuv420")
            return MiniMaxH3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                weights_dir=self._weights_dir(),
                task=self.pipeline_task,
                dit_fsdp=self.dit_fsdp,
                trace_denoise=_minimax_h3_env_bool("MINIMAX_H3_TRACE_DENOISE"),
                bucket_denoise=_minimax_h3_env_bool("MINIMAX_H3_BUCKET_DENOISE"),
                **extra,
            )'''
if s.count(old) != 1:
    print("cannot find the create_pipeline block to patch (found %d)" % s.count(old)); sys.exit(1)
open(p, "w").write(s.replace(old, new)); print("ok: dit_runners patched")
PY
  say "ok: server tree $(git -C "$TMS_ROOT" log -1 --format='%h %s' | cut -c1-70)"
}

write_env() {
  if [ -f "$ENVF" ] && [ "${H3_FORCE_ENV:-0}" != 1 ]; then say "ok: env file exists ($ENVF); H3_FORCE_ENV=1 rewrites it"; return; fi
  cat > "$ENVF" <<EOT
# MiniMax-H3 C12 quad -- verified-clean configuration (generated by h3_deploy.sh $(date -u +%F))
source $VM/metal_env_H3.sh
export ZNI_METAL_PY=$PY            # models/tt_dit + ttnn python  @ $PY_COMMIT
export ZNI_METAL_BIN=$DEV          # .so, kernels, firmware, runtime @ $DEVICE_COMMIT
export TT_METAL_HOME="\$ZNI_METAL_BIN"
export TT_METAL_RUNTIME_ROOT="\$ZNI_METAL_BIN"
export LD_LIBRARY_PATH="\$ZNI_METAL_BIN/build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PYTHONPATH="\$ZNI_METAL_PY/ttnn:\$ZNI_METAL_PY/:\$ZNI_METAL_PY/internal-prodia/"
unset TT_METAL_CACHE                       # per-host ~/.cache/tt-metal-cache; never a shared NFS dir
export MINIMAX_H3_TRACE_DENOISE=0
export MINIMAX_H3_BUCKET_DENOISE=\${MINIMAX_H3_BUCKET_DENOISE:-1}        # OFF -> DRAM OOM after ~16 distinct lengths
export MINIMAX_H3_CONSTRUCTION_WARMUP=\${MINIMAX_H3_CONSTRUCTION_WARMUP:-0}   # ON -> first served request corrupt
export MINIMAX_H3_VAE_OUTPUT=\${MINIMAX_H3_VAE_OUTPUT:-float}             # only read by pipelines >= 333841bbeb0
export MODEL_RUNNER=\${MODEL_RUNNER:-tt-minimax-h3-t2va}
EOT
  say "wrote $ENVF"
}

verify_setup() {
  [ -f "$RANKFILE" ] || die "missing $RANKFILE"
  [ -f "$RANKBIND" ] || die "missing $RANKBIND"
  [ -f "$MEDIA_ENV/bin/activate" ] || die "missing media python_env at $MEDIA_ENV"
  say "import smoke test (media python_env + env file)"
  (cd "$TMS" && bash -c "source $ENVF && source $MEDIA_ENV/bin/activate && python - <<'PY'
import os, ttnn, inspect
import ttnn._ttnn as t
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
import tt_model_runners.dit_runners
print('  ttnn python :', ttnn.__file__)
print('  _ttnn.so    :', os.path.realpath(t.__file__))
print('  TT_METAL_HOME:', os.environ['TT_METAL_HOME'])
print('  create_pipeline accepts:', [k for k in ('warmup','bucket_denoise','vae_output_type') if k in inspect.signature(MiniMaxH3Pipeline.create_pipeline).parameters])
PY" 2>&1 | grep -vE "DEBUG|INFO|WARNING|Config\{") || die "import smoke test failed"
  for h in $(hosts_list); do $SSH "$h" true 2>/dev/null || die "cannot ssh to $h"; done
  say "ok: ssh to all hosts"
}

setup() { setup_device_tree; setup_python_tree; setup_server_tree; write_env; verify_setup; say "setup complete"; }

# ---------------------------------------------------------------------------------------------------- run
task_runner() { case $1 in t2va) echo tt-minimax-h3-t2va;; fl2va) echo tt-minimax-h3-fl2va;; ref2va) echo tt-minimax-h3-ref2va;; *) die "task must be t2va|fl2va|ref2va";; esac; }
task_model()  { case $1 in t2va) echo MiniMax-H3;; fl2va) echo MiniMax-H3-FL2VA;; ref2va) echo MiniMax-H3-Ref2VA;; esac; }

held_total() { local n=0 c; for h in $(hosts_list); do c=$($SSH "$h" 'for f in /dev/tenstorrent/[0-9]*; do fuser $f 2>/dev/null; done | wc -w' 2>/dev/null || echo 0); n=$((n + c)); done; echo "$n"; }

stop() {
  say "stopping frontend on $RANK0 and ranks on all hosts"
  $SSH "$RANK0" 'pkill -TERM -f "[u]vicorn main:app" 2>/dev/null; sleep 3; pkill -KILL -f "[u]vicorn main:app" 2>/dev/null; true'
  for h in $(hosts_list); do $SSH "$h" 'pkill -TERM -f "[t]t_model_runners.video_runner" 2>/dev/null; true' & done; wait
  sleep 5
  for h in $(hosts_list); do $SSH "$h" 'pkill -KILL -f "[t]t_model_runners.video_runner" 2>/dev/null; pkill -TERM -f "[p]rted" 2>/dev/null; true' & done; wait
  pkill -TERM -f "[t]t-run --rank-binding" 2>/dev/null || true
  local i; for i in $(seq 1 12); do [ "$(held_total)" = 0 ] && break; sleep 5; done
  local held; held=$(held_total)
  [ "$held" = 0 ] && say "stopped; devices released" || say "WARNING: $held device handles still held (check 'h3_deploy.sh check'; a reset may be needed)"
  sleep 5   # let the driver settle -- launching immediately fails with 'Query mappings failed ... No such device'
}

start() {
  local task=${1:?task}; local runner model ts ranklog
  runner=$(task_runner "$task"); model=$(task_model "$task")
  [ -f "$ENVF" ] || die "no env file $ENVF -- run 'h3_deploy.sh setup' first"
  stop
  ts=$(date -u +%Y%m%d_%H%M%S); ranklog=$LOGDIR/quad.$task.$ts.log; ln -sfn "quad.$task.$ts.log" "$LOGDIR/workers.log"
  say "launching 4x32 ranks ($runner); merged rank log: $ranklog"
  (
    cd "$VM" || exit 1                      # tt-run resolves the rankfile relative to mpirun's cwd
    source "$MEDIA_ENV/bin/activate"; source "$ENVF"; export MODEL_RUNNER=$runner
    exec setsid nohup tt-run \
      --rank-binding "$RANKBIND" \
      --mpi-args "--host $HOSTS --rankfile $RANKFILE --bind-to none --tag-output --merge-stderr-to-stdout" \
      bash -c "cd $TMS && source $ENVF && export MODEL_RUNNER=$runner && source $MEDIA_ENV/bin/activate && SP_MESH_4X32=true python -m tt_model_runners.video_runner" \
      > "$ranklog" 2>&1
  ) &
  sleep 5
  say "launching SP frontend on $RANK0 (MODEL=$model, port 8000)"
  timeout 90 $SSH "$RANK0" "mkdir -p $LOGDIR; cd $TMS && source $MEDIA_ENV/bin/activate && source $ENVF && USE_ASYNC_VIDEO=true MODEL='$model' MEDIA_URL_ALLOWED_DOMAINS=samplelib.com CANARY_ENABLED=true TT_VIDEO_SHM_INPUT=tt_video_in TT_VIDEO_SHM_OUTPUT=tt_video_out MODEL_WEIGHTS_PATH='Minimax H3' USE_GREEDY_BASED_ALLOCATION=false VIDEO_REQUEST_TIMEOUT_SECONDS=5000 REQUEST_PROCESSING_TIMEOUT_SECONDS=5000 MODEL_RUNNER=sp_runner setsid nohup uvicorn main:app --host 0.0.0.0 --port 8000 > $LOGDIR/frontend.$task.$ts.log 2>&1 < /dev/null & sleep 1; echo frontend pid \$!"
  wait_ready "${2:-3600}"
}

wait_ready() {
  local deadline=$(( $(date +%s) + ${1:-3600} )) live
  say "waiting for readiness (cold kernel cache: ~12 min; warm: 1-3 min)"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if sed 's/\x1b\[[0-9;]*m//g' "$LOGDIR/workers.log" 2>/dev/null | grep -vE "DRAM Auto slice" | grep -qE "pipeline creation failed|the device is unrecoverable|TT_THROW|Device initialization failed| - ERROR - "; then
      say "FATAL in rank log:"; sed 's/\x1b\[[0-9;]*m//g' "$LOGDIR/workers.log" | grep -vE "DRAM Auto slice" | grep -E "pipeline creation failed|unrecoverable|TT_THROW|Device initialization failed| - ERROR - " | head -3 | cut -c1-200; return 1
    fi
    pgrep -f "tt-run --rank-binding" >/dev/null || { say "tt-run process gone"; tail -5 "$LOGDIR/workers.log" | cut -c1-200; return 1; }
    live=$(curl -s -m 5 "$URL/tt-liveness" 2>/dev/null)
    if grep -q "SHM bridge ready" "$LOGDIR/workers.log" 2>/dev/null && echo "$live" | grep -q '"canary_state": *"healthy"'; then
      say "READY: $(grep -c 'Model ready for inference' "$LOGDIR/workers.log") ranks; $(echo "$live" | cut -c1-120)"; return 0
    fi
    sleep 15
  done
  say "timed out waiting for readiness"; return 1
}

status() { echo "tt-run: $(pgrep -f 'tt-run --rank-binding' | tr '\n' ' ')"; curl -s -m 5 "$URL/tt-liveness"; echo; [ -L "$LOGDIR/workers.log" ] && echo "rank log: $(readlink -f "$LOGDIR/workers.log")"; }
check()  { for h in $(hosts_list); do $SSH "$h" 'echo "$(hostname): tt-held=$(for f in /dev/tenstorrent/[0-9]*; do fuser $f 2>/dev/null; done | wc -w) video_runner=$(pgrep -f tt_model_runners.video_runner | wc -l) root_free=$(df -h / | awk "NR==2{print \$4}")"'; done; }
logs()   { sed 's/\x1b\[[0-9;]*m//g' "$(readlink -f "$LOGDIR/workers.log")" | grep -vE "DRAM Auto slice|deprecated" | tail -${1:-40} | cut -c1-200; }

reset() {
  stop
  say "recover.sh on the quad (tt-smi reset + link validation, up to 3 attempts; ~2-7 min each)"
  (cd "$METAL_SHARED" && source python_env/bin/activate && source "$VM/metal_env_H3.sh" && \
   ./tools/scaleout/exabox/recover.sh --hosts "$HOSTS" \
     --cabling-descriptor-path "$DESC_DIR/${DESC_PREFIX}_cabling_descriptor.textproto" \
     --deployment-descriptor-path "$DESC_DIR/${DESC_PREFIX}_deployment_descriptor.textproto" \
     --num-iterations 10 --skip-version-check --max-attempts 3) > "$LOGDIR/recover.$(date -u +%Y%m%d_%H%M%S).log" 2>&1
  local log; log=$(ls -t "$LOGDIR"/recover.*.log | head -1)
  if grep -q "Recovery succeeded" "$log"; then say "recovery ok; settling 60 s"; sleep 60
  else say "RECOVERY FAILED -- see $log (c02u07 tray1<->tray2 asic6 ch7 is a known flaky cable; 'missing channel connection' means it needs a reseat)"; tail -3 "$log" | cut -c1-160; return 1; fi
}

# ---------------------------------------------------------------------------------------------------- probe
probe() {
  local task=${1:?task} n=${2:-3} out=$LOGDIR/probe; mkdir -p "$out"
  command -v ffmpeg >/dev/null || die "ffmpeg not found"
  local key=$out/keyframe_1344x768.jpg ref=$out/ref_512.png
  [ -f "$key" ] || ffmpeg -v error -y -f lavfi -i testsrc2=size=1344x768:rate=1 -frames:v 1 -q:v 2 "$key"
  [ -f "$ref" ] || ffmpeg -v error -y -f lavfi -i color=c=steelblue:s=512x512 -frames:v 1 "$ref"
  local i; for i in $(seq 1 "$n"); do
    "$MEDIA_ENV/bin/python" - "$task" "$URL" "$KEY" "$key" "$ref" "$out" "$i" <<'PY'
import base64, json, pathlib, subprocess, sys, time, urllib.request
task, url, key, kf, ref, out, i = sys.argv[1:8]; out = pathlib.Path(out)
b64 = lambda p: base64.b64encode(pathlib.Path(p).read_bytes()).decode()
body = {"prompt": "A calm seaside village at golden hour, gentle waves, birds", "aspect_ratio": "16:9", "duration_seconds": 5, "seed": 7}
if task == "fl2va": body["image_prompts"] = [{"image": b64(kf), "frame_pos": 0}]
if task == "ref2va": body["references"] = {"images": [{"b64": b64(ref)}]}
ENDPOINT = {"t2va": "/v1/videos/generations", "fl2va": "/v1/videos/generations/i2v", "ref2va": "/v1/videos/generations/ref2va"}
def http(m, path, data=None, timeout=300):
    r = urllib.request.Request(url + path, data=json.dumps(data).encode() if data is not None else None, method=m,
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as h: return h.read()
    except urllib.error.HTTPError as e:
        print(f"PROBE {task}-{i} FAILED HTTP {e.code} on {m} {path}: {e.read()[:300].decode(errors='replace')}"); sys.exit(1)
t0 = time.time(); job = json.loads(http("POST", ENDPOINT[task], body))["id"]
status = None
while time.time() - t0 < 1800:
    status = json.loads(http("GET", f"/v1/videos/generations/{job}", timeout=60)).get("status")
    if status in ("completed", "failed", "cancelled", "error"): break
    time.sleep(5)
wall = time.time() - t0
if status != "completed": print(f"PROBE {task}-{i} FAILED status={status} wall={wall:.0f}s job={job}"); sys.exit(1)
mp4 = out / f"{task}-{i}.{job[:8]}.mp4"; mp4.write_bytes(http("GET", f"/v1/videos/generations/{job}/download", timeout=600))
vol = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp4), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
import re
mean = float(re.search(r"mean_volume: ([-0-9.]+)", vol).group(1)); peak = float(re.search(r"max_volume: ([-0-9.]+)", vol).group(1))
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.strip().split(",")
w, h, nf = int(pr[0]), int(pr[1]), int(pr[2]); vbytes = sum(int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=size", "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.split())
bppf = 8 * vbytes / (w * h * nf)
verdict = "BAD" if (peak > -1.0 or bppf > 0.6) else "OK"
extra = ""
if task == "fl2va":
    try:
        import numpy as np; from PIL import Image
        f0 = out / f"{task}-{i}.f0.png"; subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-frames:v", "1", str(f0)], check=True)
        a = np.asarray(Image.open(f0).convert("RGB").resize((336, 192)), dtype=float); b = np.asarray(Image.open(kf).convert("RGB").resize((336, 192)), dtype=float)
        pcc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); extra = f" keyframe_pcc={pcc:.3f}"
        if pcc < 0.5: verdict = "BAD"
    except Exception as e: extra = f" (keyframe check skipped: {e})"
print(f"PROBE {task}-{i} {verdict} wall={wall:.0f}s {w}x{h} frames={nf} audio mean/peak={mean}/{peak} dB bppf={bppf:.3f}{extra} file={mp4}")
PY
  done
}

case ${1:-} in
  setup) setup ;;
  start) start "${2:?task}" "${3:-3600}" ;;
  wait-ready) wait_ready "${2:-3600}" ;;
  stop) stop ;;
  reset) reset ;;
  status) status ;;
  check) check ;;
  logs) logs "${2:-40}" ;;
  probe) probe "${2:?task}" "${3:-3}" ;;
  *) sed -n 2,22p "$0"; exit 1 ;;
esac
