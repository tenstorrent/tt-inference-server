# MiniMax-H3 fl2va on the C12 quad — the configuration that works (2026-09-10)

This is the recipe that produced clean fl2va output on the 4×32 Blackhole Galaxy quad
(bh-glx-EXP-c01u21 / c01u14 / c02u07 / c01u07): tier-1 5/5 (three runs), shapes suite 32/32 requests clean,
frame 0 follows the first keyframe at PCC 0.999–1.000 on every request, warm requests included. The same
deployment recipe serves t2va (25/25 clean) and ref2va (image/audio references clean; 3-video references still corrupt).

## 0. TL;DR — pin these, nothing else

| layer | use | do NOT use (and why) |
|---|---|---|
| tt-metal device side (`.so`, kernels, firmware, runtime) | build of **`162a86b008a`** ("add H3 bucketing") | the 2026-09-09 build of `3c9db97f5ca`: main `1b17275b8df` (LLK #55107) gives full-frame noise + railed audio from request 1 |
| tt-metal Python (`models/tt_dit`, `ttnn` python) | **`34260b25483`** ("pad audio encoding") | `333841bbeb0`+ (device-stitched yuv420 decode): first request fine, every later request scrambled |
| tt-inference-server | **`78d516584` + `70a756282`** (SP side-file fix) + the small `dit_runners.py` patch below | `78d516584` alone drops `aspect_ratio`/`duration_seconds` on the SP path (everything becomes 16:9 / 5 s) |
| python env | the media-server `python_env` (`tt-inference-server/tt-media-server/python_env`), ttnn python taken from the pinned tree via `PYTHONPATH` | the zni venv (needs extra packages) |
| knobs | `MINIMAX_H3_TRACE_DENOISE=0`, **`MINIMAX_H3_BUCKET_DENOISE=1`**, **`MINIMAX_H3_CONSTRUCTION_WARMUP=0`**, `TT_METAL_CACHE` unset | bucketing off → DRAM OOM after ~16 distinct padded lengths; construction warmup on → first request corrupt |

Do not mix layers: Python ≥ `333841bbeb0` on the old device tree corrupts warm requests (yuv420 and, intermittently,
float); kernel/firmware sources newer than the `.so` hang at the first dispatch.

## Scripted version

`zni_worktrees/h3_deploy.sh` does everything in sections 2-5 (idempotent):
```bash
WT=/data/DC-deploy/vision-models/zni_worktrees
$WT/h3_deploy.sh setup            # worktrees at the pinned commits, device-tree build if missing, symlinks, SP fix, patch, env file, import check
$WT/h3_deploy.sh start fl2va      # stop anything running, launch 4x32 ranks + frontend, wait until canary healthy
$WT/h3_deploy.sh probe fl2va 3    # three identical requests, judged (audio/bppf + frame-0 vs keyframe PCC)
$WT/h3_deploy.sh status | check | logs | stop | reset
```
Logs go to `$WT/deploy_logs/` (`workers.log` = merged rank log). Knobs: `H3_SKIP_BUILD=1`, `H3_FORCE_CHECKOUT=1`
(park a moved worktree and re-pin it), `H3_FORCE_ENV=1` (rewrite the env file), `H3_API_KEY`, `H3_WT`.

## 1. Prerequisites (already in place on C12)

- Shared NFS `/data/DC-deploy/vision-models` (trees, weights `hf_data/MiniMax-H3-diffusers`, weight cache `tt_dit_cache`).
- `/data/DC-deploy/vision-models/test_C12_rankfile` (rank 0 = c01u21, 1 = c01u14, 2 = c02u07, 3 = c01u07) and the
  rank binding `tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml` (torus_xy mesh graph).
- Cluster descriptors `/data/scaleout_configs/bh_glx_exabox/C12_{cabling,deployment}_descriptor.textproto`
  (used by `tools/scaleout/exabox/recover.sh`).
- OpenMPI ulfm 5.0.7 in `/opt/openmpi-v5.0.7-ulfm`, ffmpeg in `/data/DC-deploy/vision-models/ffmpeg`, tt-firmware 19.8.1.
- Passwordless ssh from the launching host to all four hosts (same `vision-models` account). Home directories are
  **per host** (not shared): the default kernel cache `~/.cache/tt-metal-cache` is therefore per host, which is what you want.
- `/data/DC-deploy/vision-models/metal_env_H3.sh` — the base environment (model paths, MPI, fabric timeouts, SHM names).

## 2. Source trees (three git worktrees under `/data/DC-deploy/vision-models/zni_worktrees/`)

### 2a. Device tree `tt-metal-old` — everything that runs on the chips
```bash
cd /data/DC-deploy/vision-models/tt-metal
git worktree add --detach /data/DC-deploy/vision-models/zni_worktrees/tt-metal-old 162a86b008a
cd /data/DC-deploy/vision-models/zni_worktrees/tt-metal-old
ln -s /data/DC-deploy/vision-models/tt-metal/.cpmcache .cpmcache     # avoids the boost/CPM download stall
git submodule update --init tt_metal/third_party/umd tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors
./build_metal.sh                                                     # let CMake fetch sfpi (7.72); --use-system-sfpi mismatches
```
Result: `build_Release/lib/{libtt_metal,_ttnncpp,_ttnn}.so`, `runtime/` (sfpi + hw libs), `ttnn/ttnn/_ttnn.so`.
This tree is `TT_METAL_HOME`: kernels, firmware, dispatch and runtime all come from here, so they match the `.so`.

### 2b. Python tree `tt-metal-0909` — `models/tt_dit` and the `ttnn` python package
```bash
cd /data/DC-deploy/vision-models/tt-metal
git worktree add --detach /data/DC-deploy/vision-models/zni_worktrees/tt-metal-0909 34260b25483
cd /data/DC-deploy/vision-models/zni_worktrees/tt-metal-0909
OLD=/data/DC-deploy/vision-models/zni_worktrees/tt-metal-old; SH=/data/DC-deploy/vision-models/tt-metal
ln -sfn $OLD/build_Release build
ln -sfn $SH/runtime runtime                 # unused at run time (TT_METAL_HOME is the device tree) but keeps tools happy
ln -sfn $SH/internal-prodia internal-prodia
ln -sfn $SH/python_env python_env
ln -sfn $OLD/ttnn/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so      # the python package loads the old build's extension
for s in umd tracy tt-cluster-descriptors; do rmdir tt_metal/third_party/$s 2>/dev/null; ln -sfn $OLD/tt_metal/third_party/$s tt_metal/third_party/$s; done
```

### 2c. Server tree `tms-0909`
```bash
cd /data/DC-deploy/vision-models/tt-inference-server
git worktree add --detach /data/DC-deploy/vision-models/zni_worktrees/tms-0909 78d516584
cd /data/DC-deploy/vision-models/zni_worktrees/tms-0909
git cherry-pick -x 70a756282                       # carry duration and aspect_ratio through the SP side file
ln -sfn /data/DC-deploy/vision-models/tt-inference-server/tt-media-server/python_env tt-media-server/python_env
```
Then patch `tt-media-server/tt_model_runners/dit_runners.py`, `TTMiniMaxH3Runner.create_pipeline()`: pass the
warmup gate (and the VAE knob only where the pipeline has it):
```python
            import inspect
            accepted = inspect.signature(MiniMaxH3Pipeline.create_pipeline).parameters
            extra = {}
            if "warmup" in accepted:            # gate the construction-time warmup (metal 56cdeeb9095)
                extra["warmup"] = os.environ.get("MINIMAX_H3_CONSTRUCTION_WARMUP", "1") != "0"
            if "vae_output_type" in accepted:   # only exists from 333841bbeb0 on
                extra["vae_output_type"] = os.environ.get("MINIMAX_H3_VAE_OUTPUT", "yuv420")
            return MiniMaxH3Pipeline.create_pipeline(
                mesh_device=self.ttnn_device, weights_dir=self._weights_dir(), task=self.pipeline_task,
                dit_fsdp=self.dit_fsdp,
                trace_denoise=_minimax_h3_env_bool("MINIMAX_H3_TRACE_DENOISE"),
                bucket_denoise=_minimax_h3_env_bool("MINIMAX_H3_BUCKET_DENOISE"),
                **extra,
            )
```
Keep `tt_model_runners/minimax_h3_policy.py` at `78d516584` (self-contained; the pinned metal tree has no `policy.py`).

## 3. Environment file — `zni_worktrees/env_c12_0909.sh`
```bash
source /data/DC-deploy/vision-models/metal_env_H3.sh
WT=/data/DC-deploy/vision-models/zni_worktrees
export ZNI_METAL_PY=$WT/tt-metal-0909          # models/ + ttnn python
export ZNI_METAL_BIN=$WT/tt-metal-old          # everything device-side
export TT_METAL_HOME="$ZNI_METAL_BIN"
export TT_METAL_RUNTIME_ROOT="$ZNI_METAL_BIN"
export LD_LIBRARY_PATH="$ZNI_METAL_BIN/build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ZNI_METAL_PY/ttnn:$ZNI_METAL_PY/:$ZNI_METAL_PY/internal-prodia/"   # ttnn dir FIRST: beats the env's editable install
unset TT_METAL_CACHE                           # per-host ~/.cache/tt-metal-cache; never a shared NFS dir (rank races)
export MINIMAX_H3_TRACE_DENOISE=0
export MINIMAX_H3_BUCKET_DENOISE=1
export MINIMAX_H3_CONSTRUCTION_WARMUP=0
export MODEL_RUNNER=${MODEL_RUNNER:-tt-minimax-h3-t2va}
```
`MINIMAX_H3_VAE_OUTPUT`, `MINIMAX_H3_YUV_READBACK`, `MINIMAX_H3_ENCODER_READBACK` are no-ops at this Python version.

## 4. Launch

All four ranks, from the **launching host**, with the media python_env active (it has `tt-run`), from the directory
that holds the rankfile (tt-run rewrites the rankfile path relative to mpirun's cwd):
```bash
TMS=/data/DC-deploy/vision-models/zni_worktrees/tms-0909/tt-media-server
ENVF=/data/DC-deploy/vision-models/zni_worktrees/env_c12_0909.sh
HOSTS=bh-glx-EXP-c01u21,bh-glx-EXP-c01u14,bh-glx-EXP-c02u07,bh-glx-EXP-c01u07
cd /data/DC-deploy/vision-models
source $TMS/python_env/bin/activate && source $ENVF
setsid nohup tt-run \
  --rank-binding /data/DC-deploy/vision-models/zni_worktrees/tt-metal-old/tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host $HOSTS --rankfile /data/DC-deploy/vision-models/test_C12_rankfile --bind-to none --tag-output --merge-stderr-to-stdout" \
  bash -c "cd $TMS && source $ENVF && export MODEL_RUNNER=tt-minimax-h3-fl2va && source $TMS/python_env/bin/activate && SP_MESH_4X32=true python -m tt_model_runners.video_runner" \
  > quad.fl2va.log 2>&1 &
```
The SP frontend on the rank-0 host (c01u21), same env:
```bash
ssh bh-glx-EXP-c01u21 "cd $TMS && source python_env/bin/activate && source $ENVF && \
  USE_ASYNC_VIDEO=true MODEL='MiniMax-H3-FL2VA' MEDIA_URL_ALLOWED_DOMAINS=samplelib.com CANARY_ENABLED=true \
  TT_VIDEO_SHM_INPUT=tt_video_in TT_VIDEO_SHM_OUTPUT=tt_video_out MODEL_WEIGHTS_PATH='Minimax H3' \
  USE_GREEDY_BASED_ALLOCATION=false VIDEO_REQUEST_TIMEOUT_SECONDS=5000 REQUEST_PROCESSING_TIMEOUT_SECONDS=5000 \
  MODEL_RUNNER=sp_runner setsid nohup uvicorn main:app --host 0.0.0.0 --port 8000 > frontend.fl2va.log 2>&1 < /dev/null &"
```
For t2va / ref2va: `MODEL_RUNNER=tt-minimax-h3-t2va` / `tt-minimax-h3-ref2va` on the ranks and `MODEL='MiniMax-H3'` /
`'MiniMax-H3-Ref2VA'` on the frontend. One deployment at a time — the quad is single-tenant.

**Ready** when the rank log shows `Rank 0: Model ready for inference` and `SHM bridge ready`, and
`curl http://bh-glx-EXP-c01u21:8000/tt-liveness` reports `"canary_state":"healthy"`. Ignore the
`TT_FATAL: DRAM Auto slice could not find valid slice configuration` lines — the auto-slicer catches them.
Start → ready: ~40 s–2 min with a warm kernel cache (~12 min on a cold cache).

**Stop**: TERM then KILL the frontend (it ignores TERM), `pkill -f tt_model_runners.video_runner` on every host, then
`prted`; confirm `/dev/tenstorrent/*` are no longer held (`fuser`). Wait a few seconds before relaunching —
launching while the previous ranks still hold the devices fails with `Query mappings failed on device 0: No such device`.

**Reset** after a hang ("device timeout ... unrecoverable", "Timed out while waiting for active ethernet core"):
```bash
cd /data/DC-deploy/vision-models/tt-metal && source python_env/bin/activate && source ../metal_env_H3.sh
./tools/scaleout/exabox/recover.sh --hosts $HOSTS \
  --cabling-descriptor-path /data/scaleout_configs/bh_glx_exabox/C12_cabling_descriptor.textproto \
  --deployment-descriptor-path /data/scaleout_configs/bh_glx_exabox/C12_deployment_descriptor.textproto \
  --num-iterations 10 --skip-version-check --max-attempts 3
```
"links not healthy" is usually the c02u07 tray1↔tray2 cable (asic 6 ch 7) flapping; a retry has always recovered it so
far (5 of 8 runs today needed one). Let the links settle ~60 s before launching.

## 5. Sending an fl2va request
```bash
B64=$(base64 -w0 keyframe_1344x768.jpg)
curl -s -X POST http://bh-glx-EXP-c01u21:8000/v1/videos/generations/i2v \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer your-secret-key' \
  -d "{\"prompt\":\"A calm seaside village at golden hour, gentle waves\",\"aspect_ratio\":\"16:9\",\"duration_seconds\":5,\"seed\":7,
       \"image_prompts\":[{\"image\":\"$B64\",\"frame_pos\":0}]}"
# endpoints on this server version: t2va -> /v1/videos/generations, fl2va -> /v1/videos/generations/i2v, ref2va -> /v1/videos/generations/ref2va
# frame_pos 0 = first keyframe, -1 = last keyframe (both allowed); 4-15 s; aspect 16:9 21:9 4:3 1:1 3:4 9:16
curl -s -H 'Authorization: Bearer your-secret-key' http://bh-glx-EXP-c01u21:8000/v1/videos/generations/<id>            # status
curl -s -H 'Authorization: Bearer your-secret-key' http://bh-glx-EXP-c01u21:8000/v1/videos/generations/<id>/download -o out.mp4
```
The first keyframe is stretched onto the canvas, a second (last) keyframe is cover-cropped and only softly anchored.
Quick content check: frame 0 vs the keyframe should correlate at > 0.95 (RGB Pearson after resizing both to the same
size); clean clips sit at 0.15–0.45 bits/pixel/frame with audio around -35..-45 dB mean — noise/scramble shows up as
> 0.6 bits/pixel/frame and audio at 0 dB.

Expected times (20 steps): 16:9 5 s ≈ 25 s warm (first request per padded length +5–30 s for the compile);
10 s ≈ 35 s; 15 s ≈ 50–55 s; first+last 5 s ≈ 25–30 s.
