# Running MiniMax-H3 `t2va` on a single-host Galaxy through the inference server

Text in, a video **with its own soundtrack** out. Every published 768P working point: six
aspect ratios (21:9 .. 9:16) x 5 / 10 / 15 s, at 24 fps.

Single host, 4x8 Blackhole Galaxy (32 chips). There is **no `tt-run`, no MPI, no rankfile and no
`sp_runner`** — that machinery is for multi-host deployments. This is one uvicorn process talking to
one mesh.

## 1. Build tt-metal

```bash
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git checkout a01e402540629bb5cadd8642fbde1b83c06f6a05  # 'Let a line-cabled mesh run the H3 FFN'
./build_metal.sh --build-all
```

That commit is `cglagovich/minimax-h3-linear`: current `main` plus the one gate the FFN's fused
reduce-scatter needs to fall back on a line-cabled mesh. On a ring it is a no-op, so this pin is
correct for both topologies; see [fabric topology](#fabric-topology) below. Building plain `main`
instead is fine **only** for a ring.

## 2. Clone the inference server

```bash
git clone git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server
git checkout minimax-h3-server
```

No commit pin here: this doc ships in the repo, so the revision you are reading it at is the one to
use. Only tt-metal is pinned, because the model code has to match.

## 3. Create the python env inside the media server

From the **tt-metal** repo, so the env gets `ttnn`:

```bash
cd /path_to/tt-metal
./create_venv.sh --env-dir /path_to/tt-inference-server/tt-media-server/python_env --bundle-python
source /path_to/tt-inference-server/tt-media-server/python_env/bin/activate
```

Then add the server's own requirements:

```bash
cd /path_to/tt-inference-server/tt-media-server
uv pip install -r requirements.txt
```

### diffusers is NOT required to serve

An earlier version of this doc said a MiniMax-H3 diffusers dev build was required. It is not, for
this deployment. Nothing in the serving path imports diffusers: the tt-metal pipeline, the server's
`TTMiniMaxH3Runner` and the tokenizer all load without it, and every `diffusers` mention in
`models/tt_dit/.../minimax_h3/` is a comment about how the checkpoint's keys were converted.
Verified by probing each import against a stock env (diffusers 0.35.1, transformers 4.53.0).

The dev build *is* needed to run the **model-side tests** in tt-metal, which build reference modules
from `diffusers.models.autoencoders.autoencoder_kl_minimax_h3`. If that is what you are doing:

```bash
uv pip install --force-reinstall \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

Do not force-reinstall it just to serve. This env is shared with the other models on the box, and a
diffusers bump for no reason is a way to break one of them.

### Weights (~120 GB)

A diffusers-format snapshot with `transformer/`, `transformer_ref/`, `text_encoder/`, `vae/`,
`audio_vae/`, `tokenizer/`, `processor/`, `scheduler/`, `audio_scheduler/` and
`modular_model_index.json`. The transformer partition is ~62 GB and the text encoder alone is
50.3 GB bf16 across 12 shards.

[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) is already published in
diffusers modular format, so it is used as-is — no conversion step. Not gated.

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3 --local-dir /path_to/MiniMax-H3-diffusers
```

To skip what this deployment never reads (`transformer_ref/` is `ref2va` only, and the rest is
docs and demo assets):

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3 --local-dir /path_to/MiniMax-H3-diffusers \
  --exclude "transformer_ref/*" "FL2VA/*" "Ref2VA/*" "assets/*" "docs/*" "scripts/*"
```

### ttnn weight cache (~68 GB, built on first run)

`TT_DIT_CACHE_DIR` just needs to be an empty writable directory:

```bash
mkdir -p /path_to/tt_dit_cache
```

The **first startup is much slower** — it reads the safetensors, converts and writes ~68 GB of
device-layout weights, plus builds the AdaLN modulation table for the 50-step schedule. Budget
20-30 minutes and plenty of disk. Every later start reads the cache instead. The directory must be
persistent across restarts, or each restart pays that again.

## 4. Environment

```bash
export TT_METAL_HOME=/path_to/tt-metal
export MODEL="MiniMax-H3"
export MODEL_RUNNER=tt-minimax-h3-t2va
export DEVICE="galaxy"
export ARCH_NAME=blackhole
export MESH_DEVICE="(4, 8)"

export MINIMAX_H3_MODEL_PATH=/path_to/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/path_to/tt_dit_cache

export USE_ASYNC_VIDEO=true

# Optional. Every `Settings` field is env-overridable; queue depth defaults to 5000, which is
# effectively unbounded buffering. A small value keeps the accelerator fed without letting a
# request sit for hours behind a queue.
export MAX_QUEUE_SIZE=2
```

Four of these bite if wrong:

| var | what happens if it is wrong |
|---|---|
| `TT_METAL_HOME` | unset -> the kernel cache falls back to root-owned `/built` and mesh init dies as *"Failed to generate binaries for fabric_erisc_router ... Permission denied"* |
| `TT_DIT_CACHE_DIR` | unset -> **degrades silently**, ~713 s instead of ~64 s |
| `MINIMAX_H3_MODEL_PATH` | the single weights lever. Unset -> nothing finds the snapshot, and the model-side gates *skip* rather than fail, so a run can look clean while testing nothing. (`MINIMAX_H3_DIFFUSERS_DIR` was folded into this one and is no longer read.) |
| `USE_ASYNC_VIDEO` | false -> the request blocks for ~70 s and clients time out |
| `MODEL_RUNNER` | set it. `MODEL`+`DEVICE` resolve the runner on their own, but an inherited `MODEL_RUNNER` **wins over them silently** -- and `/data/cglagovich/inference_server_env.sh` exports `MODEL_RUNNER=tt-wan2.2`, so sourcing that script loads Wan and ignores `MODEL="MiniMax-H3"` entirely. The only symptom is a `WanPipeline` traceback in a log you were expecting to say MiniMax. |
| `MINIMAX_H3_FABRIC_TOPOLOGY` | `ring` (default) or `linear`, and it must match how the mesh is cabled. Anything else is rejected at startup rather than deep in fabric init. See [fabric topology](#fabric-topology). |

The weights directory needs `transformer/`, `text_encoder/`, `vae/`, `audio_vae/` and `tokenizer/`.
They are mounted, never baked: ~62 GB for the transformer partition, and the text encoder alone is
50.3 GB bf16 across 12 shards. `TT_DIT_CACHE_DIR` must be writable and persistent, or every restart
pays the cold path.

## 5. Start the server

```bash
cd /path_to/tt-inference-server/tt-media-server
source python_env/bin/activate
./run_uvicorn.sh --skip-venv
```

## 6. Wait for readiness

Weight load only -- **nothing is warmed by default**, so this is a few minutes on a warm ttnn cache
(longer the very first time, while the ~68 GB cache and the AdaLN table are built).

```
Device 0,...,31: Loading MiniMax-H3...
Device 0,...,31: Model loaded, warming nothing (MINIMAX_H3_WARM_SHAPES unset). The first request at each shape compiles.
```

**Readiness does not mean warm.** Programs are keyed on the padded sequence length, so the first
request at each of the 18 shapes compiles inside that request -- minutes, not seconds -- and says so:

```
padded_len 37888 was not resident (resident: none); this request paid compilation.
```

It is logged once per shape, not per request. To pre-pay it for the shapes you actually serve, set
`MINIMAX_H3_WARM_SHAPES` before starting:

| value | effect |
|---|---|
| unset | warm nothing (default) |
| `16:9@5,9:16@10` | warm just those, `@` seconds |
| `all` | all 6 ratios x 3 durations -- correct for production, but **hours** of startup (4-16 min per shape, worst at 15 s / 1 MPix) |

```bash
curl -s localhost:8000/tt-liveness      # {"status":"alive","model_ready":true,...}
```

### Fabric topology

H3 runs its collectives on a **ring** by default, which is what a 4x8 Blackhole Galaxy is normally
cabled for. A deployment whose mesh is only a line sets one variable before starting:

```bash
export MINIMAX_H3_FABRIC_TOPOLOGY=linear   # 'ring' is the default; nothing else is accepted
```

It moves the fabric config and the collectives the model issues **together** -- `FABRIC_1D` instead
of `FABRIC_1D_RING`, and `Topology.Linear` into the pipeline's `CCLManager`. They are not
independently settable on purpose: a line fabric under ring collectives dies as `TT_FATAL
fabric.cpp:174 forwarding_direction.has_value()`. Confirm the pair in the log at startup:

```
Fabric Initialized with config FabricConfig::FABRIC_1D
Device 0,...,31: MiniMax-H3 collectives on Topology.Linear
```

`linear` costs about 30%, all of it in denoise, because two fusions are ring-only -- the fused
all-gather-matmul (`use_fused_agmm`) and the FFN's fused matmul + reduce-scatter. Measured at
16:9/5s, warm:

| topology | compute | denoise (49 steps) | VAE | audio |
|---|---|---|---|---|
| `ring` | 74.2 s | 60.1 s | 6.5 s | 6.8 s |
| `linear` | 96.3 s | 82.6 s | 6.6 s | 6.8 s |

The two VAEs are data-parallel over work units with replicated weights, so they do not move.

`linear` needs the tt-metal pin from step 1. On plain `main` the fused reduce-scatter is selected on
shape alone and the **first denoise step of the first request** dies as *"MinimalMatmulStridedReduce
ScatterAsync only supports Ring topology"* -- after warmup has already reported the model loaded.

## 7. Generate

`API_KEY` defaults to `your-secret-key`.

```bash
# submit -> 202 + job id. aspect_ratio and duration_seconds are optional; omitted gives 16:9 / 5 s.
curl -X POST localhost:8000/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -d '{"prompt":"A red fox steps through wet grass at dawn.","aspect_ratio":"9:16","duration_seconds":10,"seed":0}'

# poll until "status":"completed"
curl -H 'Authorization: Bearer your-secret-key' \
  localhost:8000/v1/videos/generations/<job_id>

# download the muxed mp4
curl -H 'Authorization: Bearer your-secret-key' \
  localhost:8000/v1/videos/generations/<job_id>/download -o out.mp4
```

Forward port 8000 to your machine for web access / the video request UI.

Verify both streams actually arrived — a silent track is a bug, not a partial success:

```bash
# ffprobe is often absent -- an env whose ffmpeg came from imageio_ffmpeg ships only ffmpeg -- so
# read the streams with ffmpeg itself. Expect one h264 video and one aac stereo 32000 Hz track,
# at the canvas and duration you asked for.
ffmpeg -hide_banner -i out.mp4 2>&1 | grep -E 'Stream #|Duration'
```

A missing audio stream is a bug, not a partial success. To count frames as well:

```bash
ffmpeg -v error -i out.mp4 -map 0:v:0 -f rawvideo -y /dev/null -stats 2>&1 | tail -1
```

## Served shapes

| ratio | canvas | 5 s (124f) | 10 s (243f) | 15 s (362f) |
|---|---|---|---|---|
| 21:9 | 1536x672 | 70.0 s | 175.3 s | 324.2 s |
| 16:9 | 1344x768 | 69.5 s | 174.7 s | 325.4 s |
| 4:3 | 1024x768 | 52.4 s | 121.2 s | 221.2 s |
| 1:1 | 768x768 | 38.2 s | 81.9 s | 142.3 s |
| 3:4 | 768x1024 | 52.0 s | 121.7 s | 219.9 s |
| 9:16 | 768x1344 | 69.1 s | 174.8 s | 324.2 s |

Warm compute per request, measured 2026-08-13 on `9c96923d1bb`. Durations `4`..`15` are all
servable; 5 / 10 / 15 are shown because they are what was measured. End-to-end latency tracks these
closely -- a warm request measured 72 s E2EL against 70 s of compute, so budget ~2 s of non-compute
overhead plus any queue wait. Two things the numbers say: cost
tracks pixel count and sequence length only -- equal-pixel ratios agree to within 0.6 %, so
orientation is free -- and duration is superlinear, 2.92x the frames costing 4.67x at 1 MPix but
3.67x at 0.59 MPix, because denoise carries a quadratic term while VAE and audio decode stay linear.

## What to expect

Steady-state at the default shape (16:9, 5 s), per request:

| stage | s | share |
|---|---|---|
| Encoder (novel prompt) | 0.4 | 0.6 % |
| Denoise (49 forwards) | 57.2 | 83.4 % |
| VAE decode | 4.3 | 6.2 % |
| Audio decode | 6.8 | 9.8 % |
| **Total (compute)** | **68.6** | |

1168 ms per forward, 13.3x realtime (compute / 5.17 s of video). Compute only: weight load,
prompt-embedding cache writes and mp4 export are outside these rows.

Measured 2026-08-13 on `9c96923d1bb` via `test_t2va_end_to_end[4x8-16x9_5s]`, one 4x8 Blackhole
Galaxy, warm. Reproduced within 1.3 % across three runs spanning a rebase onto main, so treat a
>5 % move as a real regression rather than noise. Numbers come from the model-side gate, not from
the server's own instrumentation, so server-side overhead is not included.

The **first request after warmup is ~12 s slower** — a first-step transient that settles from ~12 s
to ~1.1 s and then stays there. A second warmup pass does not absorb it. Cause not yet identified.

A repeated prompt is faster than a novel one: prompt embeddings are cached on disk, so the Encoder
row drops to 0.0 s.

## Constraints

| constraint | detail |
|---|---|
| `aspect_ratio` | One of `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`. Anything else is a 422 listing those. The model accepts 1:4..4:1, but only these are calibrated. Omitted gives `16:9`. |
| `duration_seconds` | Any integer `4`..`15` -- what the MiniMax API accepts. Anything else is a 422. Omitted gives `5`. The video VAE encodes in 17-frame chunks, so only `17n + 5` frame counts exist and a request rounds **up** to the next one: the clip is never shorter than asked, by at most 0.67 s (`13` -> 13.667 s). `8` is the only exact fit (192 frames). |
| Unknown fields | Rejected, not ignored. `{"resolution": "1080P", "duration": 9}` is a 422 naming the fields this deployment reads -- silently dropping them would tell a caller it got something it did not. Note the field is `duration_seconds`, and resolution is chosen with `aspect_ratio`. |
| Resolution | 768P throughout: short edge 768 from 16:9 to 9:16, area capped at ~1 MPix for wider (21:9 is 1536x672). Derived by the model's `resolve_canvas_size`, not settable directly. |
| `num_inference_steps` | **Not accepted.** Sending it at any value -- including 50 -- is a 422. The deployment always runs 50: the AdaLN modulation table is precomputed per step count. |
| Warmup | Nothing is warmed by default; the first request per shape compiles. See section 6. |
| `num_inference_steps` | Omitted is served as 50. An explicit value other than 50 is rejected. |
| One request at a time | One device worker owns the mesh; further requests queue. |
| `t2va` only | `ref2va` needs the `transformer_ref/` partition, and switching task in-process is a 62 GB reload. Separate deployment. |
