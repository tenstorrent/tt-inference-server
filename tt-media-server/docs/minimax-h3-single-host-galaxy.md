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
git checkout 9c96923d1bb  # 'Add Minimax H3 support (#52874)', the squashed merge on main
./build_metal.sh --build-all
```

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

### diffusers must be overridden (required)

tt-metal pins `diffusers==0.38.0`, which does **not** contain MiniMax-H3. The pipeline imports
`diffusers.modular_pipelines.minimax_h3` and `diffusers.models.transformers.transformer_minimax_h3`,
so a stock env fails at import. Install the dev build this was brought up against:

```bash
uv pip install --force-reinstall \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

Check it:

```bash
python -c "import diffusers, importlib.util; print(diffusers.__version__, \
  importlib.util.find_spec('diffusers.modular_pipelines.minimax_h3') is not None)"
# 0.40.0.dev0 True
```

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
export DEVICE="galaxy"
export ARCH_NAME=blackhole
export MESH_DEVICE="(4, 8)"

export MINIMAX_H3_MODEL_PATH=/path_to/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/path_to/tt_dit_cache

export USE_ASYNC_VIDEO=true
```

Three of these bite if wrong:

| var | what happens if it is wrong |
|---|---|
| `TT_METAL_HOME` | unset -> the kernel cache falls back to root-owned `/built` and mesh init dies as *"Failed to generate binaries for fabric_erisc_router ... Permission denied"* |
| `TT_DIT_CACHE_DIR` | unset -> **degrades silently**, ~713 s instead of ~64 s |
| `MINIMAX_H3_MODEL_PATH` | the single weights lever. Unset -> nothing finds the snapshot, and the model-side gates *skip* rather than fail, so a run can look clean while testing nothing. (`MINIMAX_H3_DIFFUSERS_DIR` was folded into this one and is no longer read.) |
| `USE_ASYNC_VIDEO` | false -> the request blocks for ~70 s and clients time out |

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
ffprobe out.mp4     # expect h264 1344x768 24 fps + aac stereo 32000 Hz, 5.17 s
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

Warm compute per request, measured 2026-08-13 on `9c96923d1bb`. Two things the numbers say: cost
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
| `duration_seconds` | `5`, `10` or `15`. Anything else is a 422. Only these land on a whole `17n + 5` frame count (124 / 243 / 362) without rounding a request into a different shape. Omitted gives `5`. |
| Resolution | 768P throughout: short edge 768 from 16:9 to 9:16, area capped at ~1 MPix for wider (21:9 is 1536x672). Derived by the model's `resolve_canvas_size`, not settable directly. |
| `num_inference_steps` | **Not accepted.** Sending it at any value -- including 50 -- is a 422. The deployment always runs 50: the AdaLN modulation table is precomputed per step count. |
| Warmup | Nothing is warmed by default; the first request per shape compiles. See section 6. |
| `num_inference_steps` | Omitted is served as 50. An explicit value other than 50 is rejected. |
| One request at a time | One device worker owns the mesh; further requests queue. |
| `t2va` only | `ref2va` needs the `transformer_ref/` partition, and switching task in-process is a 62 GB reload. Separate deployment. |
