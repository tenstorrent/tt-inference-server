# Running MiniMax-H3 `t2va` on a single-host Galaxy through the inference server

Text in, a 5.17 s video **with its own soundtrack** out. 1344x768, 124 frames @ 24 fps.

Single host, 4x8 Blackhole Galaxy (32 chips). There is **no `tt-run`, no MPI, no rankfile and no
`sp_runner`** — that machinery is for multi-host deployments. This is one uvicorn process talking to
one mesh.

## 1. Build tt-metal

```bash
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git checkout 759d8db199f6c57b6c32db682a68a13ee2b0b013
./build_metal.sh --build-all
```

## 2. Clone the inference server

```bash
git clone git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server
git checkout 5b1a3bbde56af30b8ffd4ac3a892ea793939bb91
```

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

export MINIMAX_H3_DIFFUSERS_DIR=/path_to/MiniMax-H3-diffusers
export MINIMAX_H3_MODEL_PATH=/path_to/MiniMax-H3-diffusers
export TT_DIT_CACHE_DIR=/path_to/tt_dit_cache

export USE_ASYNC_VIDEO=true
```

Four of these bite if wrong:

| var | what happens if it is wrong |
|---|---|
| `TT_METAL_HOME` | unset -> the kernel cache falls back to root-owned `/built` and mesh init dies as *"Failed to generate binaries for fabric_erisc_router ... Permission denied"* |
| `TT_DIT_CACHE_DIR` | unset -> **degrades silently**, ~713 s instead of ~64 s |
| `MINIMAX_H3_DIFFUSERS_DIR` / `MINIMAX_H3_MODEL_PATH` | must be the **same snapshot**; different paths made a whole gate silently skip |
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

## 6. Wait for warmup

Weight load plus one full 50-step warmup generation. **Expect ~10 minutes on a warm cache, and
20-30 minutes the very first time**, while the ttnn cache and AdaLN table are built. Readiness means warm:
serving before this would make the first request pay full compilation (~210 s instead of ~73 s).

```
Device 0,...,31: Loading MiniMax-H3...
Device 0,...,31: Model loaded, warming at the serving shape...
Device 0,...,31: warmup pass 1/1 done
Device 0,...,31: warm at padded_len=37888 (1344x768, 124 frames, 50 steps)
```

```bash
curl -s localhost:8000/tt-liveness      # {"status":"alive","model_ready":true,...}
```

## 7. Generate

`API_KEY` defaults to `your-secret-key`.

```bash
# submit -> 202 + job id
curl -X POST localhost:8000/v1/videos/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -d '{"prompt":"A red fox steps through wet grass at dawn.","seed":0}'

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

## What to expect

Steady-state, per request:

| stage | s |
|---|---|
| Encoder (novel prompt) | 1.6 |
| Denoise (49 forwards) | 52.0 |
| VAE decode | 6.5 |
| Audio decode | 1.6 |
| **Total (compute)** | **60.5** |

The **first request after warmup is ~12 s slower** — a first-step transient that settles from ~12 s
to ~1.1 s and then stays there. A second warmup pass does not absorb it. Cause not yet identified.

A repeated prompt is faster than a novel one: prompt embeddings are cached on disk, so the Encoder
row drops to 0.0 s.

## Constraints

| constraint | detail |
|---|---|
| One shape | 1344x768, 124 frames, 50 steps. Anything else is rejected rather than silently recompiled. |
| `num_frames` | Must satisfy `17n + 5`: 124, 243, 362. |
| `num_inference_steps` | Omitted is served as 50. An explicit value other than 50 is rejected. |
| One request at a time | One device worker owns the mesh; further requests queue. |
| `t2va` only | `ref2va` needs the `transformer_ref/` partition, and switching task in-process is a 62 GB reload. Separate deployment. |
