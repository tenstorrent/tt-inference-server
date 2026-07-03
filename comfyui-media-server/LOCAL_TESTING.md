# Running comfyui-media-server locally

Two ways to run it. Path 1 (no Docker) is the fast one — use it to actually test
today. Path 2 (Docker via `run.py`) is the packaging target and takes a long
first build.

## Path 1 — direct, against your host's built tt-metal (fast) ✅

Runs the relocated server straight from this directory, importing `ttnn` /
`models.*` from a tt-metal checkout you already built. No image build.

Requirements: a **built** tt-metal on the amalgamation branch
(`samt/standalone-media-20260703`), `tt-smi` on `PATH`, and model weights
reachable (set `HF_HOME`).

```bash
cd /home/stisi/tt-inference-server/comfyui-media-server
export TT_METAL_HOME=/home/stisi/tt-metal        # your built tt-metal
export HF_HOME=/path/to/hf_cache                  # so weights don't download

# SDXL (single/multi-card):
./launch_server.sh --model sdxl --board p150

# Wan 2.2 T2V (multi-chip):
./launch_server.sh --model wan22 --board p300x2

# add --dev for a faster, no-trace warmup while iterating
```

The server comes up on `http://127.0.0.1:8000` (`/health`, `/image/generations`,
`/latent/denoise`, `/vae/*`, `/video/*`). Stop it with `./stop_server.sh`.

### Drive it from ComfyUI

Since the nodes' auto-launch spawns `<TT_METAL_DIR>/launch_server.sh` (the old
in-tt-metal layout), the clean way to exercise **this** copy is to start it
yourself as above, then point the node at it:

- In **TT Checkpoint Loader**, set `server_url = http://127.0.0.1:8000` — the node
  connects to the already-running server instead of auto-launching one.

(Once `server_manager.py` is updated to drive `run.py --docker-server`, auto-launch
will target this server through Path 2. That change isn't done yet.)

## Path 2 — Docker via `run.py --docker-server` (the target) 🚧

This is what the nodes will eventually call. It needs the image built first.

```bash
# 1. Build the image (LONG — builds tt-metal from source):
./build_image.sh comfyui-media-server:dev

# 2. Launch through run.py in dev mode (requires --override-docker-image):
cd /home/stisi/tt-inference-server
python run.py --model <sdxl|wan> --workflow server \
    --tt-device p300x2 --docker-server --dev-mode \
    --override-docker-image comfyui-media-server:dev
```

`run.py` publishes the service port onto the container's `8000` and injects
`MODEL` / `DEVICE` env; `docker-entrypoint.sh` maps those onto
`launch_server.sh`.

### Known rough edges on Path 2 (not yet proven end-to-end)

- **`MODEL`/`DEVICE` mapping is best-effort.** `run.py` sets `MODEL` to the model
  spec's `model_name`; `docker-entrypoint.sh` normalizes it to `sdxl`/`sd35`/
  `wan22`. If your `--model` resolves to a name the mapping doesn't recognize,
  extend the `case` in `docker-entrypoint.sh`.
- **Name collision with tt-media-server.** `sdxl`/`wan` model names already
  resolve to the OpenAI-API image; `--override-docker-image` forces our image in
  dev, but a permanent path needs a distinct `inference_engine` / model-spec
  entry. See `README.md`.
- **Weights + `/dev/tenstorrent` mounts** are handled by `run.py`'s docker run
  command; make sure your model-spec / setup config points at real weights.
- This path hasn't been run on hardware yet — treat it as a starting point.
