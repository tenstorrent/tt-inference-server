# MiniMax-H3 FL2VA endpoint benchmark

How to measure the end-to-end latency of MiniMax-H3 FL2VA (first/last keyframe + prompt to video with audio) on
one Blackhole Galaxy through tt-media-server, and the numbers measured with this commit.

The client is [`scripts/minimax_h3_fl2va_bench.py`](../scripts/minimax_h3_fl2va_bench.py). It is one file, needs
only Python 3.9+, and generates its own keyframe. It sends one request at a time (one warmup, then N timed
requests for each of 5, 10 and 15 s clips). Each MP4 is checked for a video track, an audio track and the
requested length. The script reports per-duration latency and writes `summary.md`, `summary.json` and
`results.csv`.

## 1. Launch the server

Bare metal, on the Galaxy host. Tested with tt-metal `c2053e93d6f33cbfa37a83031a8c10d87ac33cd6` and this repo at
`acc7ff255a77e134e29bd25892f44b241bad3841`. Prerequisites:

- tt-metal built (`./build_metal.sh`) with its venv (`./create_venv.sh`).
- `uv pip install -r tt-media-server/requirements.txt` in that venv.
- `torchaudio==2.11.0+cpu` from `https://download.pytorch.org/whl/cpu` (it must match the CPU torch).
- `imageio-ffmpeg`.
- `ffmpeg` on `PATH`. Without a system ffmpeg, symlink the `imageio_ffmpeg` binary.
- Blackhole firmware 19.11.0 or newer, so that `TT_METAL_TDP_LIMIT_WATTS` takes effect.

```bash
export TT_METAL_HOME=/path/to/tt-metal
source $TT_METAL_HOME/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
export TT_METAL_TDP_LIMIT_WATTS=160          # per-chip power cap; check for 32 "is now 160 W" log lines
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120
export TT_METAL_DISABLE_XIP_DUMP=1 TT_METAL_SHM_TRACKING_DISABLED=1 TT_METAL_INSPECTOR=0
export MODEL=MiniMax-H3-FL2VA MODEL_RUNNER=tt-minimax-h3-fl2va   # base model, 50 steps; also serves text-only
export DEVICE=galaxy ARCH_NAME=blackhole MESH_DEVICE="(4, 8)"
export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers       # hf download MiniMaxAI/MiniMax-H3 (see specs)
export TT_DIT_CACHE_DIR=/path/to/persistent/tt_dit_cache
export USE_ASYNC_VIDEO=false                 # sync: the POST returns the MP4 (the script also handles async)
export API_KEY=<choose-a-key>
cd tt-media-server && ./run_uvicorn.sh --skip-venv
```

On a cold start with an empty `TT_DIT_CACHE_DIR`, the server took **43 min** until `/tt-liveness` returned 200: it
converts the weights into the cache, compiles the kernels and warms all 26 denoise buckets. The power cap stays set
after the process exits, until the chips are reset.

## 2. Run the benchmark

```bash
export H3_URL=http://<host>:8000 H3_API_KEY=<key>
python3 scripts/minimax_h3_fl2va_bench.py --wait-ready 7200 --save-videos
```

Options: `--runs N`, `--durations 5,10`, `--first-frame IMG`, `--last-frame IMG`, `--first-and-last`,
`--aspect-ratio`, `--seed`, `--prompt`/`--prompt-file`. It exits 1 if any timed request fails, and stops at the first
request if the key is rejected.

## 3. Results (2026-10-13, one Blackhole Galaxy, 160 W)

- Server: FL2VA base (50 steps), sync mode, firmware 19.12.0, `TT_METAL_TDP_LIMIT_WATTS=160`.
- Client: on the same host; generated 1344x768 first keyframe, 16:9, seed 42.
- Each duration: 1 warmup + 3 timed requests. All 12 passed.

| Clip | Runs OK | E2E median (s) | E2E min - max (s) | Server generation median (s) |
|---|---|---|---|---|
| 5 s | 3/3 | **53.61** | 53.51 - 53.69 | 53.58 |
| 10 s | 3/3 | **123.86** | 123.74 - 123.87 | 123.80 |
| 15 s | 3/3 | **244.46** | 244.16 - 244.51 | 244.40 |

E2E is from the POST being sent to the last MP4 byte being received. Server generation is the `X-Generation-Time`
header. MP4 lengths were 5.17, 10.13 and 15.08 s, all with video and audio tracks.

Per-stage breakdown of the same timed requests, from the server log (`denoise breakdown` and `VAE decode profile`
lines). Program-cache misses were 0 for every stage:

| Clip | Packed seq -> bucket (incl. 1008 keyframe rows) | Steady ms/step | Denoise total | VAE decode |
|---|---|---|---|---|
| 5 s | 39787 -> 41984 | 1053 - 1056 | ~51.9 s | 0.75 s |
| 10 s | 75463 -> 76800 | 2465 - 2466 | ~121.0 s | 1.41 - 1.42 s |
| 15 s | 111139 -> 111616 | 4902 - 4907 | ~240.6 s | 2.05 - 2.12 s |

Denoising (49 forwards) is 97-99% of the request. With a client on the same host, E2E is within 0.1 s of the
server's own generation time.
