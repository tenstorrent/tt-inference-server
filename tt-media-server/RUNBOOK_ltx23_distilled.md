# Runbook — LTX-2.3 distilled (audio+video) on Galaxy

Serve the LTX-2.3 distilled text→audio-video pipeline through `tt-media-server` on a
Blackhole Galaxy (32 chips). Verified end-to-end: 1080p h264+aac, ~11s served latency (warm).

## Sources
- **tt-metal**: branch `main` @ `3a73746196f` (LTX code is upstream — no fork needed).
- **tt-inference-server**: branch `rsalman/ltx-2.3-distilled-runner`.
- **Python env**: use the media server's own `python_env` (`tt-media-server/python_env`);
  it has `uvicorn`/`fastapi` + `ttnn`. The tt-metal repo's `python_env` does NOT.

## Prereqs (one-time)
1. Build tt-metal: `./build_metal.sh`. If configure fails on a missing submodule
   `CMakeLists.txt`, run `git submodule update --init --force --recursive` first.
2. LTX deps into the media-server env:
   `VIRTUAL_ENV=<...>/tt-media-server/python_env uv pip install --no-deps av==17.0.1`
   (`diffusers==0.38.0` is already present).
3. Weights (gated — `hf auth login`, accept licenses, ~69 GB → HF cache):
   - `Lightricks/LTX-2.3` → `ltx-2.3-22b-distilled-1.1.safetensors`
   - `google/gemma-3-12b-it-qat-q4_0-unquantized`
4. Own every path (owner-writable dirs): `TT_METAL_HOME`, `TT_DIT_CACHE_DIR`,
   `TT_VIDEO_OUTPUT_DIR`. Pointing any at another user's dir → permission errors.

## Serve
```bash
cd <repo>/tt-inference-server/tt-media-server
TT_METAL_HOME=<tt-metal> \
TT_DIT_CACHE_DIR=<owned>/tt_dit_cache \
TT_VIDEO_OUTPUT_DIR=<owned>/tt-media-videos \
LTX_YUV_EXPORT=1 \
DEVICE=galaxy MODEL=LTX-2.3-distilled \
./python_env/bin/uvicorn --host 0.0.0.0 main:app --lifespan on --port 8082
```
First run does a traced warmup (one full 1080p gen) before `/health` → 200; ~10 min cold,
faster once the DIT cache + kernels are on disk.

## Generate (async; needs API key)
```bash
AUTH='Authorization: Bearer your-secret-key'   # default; override with API_KEY, or NO_AUTH=1
JOB=$(curl -s -X POST localhost:8082/v1/videos/generations -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{"prompt":"...","seed":10}' | jq -r .id)
curl -s -H "$AUTH" localhost:8082/v1/videos/generations/$JOB          # poll until "completed"
curl -s -H "$AUTH" -o out.mp4 localhost:8082/v1/videos/generations/$JOB/download
```
`/health` and `/v1/models` need no key; `/generations` does. Browser UI: `/docs`.

## Gotchas
- **Only one process owns the chips** (`CHIP_IN_USE` lock) — stop any other server first.
- **Mesh param**: this Galaxy has **2 links**, so use the ring `(4,8)` config; the runner
  hardcodes `l1_small_size=32768` (without it the audio vocoder OOMs: `bank size is 0 B`).
- **ModelConfigs key**: `DEVICE=galaxy` → `DeviceTypes.GALAXY` (NOT `BLACKHOLE_GALAXY`);
  wrong key silently falls back to the SDXL default runner.
- **DIT weight cache** ignores topology in its key — a cache built under a different
  topology loads as a false hit (`LoadingError: shape mismatch`). Clear
  `$TT_DIT_CACHE_DIR/ltx-2.3-*` to regenerate.

## Optional: on-device sanity test (no server)
```bash
cd <tt-metal>
TT_METAL_HOME=<tt-metal> TT_DIT_CACHE_DIR=<owned>/tt_dit_cache \
OUTPUT_PATH=<owned>/ltx_test.mp4 LTX_YUV_EXPORT=1 RUN_VBENCH=0 RUN_CLIP=0 \
NUM_FRAMES=145 HEIGHT=1088 WIDTH=1920 LTX_TRACED=1 \
./tt-inference-server/tt-media-server/python_env/bin/python3 -m pytest -svq --timeout=0 \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
  -k "4x8sp1tp0nl2_ring_is_fsdp0"
```

## Perf (1080p, 145f, traced, steady-state)
~6–7s compute/gen (`LTX_YUV_EXPORT=1` ~13% faster). Audio decode dominates the untraced path.

## Implementation (this branch)
`config/constants.py` (LTX `ModelNames`/`SupportedModels`/`ModelRunners` + video-service &
`ModelConfigs` `(TT_LTX_2_3_DISTILLED, GALAXY)` entries), `tt_model_runners/dit_runners.py`
(`TTLTX23DistilledRunner`), `tt_model_runners/runner_fabric.py` (dispatch),
`utils/video_manager.py` (`TT_VIDEO_OUTPUT_DIR` override). Serving is text→AV: `run()` writes
the MP4 and returns its path; `VideoService.post_process` passes the path straight to `FileResponse`.
