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
2. LTX deps into the media-server env: `av==17.0.1` is declared in `requirements.txt`,
   so a normal install covers it. `diffusers==0.38.0` must be installed separately --
   it declares `safetensors>=0.8.0rc0` but never version-gates it at import and uses
   only stable API, so `--no-deps` keeps the env's stable safetensors:
   `VIRTUAL_ENV=<...>/tt-media-server/python_env uv pip install --no-deps diffusers==0.38.0`
3. Weights (gated — `hf auth login`, accept licenses, ~69 GB → HF cache):
   - `Lightricks/LTX-2.3` → `ltx-2.3-22b-distilled-1.1.safetensors`
   - `google/gemma-3-12b-it-qat-q4_0-unquantized`
4. Own every path (owner-writable dirs): `TT_METAL_HOME`, `TT_DIT_CACHE_DIR`,
   `TT_VIDEO_OUTPUT_DIR`. Pointing any at another user's dir → permission errors.

## Serve

### Full-time hosting (systemd) — how this box actually runs
`start_ltx_fast.sh` bakes in every env var the server needs, so a restart is
reproducible and `SERVED_MODEL_NAME` survives it. `deploy/` holds the units:

```bash
sudo cp deploy/ltx-media-server*.service deploy/ltx-media-server-health.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ltx-media-server.service ltx-media-server-health.timer
```

Serves on **:8000**. `ltx-media-server.service` restarts on failure, with a circuit
breaker (5 starts / 30 min) so a device needing a manual `tt-smi -r` can't restart-loop.
Logs append to `/home/rsalman/ltx-media-server.log`.

The health timer exists because `CANARY_GATE_READINESS=true` makes a wedged device return
non-200 **while the process stays alive** — which `Restart=on-failure` cannot observe.
`ltx_healthcheck.sh` polls every 60s and restarts past a 900s warmup grace.

> The unit files hardcode `/home/rsalman/…` paths and `User=rsalman`. Adjust both, and the
> `TT_*` paths in `start_ltx_fast.sh`, before deploying on another host.

Watch a (re)start with `journalctl -fu ltx-media-server` or `tail -f` on the log above.

**Job records survive restarts.** `ENABLE_JOB_PERSISTENCE=true` with an absolute
`JOB_DATABASE_PATH` keeps jobs in SQLite; on boot the server logs
`Restored N job(s) from database`. Without it jobs live only in memory, so every
restart turns every outstanding job id into a permanent 404 and clients poll dead
ids forever. Note this also makes `job_retention_seconds` (24h) effective: completed
jobs and their MP4s are now actually deleted on schedule.

### Manual / one-off
```bash
cd <repo>/tt-inference-server/tt-media-server
TT_METAL_HOME=<tt-metal> \
TT_DIT_CACHE_DIR=<owned>/tt_dit_cache \
TT_VIDEO_OUTPUT_DIR=<owned>/tt-media-videos \
LTX_YUV_EXPORT=1 \
DEVICE=galaxy MODEL=LTX-2.3-distilled \
./python_env/bin/uvicorn --host 0.0.0.0 main:app --lifespan on --port 8082
```

Either way, startup does a traced warmup (one full 1080p gen) before `/health` → 200;
~10 min cold, faster once the DIT cache + kernels are on disk. That warmup gen's MP4 is
discarded rather than left in `TT_VIDEO_OUTPUT_DIR` (see `_discard_warmup_output`).

## Generate (async; needs API key)
```bash
AUTH='Authorization: Bearer your-secret-key'   # default; override with API_KEY, or NO_AUTH=1
PORT=8000                                      # 8082 if started via the manual command above
JOB=$(curl -s -X POST localhost:$PORT/v1/videos/generations -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{"prompt":"...","seed":10}' | jq -r .id)
curl -s -H "$AUTH" localhost:$PORT/v1/videos/generations/$JOB          # poll until "completed"
curl -s -H "$AUTH" -o out.mp4 localhost:$PORT/v1/videos/generations/$JOB/download
```
`/health` and `/v1/models` need no key; `/generations` does. The browser UI at
`/docs` (plus `/redoc`, `/openapi.json`) is **off** under the default
`ENVIRONMENT=production`; start with `ENVIRONMENT=development` to enable it.

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
`utils/video_manager.py` (`TT_VIDEO_OUTPUT_DIR` override), `start_ltx_fast.sh` +
`ltx_healthcheck.sh` + `deploy/` (systemd hosting), and `SERVED_MODEL_NAME` support in
`open_ai_api/models.py` / `model_services/base_job_service.py`. Serving is text→AV: `run()` writes
the MP4 and returns its path; `VideoService.post_process` passes the path straight to `FileResponse`.
