# Runbook — Wan2.2 T2V on Galaxy (tt-media-server)

Formalizes the process for standing up the `tt-media-server` inference server for
`Wan2.2-T2V-A14B-Diffusers` on a Galaxy (Blackhole, 32-chip) host. Based on Sam's
original steps, refined with the fixes we hit in practice.

> **Status:** draft. Paths below use the `rsalman` account on `g03blx01` as the
> worked example — swap in your own user/paths where noted (`<...>`).

---

## 0. Key principle (why most of the pain happened)

The server must run entirely out of **paths your own user can read and write**.
Every failure we hit was a borrowed, another-user-owned path. When adapting this
runbook, make sure all four of these point somewhere *you* own:

| Purpose | Env var / path | Symptom if it points at someone else's dir |
|---|---|---|
| tt-metal install (kernel build cache) | `TT_METAL_HOME` | `Permission denied [/built/...]` on device init |
| DIT model cache | `TT_DIT_CACHE_DIR` | `PermissionError: .../cache_dict.json` |
| Video output | `TT_VIDEO_OUTPUT_DIR` | FFmpeg `Permission denied` writing `.mp4` |
| Python env | `./python_env` (media-server's own) | `uvicorn: command not found` / missing `fastapi` |

---

## 1. Build tt-metal

```bash
cd <TT_METAL_HOME>          # e.g. /home/rsalman/tt-metal
./build_metal.sh
```

**Gotcha — drifted submodules.** If configure fails with
`does not contain a CMakeLists.txt file` (e.g. under `third_party/tracy`), the
`tracy`/`umd` submodules have drifted off their pinned commits. Reset them:

```bash
git submodule update --init --force --recursive \
    tt_metal/third_party/tracy tt_metal/third_party/umd
```

Then re-run `./build_metal.sh`.

---

## 2. One-time environment prep

```bash
# Directories you own for the two caches + video output
mkdir -p <TT_METAL_HOME>/tt_dit_cache
mkdir -p <HOME>/tt-media-videos     # e.g. /home/rsalman/tt-media-videos
```

**Local patch required** — `utils/video_manager.py` hardcodes the output dir to
`/tmp/videos`. Make it honor `TT_VIDEO_OUTPUT_DIR`:

```python
# utils/video_manager.py
_VIDEO_OUTPUT_DIR = Path(os.environ.get("TT_VIDEO_OUTPUT_DIR", "/tmp/videos"))
```

> TODO: upstream this so the env var is supported by default (currently a local edit).

---

## 3. Make sure the chips are free

Only **one** process can hold the devices at a time (UMD `CHIP_IN_USE_0_PCIe`
lock). If startup logs
`Waiting for lock 'CHIP_IN_USE_0_PCIe' ... held by ... PID: <pid>`, another server
already owns the cluster. Find it and stop it before launching:

```bash
ps -eo pid,user,etime,cmd | grep -iE "uvicorn|main:app" | grep -v grep
# gracefully stop the holder (coordinate with its owner first):
kill -TERM <master_pid>          # sudo only if it's another user's process
```

Stale `/dev/shm/tt_device_*_memory` objects from a prior run produce
`Failed to create shared memory ... Permission denied` **warnings** — harmless,
ignore them.

---

## 4. Launch

Run from the media-server dir, using **its own `python_env`** (the tt-metal repo's
`python_env` has `ttnn` but NOT `uvicorn`/`fastapi`, so `run_uvicorn.sh` will fail
here — invoke uvicorn directly instead):

```bash
cd <TT_METAL_HOME>/tt-inference-server/tt-media-server

TT_METAL_HOME=<TT_METAL_HOME> \
TT_DIT_CACHE_DIR=<TT_METAL_HOME>/tt_dit_cache \
TT_VIDEO_OUTPUT_DIR=<HOME>/tt-media-videos \
DEVICE=galaxy MODEL=Wan2.2-T2V-A14B-Diffusers \
./python_env/bin/uvicorn --host 0.0.0.0 main:app --lifespan on --port 8081
```

**First launch is slow (~10 min cold start):** it downloads/loads the HF weights,
regenerates the DIT cache (`Cache does not exist. Loading PyTorch state dict.` →
`Writing cache to ...`, ~4 min, ~66 GB across transformer/transformer_2/
text_encoder/vae), then JIT-compiles the device kernels. Both caches persist, so
**subsequent starts are much faster.**

---

## 5. Verify it's up

Readiness is gated on the device worker, not just the HTTP layer. Wait for:

```
Model loaded successfully
Model warmup completed
Worker (...) reported ready ; flipping /health to 200
All workers ready
```

Then:

```bash
curl -s http://127.0.0.1:8081/health              # -> 200
curl -s http://127.0.0.1:8081/v1/models           # lists the Wan2.2 model
```

Browser UI (Swagger): **http://<host>:8081/docs**  (`/redoc` also works; `/` is 404).
From a laptop, tunnel first:

```bash
ssh -L 8081:localhost:8081 <user>@<host>
# then open http://localhost:8081/docs
```

---

## 6. Where the videos land

Each generation is written to **`$TT_VIDEO_OUTPUT_DIR/<uuid>.mp4`**
(e.g. `/home/rsalman/tt-media-videos/`).

- **Sync mode:** the `POST /v1/videos/...` endpoint returns the MP4 directly as a
  file download (`Content-Disposition: attachment`).
- **Async mode:** `GET /v1/videos/generations/{job_id}/download` streams the MP4.

Copy one to your laptop:

```bash
scp <user>@<host>:<HOME>/tt-media-videos/<uuid>.mp4 .
```

---

## Appendix — worked example values (g03blx01 / rsalman)

| Variable | Value |
|---|---|
| Host | `g03blx01` (LAN `172.27.29.139`) |
| `TT_METAL_HOME` | `/home/rsalman/tt-metal` |
| `TT_DIT_CACHE_DIR` | `/home/rsalman/tt-metal/tt_dit_cache` |
| `TT_VIDEO_OUTPUT_DIR` | `/home/rsalman/tt-media-videos` |
| Python env | `.../tt-inference-server/tt-media-server/python_env` |
| Port | `8081` |
| Server log (this session) | `/tmp/rsalman_media_8081.log` |
