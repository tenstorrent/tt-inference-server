# TT-Home Voice Assistant — README

Reference for the TT-Home voice assistant work. Covers the Docker image we built,
the git branch, the app folders, and how to run / debug `start.sh`.

---

## 1. Git repo & branch

- **Repo:** `git@github.com:tenstorrent/tt-inference-server.git`
- **Branch:** `tvardhineni/voice_demo`
- All app folders below live at the repo root.

```bash
git clone -b tvardhineni/voice_demo git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server
```

---

## 2. App folders (what each one is)

| Folder | Hardware | Docker image | Description |
|--------|----------|--------------|-------------|
| `app_wh_glxy_qwen3tts` | Wormhole Galaxy (N150 per model) | `wh-glxy-weights` | WH-Galaxy version, each model on its own N150, with Qwen3-TTS |
| `app_bh_qb2_stable_qwen3-tts` | Blackhole QB2 (4× P150) | `push-button-v2` | **Stable** English version on QB2 with latest Qwen3-TTS |
| `app_bh_qb2_elyza_japan` | Blackhole QB2 (4× P150) | `push-button-v2` | Japanese version (ELYZA-JP-8B) — uses the **large** all-in-one image (weights baked in, ~172 GB) |
| `app_manual_tt-home_jp` | Blackhole QB2 (4× P150) | `manual_jp_tt-home` | Japanese version with **manual weights download** — lean image + weights mounted at runtime |

> Other folders (`app_stable_fix`, `app_stable_20thapr`, `app_21stapr_stable`, `app_bkp_all_modes`,
> `app_phone_control`, `app_wh_glxy_speecht5`) are older stable snapshots / experiments — not the current demo path.

---

## 3. Docker images (GHCR)

Registry path: `ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant`

| Tag | Size | Notes |
|-----|------|-------|
| `push-button` / `push-button-v2` | ~172 GB | Full all-in-one (tt-metal + all weights + caches baked in). Used by `app_bh_qb2_stable_qwen3-tts` and `app_bh_qb2_elyza_japan`. |
| `push-button-old` | ~105 GB | Older full image. |
| `wh-glxy-weights` | — | Wormhole Galaxy image with weights. Used by `app_wh_glxy_qwen3tts`. |
| **`manual_jp_tt-home`** | **~77 GB** | **Lean image for `app_manual_tt-home_jp`.** Contains tt-metal + Python env + packages + ELYZA tensor cache (6.7 GB) + TT JIT kernel cache (1.6 GB). **No model weights, no app code.** |

### Pull access
Granted via the package's **"Manage access"** section on GitHub
(tenstorrent org → Packages → `tt-voice-assistant` → Package settings → Manage access).
Add the person (e.g. `yasuhiroitoTT`) with **Read** role. They pull with a token that has `read:packages`.

---

## 4. How the lean `manual_jp_tt-home` image was created

Three stages:

**Stage 1 — lean tt-metal base (`tt-metal-base`, ~58.7 GB):**
```bash
cd tt-inference-server
docker build -t tt-metal-base --platform=linux/amd64 \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=9542bc34cc738d723de8e69c4d0bc474e8b07baf \
  -f tt-metal-sdxl/Dockerfile .
```
This compiles tt-metal + Python env. No model weights.

**Stage 2 — install runtime packages (`tt-metal-voice-base`, ~61.6 GB):**
Run a container from `tt-metal-base`, install system + Python deps, commit.
```bash
# system deps (as root)
docker exec -u root <container> apt-get update && apt-get install -y ffmpeg
# python deps (venv uses `uv`, not pip)
docker exec <container> bash -lc 'source /home/container_app_user/tt-metal/python_env/bin/activate && \
  uv pip install onnx PyPDF2 requests beautifulsoup4 num2words uvicorn fastapi \
  python-multipart jiwer sacrebleu fugashi mecab-python3 unidic-lite'
docker commit <container> tt-metal-voice-base:latest
```

**Stage 3 — bake in the caches (`manual_jp_tt-home`, ~77 GB):**
Generate caches once (run ELYZA + Qwen3-TTS to produce tensor + JIT kernel caches),
copy them into a fresh container from `tt-metal-voice-base`, then commit.
```bash
docker run -d --name tt-bake --entrypoint /bin/bash tt-metal-voice-base:latest -c "sleep infinity"
# ELYZA tensor cache  -> model_cache
docker cp <elyza_cache>/elyza tt-bake:/home/container_app_user/tt-metal/model_cache/
# TT JIT kernel cache -> ~/.cache/tt-metal-cache
docker cp <jit_cache>/. tt-bake:/home/container_app_user/.cache/tt-metal-cache/
docker commit tt-bake ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:manual_jp_tt-home
docker push   ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:manual_jp_tt-home
```

> Backups of the caches are kept on the host at:
> - `~/tt-home-weights/model_cache/elyza/`  (ELYZA tensor cache, 6.7 GB)
> - `~/tt-home-weights/tt-metal-cache-backup/`  (TT JIT kernel cache, 1.6 GB)

---

## 5. Running `app_manual_tt-home_jp` (the lean setup)

### 5a. Download the 3 model weights (one time, ~43 GB)
```bash
mkdir -p ~/tt-home-weights/hf_cache
pip install -U "huggingface_hub[cli]"
export HF_HUB_CACHE=~/tt-home-weights/hf_cache
hf download elyza/Llama-3-ELYZA-JP-8B
hf download openai/whisper-large-v3
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```
Result:
```
~/tt-home-weights/hf_cache/
├── models--elyza--Llama-3-ELYZA-JP-8B
├── models--openai--whisper-large-v3
└── models--Qwen--Qwen3-TTS-12Hz-1.7B-Base
```

### 5b. Run
```bash
cd tt-inference-server/app_manual_tt-home_jp
bash start.sh
# open http://localhost:8080/
```
First clean start is ~20-60 s (caches are baked into the image, no regeneration).

---

## 6. `start.sh` — reference

Key variables (top of `start.sh`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `IMAGE` | `ghcr.io/.../tt-voice-assistant:manual_jp_tt-home` | Docker image to use |
| `CONTAINER` | `tt-bh-qb2-elyza-jp-manual` | Container name |
| `WEIGHTS_DIR` | `$HOME/tt-home-weights` | Host dir with `hf_cache/` |
| `LLAMA_DEVICES` | `0` | ELYZA-JP-8B chip |
| `WHISPER_DEVICES` | `1` | Whisper large-v3 chip |
| `TTS_DEVICES` | `2` | Qwen3-TTS GUEST (Jim voice) |
| `TTS_HOST_DEVICES` | `3` | Qwen3-TTS HOST (Riata voice, podcast) |

What `start.sh` does (6 steps):
1. Create container if missing (mounts **only** `hf_cache`; maps port 8080).
2. Stop any existing in-container processes, remove stale sockets.
3. Copy app code into the container (`main.py`, `services/`, `servers/`, `templates/`, `static/`).
4. Copy Qwen3-TTS model code (`qwen3_tts_latest/` → tt-metal `models/demos/qwen3_tts/`).
5. Start Whisper + 2× Qwen3-TTS servers (devices 1, 2, 3).
6. Start the main app / ELYZA (device 0) on port 8080.

Override examples:
```bash
WEIGHTS_DIR=/data/weights bash start.sh        # different weights path
IMAGE=some/other:tag bash start.sh             # different image
```

---

## 7. Operations & debugging

### Service status (one-liner)
```bash
docker exec tt-bh-qb2-elyza-jp-manual bash -c '
  test -S /tmp/whisper_server.sock      && echo "Whisper: UP"
  test -S /tmp/tts_server.sock          && echo "TTS-Jim: UP"
  test -S /tmp/tts_server_guest.sock    && echo "TTS-Riata: UP"
  grep -q "All services ready" /tmp/main_app.log && echo "ELYZA: UP"'
```

### Logs (inside container)
| Log | What |
|-----|------|
| `/tmp/main_app.log` | Main app + ELYZA (also ASR/LLM/TTS pipeline activity) |
| `/tmp/whisper_server.log` | Whisper ASR server |
| `/tmp/tts_server.log` | Qwen3-TTS GUEST (Jim) |
| `/tmp/tts_server_guest.log` | Qwen3-TTS HOST (Riata) |

```bash
docker exec tt-bh-qb2-elyza-jp-manual tail -50 /tmp/main_app.log
```

### Stop / restart
```bash
docker stop tt-bh-qb2-elyza-jp-manual   # caches preserved (only `docker rm` loses the JIT cache)
tt-smi -r                               # reset the 4 P150 devices
bash start.sh                           # start again
```

### IMPORTANT — device hangs
- Killing a TTS/Whisper process (SIGKILL / `docker stop`) can leave a TT device in a hung
  state ("failed to initialize FW" / "sysmem mapped at unexpected NOC address").
- **Always check devices are free before launching**, and reset if needed:
  ```bash
  sudo lsof /dev/tenstorrent/*      # should be empty
  tt-smi -r                         # reset if a launch fails with FW/sysmem errors
  ```
- `tt-smi` is an interactive TUI — use `tt-smi -r` (reset) for scripting, not `tt-smi` alone.

---

## 8. Japanese-specific config (in `app_manual_tt-home_jp`)

- **Whisper** is forced to Japanese: `whisper_server.py --model openai/whisper-large-v3 --language ja`.
  Japanese speech transcribes accurately; English speech gets phonetically mapped to Japanese (expected for a JP-only build).
- **ELYZA system prompt** (in `services/llama_service.py`) forces Japanese-only responses, even to English input.
  Note: the original ELYZA-recommended prompt allows English drift on follow-ups; the current prompt is the stricter Japanese-forcing variant.
- **Sentence splitting** in `llama_service.py` includes Japanese punctuation (`。！？`) so long stories aren't sent to TTS as one oversized chunk.
- **Mode prompts** (`services/mode_prompts.py`) and the podcast trigger (`templates/index_chat.html`) are translated to Japanese.
- UI model badge shows **ELYZA-JP-8B**.
