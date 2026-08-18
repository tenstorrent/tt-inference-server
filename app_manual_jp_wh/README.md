# TT-HOME Voice Assistant — Japanese on Wormhole Galaxy (ELYZA-JP-8B)

Copy of `app_manual_tt-home_jp`, retargeted from Blackhole P150 to Galaxy **N150**.

## Architecture

Each model is one isolated Wormhole ASIC (`TT_VISIBLE_DEVICES` + N150 8×8 mesh). On Galaxy we skip odd chips so each process is a true 1-chip N150, matching `app_wh_glxy_qwen3tts`.

| Device | Model | Purpose |
|--------|-------|---------|
| N150 Device 0 | ELYZA-JP-8B | Japanese LLM |
| N150 Device 2 | Whisper large-v3 | Japanese speech-to-text |
| N150 Device 4 | Qwen3-TTS (Jim) | Guest voice |
| N150 Device 6 | Qwen3-TTS (Riata) | Host voice (podcast) |

## Prerequisites

1. Docker image:
   ```
   ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:manual_jp_wh
   ```
   Wormhole JIT/kernel caches for ELYZA, Whisper large-v3, and both Qwen3-TTS voices are in this tag (~1.8 GB). HF weights are still mounted from the host (below). Override with `IMAGE=...`.
2. Hugging Face weights at `~/tt-home-weights/hf_cache/` (see below).

## Download weights (one time, ~43 GB)

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

## Quick Start

Need docker group access (`newgrp docker` if `docker ps` fails).

```bash
# 1. Devices idle
tt-smi -r   # or Galaxy: tt-smi -glx_reset_auto if wedged

# 2. Launch
cd /home/tt-admin/teja/tt-inference-server/app_manual_jp_wh
./start.sh

# 3. Browser
# http://localhost:8080/
```

First start should be fast: Wormhole JIT for all four models is already in the image. Weights still come from the host mount.

## Docker / env overrides

```bash
IMAGE=ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:manual_jp_wh bash start.sh
WEIGHTS_DIR=/data/weights bash start.sh
```

Default container name: `tt-wh-glxy-elyza-jp-manual`.

## Logs

```bash
CONTAINER=tt-wh-glxy-elyza-jp-manual
docker exec $CONTAINER tail -20 /tmp/main_app.log           # Main app + ELYZA
docker exec $CONTAINER tail -20 /tmp/whisper_server.log      # Whisper ASR
docker exec $CONTAINER tail -20 /tmp/tts_server.log          # TTS Jim
docker exec $CONTAINER tail -20 /tmp/tts_server_guest.log    # TTS Riata
```

## Testing Japanese Voice Input

- **こんにちは** (Konnichiwa) — Hello
- **おはよう** (Ohayo) — Good morning
- **ありがとう** (Arigato) — Thank you
- **東京のおすすめの場所を教えて** — Tell me recommended places in Tokyo
- **人工知能とは何ですか** — What is artificial intelligence?
- **日本の有名な食べ物を教えてください** — Tell me about famous Japanese food

## Troubleshooting

- **Device errors**: reset (`tt-smi -r` / Galaxy `tt-smi -glx_reset_auto`), then re-run `./start.sh`
- **Port 8080 in use**: `docker stop <container>`
- **Do not** use the Blackhole image `manual_jp_tt-home` on this Galaxy box. This image is Wormhole-only.
