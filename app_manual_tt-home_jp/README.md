# TT-HOME Voice Assistant — Japanese (ELYZA-JP-8B)

Blackhole QB2 version with full Japanese voice assistant experience.

## Architecture

| Device | Model | Purpose |
|--------|-------|---------|
| P150 Device 0 | ELYZA-JP-8B | Japanese LLM (fine-tuned Llama 3) |
| P150 Device 1 | Whisper large-v3 | Japanese speech-to-text |
| P150 Device 2 | Qwen3-TTS (Jim) | Text-to-speech (guest voice) |
| P150 Device 3 | Qwen3-TTS (Riata) | Text-to-speech (host voice) |

## Quick Start

```bash
# 1. Make sure devices are healthy
tt-smi -r

# 2. Launch (all 4 services start automatically)
cd app_bh_qb2_elyza_japan
./start.sh

# 3. Open browser
# http://localhost:8080/
```

Startup takes ~20 seconds (all caches pre-baked in Docker image).

## Docker Image

```
ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:push-button-v2
```

Pre-baked caches:
- ELYZA-JP-8B tensor cache (`tt-metal/model_cache/elyza/`)
- Llama 3.1-8B tensor cache (`tt-metal/model_cache/meta-llama/`)
- Whisper large-v3 + distil-large-v3 HF cache
- Qwen3-TTS model weights
- ffmpeg, jiwer, sacrebleu, fugashi, mecab-python3, unidic-lite pre-installed

## Key Differences from English Version

| Component | English (`app_bh_qb2_stable_qwen3-tts`) | Japanese (`app_bh_qb2_elyza_japan`) |
|-----------|------------------------------------------|-------------------------------------|
| LLM | Llama 3.1-8B-Instruct | ELYZA-JP-8B |
| Whisper | distil-large-v3 (en) | large-v3 (ja) |
| Container | tt-bh-qb2-qwen3 | tt-bh-qb2-elyza-jp |

## Testing Japanese Voice Input

Speak these phrases into the microphone:

- **こんにちは** (Konnichiwa) — Hello
- **おはよう** (Ohayo) — Good morning
- **ありがとう** (Arigato) — Thank you
- **東京のおすすめの場所を教えて** — Tell me recommended places in Tokyo
- **人工知能とは何ですか** — What is artificial intelligence?
- **日本の有名な食べ物を教えてください** — Tell me about famous Japanese food

## Logs

```bash
CONTAINER=tt-bh-qb2-elyza-jp
docker exec $CONTAINER tail -20 /tmp/main_app.log           # Main app + Llama
docker exec $CONTAINER tail -20 /tmp/whisper_server.log      # Whisper ASR
docker exec $CONTAINER tail -20 /tmp/tts_server.log          # TTS Jim
docker exec $CONTAINER tail -20 /tmp/tts_server_guest.log    # TTS Riata
```

## Troubleshooting

- **Device errors**: Run `tt-smi -r` to reset, then re-run `./start.sh`
- **Cache regenerating**: Ensure using the correct Docker image with pre-baked caches
- **Port 8080 in use**: Stop other containers first: `docker stop <container>`
