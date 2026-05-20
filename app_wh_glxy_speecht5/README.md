# TT-HOME Voice Assistant - Wormhole Galaxy (SpeechT5)

## Quick Start
```bash
ssh -L 8080:localhost:8080 -p 32101 tt-admin@208.184.210.227
cd ~/teja/tt-inference-server/app_wh_glxy_speecht5
source ~/teja/tt-metal/python_env/bin/activate && tt-smi -r
./start.sh
# Wait for "ALL SERVICES READY!" then open http://localhost:8080/
```

## Architecture
- **Docker**: `ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:push-button`
- **Container**: `tt-wh-glxy-v2`
- **Hardware**: Wormhole Galaxy (32 N150 chips, 4 T3Ks)

## Device Allocation
| Model | Device | Hardware | Notes |
|-------|--------|----------|-------|
| Llama 3.1-8B | 0 | N150 | LLM, fabric disabled |
| Whisper distil-large-v3 | 2 | N150 | STT, traced |
| SpeechT5 TTS HOST | 4 | N150 | Traced, speaker 7306 (female) |
| SpeechT5 TTS GUEST | 6 | N150 | Traced, speaker 1138 (male) |

## Key Design Decisions
- Uses **push-button Docker** (older tt-metal) for stable SpeechT5 traces
- **Podcast mode**: CPU fallback with speaker_id for reliability
- Markdown stripped before TTS (SpeechT5 can't speak formatting)
- Sentence flush thresholds reduced (8/60 chars) for faster streaming

## Troubleshooting
```bash
sg docker -c "docker exec tt-wh-glxy-v2 tail -30 /tmp/main_app.log"
sg docker -c "docker exec tt-wh-glxy-v2 tail -30 /tmp/tts_server.log"
sg docker -c "docker exec tt-wh-glxy-v2 tail -30 /tmp/tts_server_guest.log"
sg docker -c "docker exec tt-wh-glxy-v2 tail -30 /tmp/whisper_server.log"
```
