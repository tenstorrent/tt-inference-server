# TT-HOME Voice Assistant - Wormhole Galaxy (Qwen3 TTS)

## Status: IN PROGRESS - Waiting for Qwen3 TTS TTNN model

## Architecture (planned)
- **Docker**: TBD (may need newer tt-metal with Qwen3 TTS support)
- **Hardware**: Wormhole Galaxy (32 N150 chips, 4 T3Ks)

## Device Allocation (planned)
| Model | Device | Hardware | Notes |
|-------|--------|----------|-------|
| Llama 3.1-8B | 0 | N150 | LLM |
| Whisper distil-large-v3 | 2 | N150 | STT, traced |
| Qwen3 TTS HOST | 4 | N150 | TBD - voice 1 |
| Qwen3 TTS GUEST | 6 | N150 | TBD - voice 2 |

## TODO
- [ ] Get Qwen3 TTS TTNN model code
- [ ] Create qwen3_tts_server.py (socket-based, same pattern as speecht5_ttnn_server.py)
- [ ] Update tts_service_socket.py if API differs
- [ ] Update start.sh with Qwen3 server launch commands
- [ ] Test multi-speaker (host/guest) support
- [ ] Determine if traces are supported for Qwen3 TTS

## Notes
- Qwen3 TTS supports zero-shot voice cloning (provide reference audio)
- Can use different reference audio clips for HOST vs GUEST voices
- Should have higher quality and more natural speech than SpeechT5
- Base code copied from app_wh_glxy_speecht5 - swap TTS server/service when ready
