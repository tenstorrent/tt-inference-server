# TODO - app_bh_qb2_stable_qwen3-tts

## Next Steps

### 1. Japanese LLM - Replace Llama 3.1 with ELYZA JP
- **Model**: `HF_MODEL=elyza/Llama-3-ELYZA-JP-8B`
- **Weights**: Already downloaded in Docker HF cache (15GB)
- **Change**: Update `start.sh` → `export HF_MODEL=elyza/Llama-3-ELYZA-JP-8B`
- **Architecture**: Same as Llama 3.1-8B (LlamaForCausalLM, 32 layers, 4096 hidden, 8 KV heads, vocab 128256) — drop-in replacement
- **Note**: RoPE scaling differs (Llama 3 base vs 3.1 extended) but tt-metal reads config from HF automatically
- **Do this in a separate folder** based on stable Task 1

### 2. Live Translation (JP <-> EN)
- **Whisper model**: `openai/whisper-large-v3` (already cached in Docker, 2.9GB)
- Supports `translate` task (any language → English) natively
- `distil-whisper/distil-large-v3` does NOT support translation well
- Start with manual translation, then automate

### 3. Docker Image Commit
- Container `tt-metal-rebuild` has all caches ready:
  - tt-metal kernel cache: 1.7GB (Llama + Whisper + Qwen3-TTS)
  - HF weights: Llama 3.1, ELYZA-JP, Qwen3-TTS, Whisper distil-large-v3, Whisper large-v3
  - ffmpeg installed
- **Commit only after full TT-Home end-to-end test passes**

### 4. Other Japanese Models (from manager - verify names)
- Llama-3-Daiat-8B — not found on HuggingFace, verify exact repo name
- Llama-3-Yomogi-8B — not found on HuggingFace, verify exact repo name
- Llama-3-Elysium-8B — not found on HuggingFace, verify exact repo name
