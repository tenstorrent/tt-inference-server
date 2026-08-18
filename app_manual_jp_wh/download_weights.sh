#!/bin/bash
# Download TT-Home JP weights into ~/tt-home-weights/hf_cache
# Always use a full `hf download` of each repo. Do NOT copy from an existing
# ~/.cache/huggingface hub that tt-metal populated with allow_patterns
# (that cache can have speech_tokenizer/model.safetensors but no config.json).
set -euo pipefail
CACHE="${HF_HUB_CACHE:-$HOME/tt-home-weights/hf_cache}"
mkdir -p "$CACHE"
export HF_HUB_CACHE="$CACHE"
export HUGGINGFACE_HUB_CACHE="$CACHE"
export PATH="${HOME}/.local/bin:${PATH}"

echo "Downloading weights into $CACHE"
hf download elyza/Llama-3-ELYZA-JP-8B
hf download openai/whisper-large-v3
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base

echo "Verifying required Qwen3-TTS files..."
QWEN_SNAP=$(find "$CACHE/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
MISSING=0
for f in \
    config.json \
    tokenizer_config.json \
    speech_tokenizer/config.json \
    speech_tokenizer/model.safetensors \
    model.safetensors
do
    if [ ! -e "$QWEN_SNAP/$f" ]; then
        echo "MISSING: $QWEN_SNAP/$f"
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "ERROR: Qwen3-TTS cache is incomplete. Re-run: hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    exit 1
fi
echo "Qwen3-TTS snapshot OK: $QWEN_SNAP"

echo "Done. Cache:"
du -sh "$CACHE"/models--*
