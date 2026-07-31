#!/bin/bash
# Build push-button-v2-slim -> new_slim (Gemma4 baked, Llama dropped).
# Does NOT modify :push-button-v2-slim.
#
# Usage: ./build_new_slim.sh [--push]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_IMAGE="ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:push-button-v2-slim"
OUT_IMAGE="ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:new_slim"
BUILD_CTR="tt-voice-assistant-new-slim-build"

HOST_HF_HOME="${HOST_HF_HOME:-$HOME/.cache/huggingface}"
HOST_HF_HUB="${HOST_HF_HUB:-$HOST_HF_HOME/hub}"
HOST_TT_CACHE_ROOT="${HOST_TT_CACHE_ROOT:-$HOST_HF_HOME/tt_cache}"
GEMMA4_SRC="${GEMMA4_SRC:-$HOME/Teja/gemma3_metal/tt-metal/models/demos/gemma4}"
GEMMA_HF="$HOST_HF_HUB/models--google--gemma-4-12B-it"
GEMMA_TT="$HOST_TT_CACHE_ROOT/google--gemma-4-12B-it"
DO_PUSH=false
[[ "${1:-}" == "--push" ]] && DO_PUSH=true

CTR_HOME="/home/container_app_user"
CTR_HUB="$CTR_HOME/.cache/huggingface/hub"
CTR_TT="$CTR_HOME/.cache/huggingface/tt_cache"
CTR_APP="$CTR_HOME/voice-assistant"
CTR_METAL="$CTR_HOME/tt-metal"

echo "=========================================="
echo "  Build :new_slim from :push-button-v2-slim"
echo "=========================================="
echo "  Base:   $BASE_IMAGE"
echo "  Output: $OUT_IMAGE"
echo "  Push:   $DO_PUSH"
echo ""

for p in "$GEMMA_HF" "$GEMMA_TT" "$GEMMA4_SRC/tt"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing required path: $p"
    exit 1
  fi
done

echo "[1/8] Starting clean build container (no host mounts over hub)..."
docker rm -f "$BUILD_CTR" >/dev/null 2>&1 || true
docker run -d --name "$BUILD_CTR" \
  --entrypoint /bin/bash \
  "$BASE_IMAGE" \
  -c "sleep infinity"
sleep 2

echo "[2/8] Dropping Llama HF weights + TT model_cache from image..."
docker exec "$BUILD_CTR" bash -c "
set -e
# HF hub weights
rm -rf \
  $CTR_HUB/models--meta-llama--Llama-3.1-8B-Instruct \
  $CTR_HUB/models--meta-llama--Meta-Llama-3.1-8B-Instruct \
  $CTR_TT/meta-llama* \
  $CTR_TT/*Llama* \
  $CTR_TT/*llama* \
  2>/dev/null || true
# Slim/Llama path used by create_tt_model (see app_bh_qb2_slim_qwen3-tts):
#   /home/container_app_user/tt-metal/model_cache/meta-llama/...
rm -rf $CTR_METAL/model_cache/meta-llama
echo '  Remaining HF models:'
ls -d $CTR_HUB/models--* 2>/dev/null || true
echo '  Remaining model_cache:'
ls -la $CTR_METAL/model_cache 2>/dev/null || echo '(none)'
"

echo "[3/8] Baking Gemma4-12B HF weights (~23G)..."
docker exec "$BUILD_CTR" mkdir -p "$CTR_HUB" "$CTR_TT"
docker cp "$GEMMA_HF" "$BUILD_CTR:$CTR_HUB/models--google--gemma-4-12B-it"

echo "[4/8] Baking Gemma4 TT cache (~15G)..."
docker cp "$GEMMA_TT" "$BUILD_CTR:$CTR_TT/google--gemma-4-12B-it"

echo "[5/8] Baking app + Qwen3 TTS + Gemma4 model code..."
docker exec "$BUILD_CTR" mkdir -p \
  "$CTR_APP/output" "$CTR_APP/logs" \
  "$CTR_METAL/models/demos/qwen3_tts" \
  "$CTR_METAL/models/demos/gemma4"
docker cp "$SCRIPT_DIR/main.py" "$BUILD_CTR:$CTR_APP/main.py"
docker cp "$SCRIPT_DIR/services/." "$BUILD_CTR:$CTR_APP/services/"
docker cp "$SCRIPT_DIR/templates/." "$BUILD_CTR:$CTR_APP/templates/"
docker cp "$SCRIPT_DIR/servers/." "$BUILD_CTR:$CTR_APP/servers/"
docker cp "$SCRIPT_DIR/static/." "$BUILD_CTR:$CTR_APP/static/"
docker exec "$BUILD_CTR" rm -rf "$CTR_METAL/models/demos/qwen3_tts" "$CTR_METAL/models/demos/gemma4"
docker exec "$BUILD_CTR" mkdir -p "$CTR_METAL/models/demos/qwen3_tts" "$CTR_METAL/models/demos/gemma4"
docker cp "$SCRIPT_DIR/qwen3_tts_latest/." "$BUILD_CTR:$CTR_METAL/models/demos/qwen3_tts/"
docker cp "$GEMMA4_SRC/." "$BUILD_CTR:$CTR_METAL/models/demos/gemma4/"

echo "[6/8] Installing transformers==5.12.1 + Whisper compat patches..."
docker exec "$BUILD_CTR" bash -c '
set -e
source /home/container_app_user/tt-metal/python_env/bin/activate
pip install -q "transformers==5.12.1" "huggingface_hub>=0.30.0" "tokenizers>=0.21.0"
python - <<PY
import transformers
assert transformers.__version__.startswith("5.12."), transformers.__version__
print("transformers", transformers.__version__)
PY

python3 - <<PY
from pathlib import Path
p = Path("/home/container_app_user/tt-metal/models/common/generation_utils.py")
text = p.read_text()
if "HammingDiversityLogitsProcessor = None" not in text:
    old = "    HammingDiversityLogitsProcessor,\n"
    if old in text:
        text = text.replace(old, "", 1)
        needle = "    SuppressTokensLogitsProcessor,\n)\n"
        insert = needle + "try:\n    from transformers.generation.logits_process import HammingDiversityLogitsProcessor\nexcept ImportError:\n    HammingDiversityLogitsProcessor = None\n"
        if needle in text:
            text = text.replace(needle, insert, 1)
        old_use = """    if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )"""
        new_use = """    if (
        HammingDiversityLogitsProcessor is not None
        and generation_config.diversity_penalty is not None
        and generation_config.diversity_penalty > 0.0
    ):
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )"""
        if old_use in text:
            text = text.replace(old_use, new_use, 1)
        p.write_text(text)
        print("hamming patch applied")
    else:
        print("WARN: HammingDiversity import not found")
else:
    print("hamming patch already present")

p = Path("/home/container_app_user/tt-metal/models/demos/audio/whisper/tt/whisper_generator.py")
text = p.read_text()
if "transformers>=5 moved max_length" not in text:
    old = "        MAX_GEN_LEN = self.config.max_length\n"
    new = """        # transformers>=5 moved max_length off WhisperConfig onto GenerationConfig
        MAX_GEN_LEN = getattr(self.config, "max_length", None)
        if MAX_GEN_LEN is None:
            MAX_GEN_LEN = getattr(getattr(self, "generation_config", None), "max_length", None)
        if MAX_GEN_LEN is None:
            MAX_GEN_LEN = getattr(self.config, "max_target_positions", 448)
"""
    if old in text:
        p.write_text(text.replace(old, new, 1))
        print("whisper max_length patch applied")
    else:
        print("WARN: whisper max_length line not found")
else:
    print("whisper max_length patch already present")
PY
'

echo "[7/8] Verifying baked contents..."
docker exec "$BUILD_CTR" bash -c '
set -e
test -d /home/container_app_user/.cache/huggingface/hub/models--google--gemma-4-12B-it
test -d /home/container_app_user/.cache/huggingface/tt_cache/google--gemma-4-12B-it
test ! -d /home/container_app_user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct
test -f /home/container_app_user/tt-metal/models/demos/gemma4/tt/generator.py
test -f /home/container_app_user/voice-assistant/services/llama_service.py
source /home/container_app_user/tt-metal/python_env/bin/activate
python -c "import transformers; assert transformers.__version__.startswith(\"5.12.\")"
echo "  OK: Gemma HF+TT, no Llama, transformers 5.12.x, app present"
du -sh \
  /home/container_app_user/.cache/huggingface/hub/models--google--gemma-4-12B-it \
  /home/container_app_user/.cache/huggingface/tt_cache/google--gemma-4-12B-it \
  /home/container_app_user/.cache/huggingface/hub
'

echo "[8/8] Committing image $OUT_IMAGE ..."
docker commit \
  --change 'LABEL tt.voice-assistant.llm="google/gemma-4-12B-it"' \
  --change 'LABEL tt.voice-assistant.base="push-button-v2-slim"' \
  --change 'CMD ["sleep", "infinity"]' \
  "$BUILD_CTR" "$OUT_IMAGE"

echo "    Cleaning build container..."
docker rm -f "$BUILD_CTR" >/dev/null

echo ""
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.ID}}' | grep -E 'new_slim|push-button-v2-slim' || true
echo ""

if $DO_PUSH; then
  echo "Pushing $OUT_IMAGE ..."
  docker push "$OUT_IMAGE"
  echo "Push complete."
else
  echo "Local image ready. Push later with:"
  echo "  docker push $OUT_IMAGE"
  echo "or re-run:  ./build_new_slim.sh --push"
fi

echo ""
echo "Done. Customer pull:"
echo "  docker pull $OUT_IMAGE"
