#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-/app/model}"
HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"

mkdir -p "$MODEL_PATH"

# Ensure config.json exists (required by frontend)
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[entrypoint] No config.json found, creating minimal config..."
    cat > "$MODEL_PATH/config.json" << 'CONF'
{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":128256,"eos_token_id":[128001,128008,128009],"bos_token_id":128000}
CONF
fi

# Download tokenizer if not already present
if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "[entrypoint] Downloading model files from $HF_MODEL_ID..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
model_id = os.environ.get('HF_MODEL_ID', '$HF_MODEL_ID')
token = os.environ.get('HF_TOKEN', None)
dest = '$MODEL_PATH'
for f in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
    try:
        path = hf_hub_download(model_id, f, token=token)
        shutil.copy(path, os.path.join(dest, f))
        print(f'  Downloaded {f}')
    except Exception as e:
        print(f'  Skipped {f}: {e}')
" || echo "[entrypoint] WARNING: HF download failed (gated model?). Frontend will start without tokenizer."
    echo "[entrypoint] Model files ready."
else
    echo "[entrypoint] Model files already present at $MODEL_PATH"
fi

echo "[entrypoint] Starting Dynamo frontend on port $HTTP_PORT..."
echo "  DYN_DISCOVERY_BACKEND=$DYN_DISCOVERY_BACKEND"
if [ "$DYN_DISCOVERY_BACKEND" = "etcd" ]; then
    echo "  ETCD_ENDPOINTS=$ETCD_ENDPOINTS"
fi
echo "  DYN_REQUEST_PLANE=$DYN_REQUEST_PLANE"
echo "  DYN_EVENT_PLANE=$DYN_EVENT_PLANE"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  MODEL_PATH=$MODEL_PATH"

exec python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --model-name "$MODEL_NAME" \
    --model-path "$MODEL_PATH"
