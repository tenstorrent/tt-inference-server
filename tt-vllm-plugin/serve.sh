VLLM_USE_V1=1 HF_MODEL='meta-llama/Llama-3.1-8B' \
vllm serve 'meta-llama/Llama-3.1-8B' \
    --max-model-len 1024 \
    --max-num-seqs 16 \
    --block-size 32 \
    --max-num-batched-tokens 1024