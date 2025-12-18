# Download ShareGPT dataset if it doesn't exist
if [ ! -f "datasets/sharegpt.json" ]; then
    echo "ShareGPT dataset not found. Downloading..."
    mkdir -p datasets
    wget -O datasets/sharegpt.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    echo "Download complete."
fi

VLLM_USE_V1=1 \
vllm bench serve --model 'meta-llama/Llama-3.1-8B-Instruct' \
                 --host localhost --port 8000 \
                 --dataset-name sharegpt --dataset-path datasets/sharegpt.json \
                 --request-rate 2 --num-prompts 500 \
                 --save-result 