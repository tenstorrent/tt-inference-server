# just test that defined
JWT_SECRET=${JWT_SECRET:?}
HF_TOKEN=${HF_TOKEN:?}
# skip prompting for location, use default HF
export AUTOMATIC_HOST_SETUP=1
export MODEL_SOURCE=huggingface

python3 run.py --model Qwen/Qwen2.5-72B-Instruct --device galaxy --workflow release --docker-server --dev-mode
python3 run.py --model Llama-3.1-8B-Instruct --device galaxy --workflow release --docker-server --dev-mode
python3 run.py --model Qwen/Qwen3-32B --device galaxy --workflow release --docker-server --dev-mode
python3 run.py --model Qwen/Qwen3-8B --device galaxy --workflow release --docker-server --dev-mode
python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow release --docker-server --dev-mode
