# just test that defined
JWT_SECRET=${JWT_SECRET:?}
HF_TOKEN=${HF_TOKEN:?}
# skip prompting for location, use default HF
export AUTOMATIC_HOST_SETUP=1
export MODEL_SOURCE=huggingface


# run each model
#python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow release --docker-server
python3 run.py --model Qwen2.5-72B-Instruct --device galaxy --workflow release --docker-server --dev-mode --ci-mode
sleep 10; ~/.tenstorrent-venv/bin/tt-smi -glx_reset; sleep 10; echo "tt-smi reset done";
python3 run.py --model Qwen3-32B --device galaxy --workflow release --docker-server --dev-mode --ci-mode
sleep 10; ~/.tenstorrent-venv/bin/tt-smi -glx_reset; sleep 10; echo "tt-smi reset done";
python3 run.py --model QwQ-32B --device galaxy --workflow release --docker-server --dev-mode --ci-mode
sleep 10; ~/.tenstorrent-venv/bin/tt-smi -glx_reset; sleep 10; echo "tt-smi reset done";
python3 run.py --model Qwen3-8B --device galaxy --workflow release --docker-server --dev-mode --ci-mode
sleep 10; ~/.tenstorrent-venv/bin/tt-smi -glx_reset; sleep 10; echo "tt-smi reset done";
python3 run.py --model Llama-3.1-8B-Instruct --device galaxy --workflow release --docker-server --dev-mode --ci-mode
sleep 10; ~/.tenstorrent-venv/bin/tt-smi -glx_reset; sleep 10; echo "tt-smi reset done";

# run repeated inferences on same vLLM server
python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow server --docker-server --dev-mode
sleep 300
for ((i=0; i < 1000; i++)); do
  echo "=== Iteration $i ==="
  python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow release --ci-mode
done
