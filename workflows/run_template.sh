# python3 run.py --model QwQ-32B --workflow evals --docker-server --device t3k --dev-mode
# python3 run.py --model Qwen2.5-72B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.3-70B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.2-11B-Vision-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.2-11B-Vision-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.2-3B-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.2-3B-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.2-3B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Qwen2.5-7B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Qwen2.5-7B-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Qwen2.5-7B-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.1-8B-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.1-8B-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.1-8B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model DeepSeek-R1-Distill-Llama-70B --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Qwen2.5-7B-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Qwen2.5-72B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.3-70B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow evals --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow evals --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow evals --docker-server --device t3k --dev-mode

# python3 run.py --model Llama-3.3-70B-Instruct --workflow benchmarks --device t3k
# python3 run.py --model QwQ-32B --workflow evals --docker-server --device t3k --dev-mode
# python3 run.py --model DeepSeek-R1-Distill-Llama-70B --workflow evals --docker-server --device t3k --dev-mode
# python3 run.py --model Qwen2.5-72B-Instruct --workflow evals --docker-server --device t3k --dev-mode

# python3 run.py --model Qwen2.5-7B-Instruct --workflow evals --docker-server --device n150 --dev-mode

# python3 run.py --model Llama-3.2-1B-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.2-1B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.2-3B-Instruct --workflow release --docker-server --device n300 --dev-mode
# python3 run.py --model Llama-3.2-3B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Qwen2.5-7B-Instruct --workflow release --docker-server --device t3k --dev-mode
# python3 run.py --model Llama-3.1-8B-Instruct --workflow release --docker-server --device n150 --dev-mode
# python3 run.py --model Llama-3.1-8B-Instruct --workflow release --docker-server --device t3k --dev-mode

# python3 run.py --model Llama-3.2-1B-Instruct --workflow reports --device n150
# python3 run.py --model Llama-3.2-1B-Instruct --workflow reports --device n300
# python3 run.py --model Llama-3.2-1B-Instruct --workflow reports --device t3k
# python3 run.py --model Llama-3.2-3B-Instruct --workflow reports --device n150
# python3 run.py --model Llama-3.2-3B-Instruct --workflow reports --device n300
# python3 run.py --model Llama-3.2-3B-Instruct --workflow reports --device t3k

# python3 run.py --model Llama-3.2-3B-Instruct --workflow reports --device n300
# python3 run.py --model Qwen2.5-72B-Instruct --workflow reports --device t3k
# python3 run.py --model Qwen2.5-7B-Instruct --workflow reports --device n300

# python3 run.py --model Llama-3.2-11B-Vision-Instruct --device t3k --workflow release --docker-server --dev-mode
# python3 run.py --model Llama-3.2-11B-Vision-Instruct --device n300 --workflow release --docker-server --dev-mode
# python3 run.py --model Qwen2.5-7B-Instruct --device t3k --workflow release --docker-server --dev-mode 
# python3 run.py --model Qwen2.5-7B-Instruct --device n300 --workflow release --docker-server --dev-mode
SLEEP_TIME=1
echo "sleep for $SLEEP_TIME seconds ..."
sleep $SLEEP_TIME
# python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow release  --docker-server
# python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow release --docker-server
# python3 run.py --model Llama-3.2-1B-Instruct --device t3k --workflow release --docker-server
# python3 run.py --model Llama-3.2-3B-Instruct --device n150 --workflow release --docker-server
# python3 run.py --model Llama-3.2-3B-Instruct --device n300 --workflow release --docker-server
# python3 run.py --model Llama-3.2-3B-Instruct --device t3k --workflow release --docker-server
# python3 run.py --model Llama-3.1-8B-Instruct --device n150 --workflow release --docker-server
# python3 run.py --model Llama-3.1-8B-Instruct --device n300 --workflow release --docker-server
# python3 run.py --model Llama-3.1-8B-Instruct --device t3k --workflow release --docker-server
# python3 run.py --model Qwen2.5-72B-Instruct --device t3k --workflow benchmarks --docker-server --dev-mode
# python3 run.py --model Llama-3.3-70B-Instruct --device t3k --workflow release --docker-server --dev-mode
# python3 run.py --model QwQ-32B --device t3k --workflow evals --docker-server --dev-mode
# python3 run.py --model DeepSeek-R1-Distill-Llama-70B --device t3k --workflow evals --docker-server --dev-mode
# python3 run.py --model DeepSeek-R1-Distill-Llama-70B --device t3k --workflow reports

# python3 run.py --model Llama-3.2-11B-Vision-Instruct --device t3k --workflow release --docker-server --dev-mode
# python3 run.py --model Llama-3.3-70B-Instruct --device t3k --workflow benchmarks --docker-server --dev-mode
# python3 run.py --model Qwen2.5-72B-Instruct --device t3k --workflow release --docker-server --dev-mode

python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow release --docker-server --dev-mode
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow release --docker-server --dev-mode
python3 run.py --model Llama-3.2-1B-Instruct --device t3k --workflow release --docker-server --dev-mode
python3 run.py --model Qwen2.5-7B-Instruct --device n300 --workflow release --docker-server --dev-mode
python3 run.py --model Llama-3.2-3B-Instruct --device n300 --workflow release --docker-server --dev-mode
python3 run.py --model Llama-3.2-3B-Instruct --device t3k --workflow release --docker-server --dev-mode
# python3 run.py --model Llama-3.2-11B-Vision-Instruct --device n300 --workflow release --docker-server --dev-mode


# TODO:
## post fix L1 OOM
python3 run.py --model Llama-3.2-3B-Instruct --device n150 --workflow release --docker-server --dev-mode
# python3 run.py --model Llama-3.1-8B-Instruct --device t3k --workflow release --docker-server --dev-mode



# TESTS
