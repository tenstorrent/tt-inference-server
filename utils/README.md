# Utils


## Prompt Client and Generation Tools

This repository contains tools for generating prompts and interfacing with a Open AI API compliant vLLM-based inference API. The two main components are:

1. `prompt_client_cli.py`: A command-line interface for sending prompts to a vLLM API server
2. `prompt_generation.py`: A utility for generating and processing prompts with various configurations

The primary usecase is stress testing of an Inference API server with many batches of inference with randomly varying context and timing.

### Using prompt_client_cli.py

The prompt client CLI tool allows you to send prompts to a vLLM API server with various configuration options.

#### Environment Variables

- `AUTHORIZATION`: Bearer token for API authentication
- `JWT_SECRET`: Alternative to AUTHORIZATION for JWT-based authentication (is available in Docker container)
- `DEPLOY_URL`: API server URL (default: http://127.0.0.1)
- `SERVICE_PORT`: API server port (default: 8000)
- `CACHE_ROOT`: Directory for saving response files (default: current directory)
- `VLLM_MODEL`: Model name (default: meta-llama/Llama-3.1-70B-Instruct)

#### Key Arguments

- `--num_prompts`: Number of prompts to generate
- `--batch_size`: Number of concurrent requests
- `--max_prompt_length`: Maximum length for generated prompts
- `--output_seq_len`: Maximum length for completions
- `--num_full_iterations`: Number of times to repeat the full prompt set
- `--vary-batch-size`: Randomize batch sizes using normal distribution
- `--input_seq_len`: Fixed length for input sequences (-1 for variable)
- `--inter_batch_delay`: Delay between batches in seconds
- `--no-stream`: Disable streaming responses
- `--dataset`: Source dataset (random, alpaca_eval)
- `--distribution`: Prompt length distribution (fixed, uniform, normal)
- `--template`: Path to Jinja2 template or "chat_template" for model tokenizer default

#### Example Usage

To have the python requirements and environment variables pre-set to match deployed inference server
it is recommended for testing to enter a bash shell in the container running the inference server
and run `prompt_client_cli.py` from there:

```bash
# oneliner to enter interactive shell on most recently ran container
docker exec -it $(docker ps -q | head -n1) bash
cd ~/app/utils

# send random prompts by default
python prompt_client_cli.py \
    --num_prompts 10 \
    --batch_size 4 \
    --tokenizer_model meta-llama/Llama-3.1-70B-Instruct \
    --max_prompt_length 512 \
    --output_seq_len 2048

# send prompts from alpaca_eval using chat template from tokenizer
python prompt_client_cli.py \
    --num_prompts 12 \
    --batch_size 4 \
    --tokenizer_model meta-llama/Llama-3.1-70B-Instruct \
    --max_prompt_length 2048 \
    --template chat_template \
    --dataset alpaca_eval \
    --num_full_iterations 1

# with random batch sizes and delays between batches
python prompt_client_cli.py \
    --num_prompts 12 \
    --batch_size 4 \
    --tokenizer_model meta-llama/Llama-3.1-70B-Instruct \
    --max_prompt_length 2048 \
    --template chat_template \
    --dataset alpaca_eval \
    --vary_batch_size \
    --inter_batch_delay 2 \
    --num_full_iterations 1

# with jinja2 prompt template
python prompt_client_cli.py \
    --num_prompts 4 \
    --batch_size 1 \
    --tokenizer_model meta-llama/Llama-3.1-70B-Instruct \
    --max_prompt_length 2048 \
    --template prompt_templates/llama_instruct_example.jinja \
    --dataset alpaca_eval
```

#### Python dependencies

The recommended usage (described above) is from within the Docker container where these are installed by default.

```bash
pip install transformers datasets torch requests jwt jinja2
```

#### Response Format

The client saves responses in JSON format with the following structure:

```json
{
    "response_idx": 0,
    "prompt": "example prompt",
    "response": "model response",
    "prompt_length": 128,
    "num_completion_tokens": 256,
    "tps": 45.6,
    "ttft": 0.15
}
```

#### Logging

Both tools use Python's logging module with INFO level by default. You can see detailed progress, including:
- Prompt generation progress
- API request status
- Tokens per second (TPS)
- Time to first token (TTFT)
- Completion token counts
- Batch processing status

### Using prompt_generation.py separately

The prompt generation module can be imported and used in your own scripts for fine-grained control over prompt generation.

```python
from prompt_generation import generate_prompts
from types import SimpleNamespace

# Create custom args
args = SimpleNamespace(
    tokenizer_model="meta-llama/Llama-3.1-70B-Instruct",
    dataset="alpaca_eval",
    max_prompt_length=512,
    input_seq_len=-1,
    num_prompts=5,
    distribution="normal",
    template="templates/chat.j2",
    save_path="generated_prompts.jsonl",
    lm_eval_task=None
)

# Generate prompts
prompts, prompt_lengths = generate_prompts(args)
```

