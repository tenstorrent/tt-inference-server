# Utils


## Prompt Client and Generation Tools

This repository contains tools for generating prompts and interfacing with a Open AI API compliant vLLM-based inference API. The two main components are:

1. `prompt_client_cli.py`: A command-line interface for sending prompts to a vLLM API server
2. `prompt_generation.py`: A utility for generating and processing prompts with various configurations

The primary usecase is stress testing of an Inference API server with many batches of inference with randomly varying context and timing.

### Using prompt_client_cli.py

The prompt client CLI tool allows you to send prompts to a vLLM API server with various configuration options.

#### Environment Variables

- `VLLM_API_KEY`: Bearer token for API authentication
- `JWT_SECRET`: Alternative to VLLM_API_KEY for JWT-based authentication (is available in Docker container)
- `DEPLOY_URL`: API server URL (default: http://127.0.0.1)
- `SERVICE_PORT`: API server port (default: 8000)
- `CACHE_ROOT`: Directory for saving response files (default: current directory)
- `VLLM_MODEL`: Model name (default: meta-llama/Llama-3.1-70B-Instruct)

**Authentication Priority Order:**
1. `VLLM_API_KEY` - used if set (production standard)
2. `JWT_SECRET` - generates token if VLLM_API_KEY is not set (development/container use)
3. No authentication - proceeds without auth if none are set

#### Command Line Arguments

##### Core Parameters

- `--num_prompts` (default: 1)  
  Number of unique prompts to generate for testing.

- `--max_concurrent` (default: 32)  
  Max number of concurrent requests to send to the API server. Controls parallelization level.

- `--num_full_iterations` (default: 1)  
  Number of complete iterations over the entire prompt set. Useful for extended testing cycles.

##### Model Configuration

- `--vllm_model` (default: "meta-llama/Llama-3.1-70B-Instruct")  
  Model identifier for the vLLM API server. Can be overridden by VLLM_MODEL environment variable.

- `--tokenizer_model` (default: None)  
  Specific tokenizer model to use for vocabulary, truncation, and templating operations.

- `--use_chat_api` (default: False)
  Use /v1/chat/completions API: https://platform.openai.com/docs/api-reference/chat/create

##### Sequence Length Controls

- `--input_seq_len` (default: -1)  
  Length parameter for input sequences when using random prompts. -1 allows variable lengths.

- `--output_seq_len` (default: 2048)  
  Forces all completions to a fixed maximum length for consistent testing.

- `--max_prompt_length` (default: -1)  
  Maximum allowed length for generated prompts. -1 indicates no length restriction.

##### Batch Processing Options

- `--vary_max_concurrent` (default: False)  
  When enabled, randomizes the batch size for each prompt batch using normal distribution.

- `--inter_batch_delay` (default: 0)  
  Seconds to wait between processing each batch. Useful for rate limiting.

- `--no-stream` (default: False)  
  Disables streaming responses. By default, streaming is enabled.

##### Prompt Generation Settings

- `--distribution` (default: "fixed")  
  Method for determining random prompt lengths:
  - "fixed": Constant length
  - "uniform": Uniform distribution
  - "normal": Normal distribution

- `--dataset` (default: "random")  
  Source dataset for prompt generation. Use "random" for synthetic prompts.

- `--template` (default: None)  
  Jinja2 template for formatting prompts. Can be a file path or template string.

##### Output Controls

- `--save_path` (default: None)  
  File path to save generated prompts in JSONL format.

- `--print_prompts` (default: False)  
  Enable printing of generated prompts to stdout.

- `--skip_trace_precapture` (default: False)  
  Skips trace precapture phase, use to speed up execution if trace captures have already completed.

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
    --max_concurrent 4 \
    --tokenizer_model ${HF_MODEL_REPO_ID} \
    --input_seq_len 512 \
    --output_seq_len 2048

# send prompts from alpaca_eval using chat template from tokenizer
python prompt_client_cli.py \
    --num_prompts 12 \
    --max_concurrent 4 \
    --tokenizer_model ${HF_MODEL_REPO_ID} \
    --max_prompt_length 2048 \
    --template chat_template \
    --dataset alpaca_eval \
    --num_full_iterations 1

# with random batch sizes and delays between batches
python prompt_client_cli.py \
    --num_prompts 12 \
    --max_concurrent 4 \
    --tokenizer_model ${HF_MODEL_REPO_ID} \
    --max_prompt_length 2048 \
    --template chat_template \
    --dataset alpaca_eval \
    --vary_max_concurrent \
    --inter_batch_delay 2 \
    --num_full_iterations 1

# with jinja2 prompt template
python prompt_client_cli.py \
    --num_prompts 4 \
    --max_concurrent 1 \
    --tokenizer_model ${HF_MODEL_REPO_ID} \
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
  "response_idx": number,   // Response index in batch
  "prompt": string,         // Input prompt
  "response": string,       // Generated completion text
  "input_seq_len": number,  // Prompt length in tokens
  "output_seq_len": number, // Completion length in tokens
  "itl_ms": number[],        // Inter Token Latency (ITL) ms
  "tpot_ms": number,        // Time Per Output Token (TPOT) average, ms 
  "ttft_ms": number         // Time To First Token (TTFT) ms
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
    template="prompt_templates/llama_instruct_example.jinja",
    save_path="generated_prompts.jsonl",
    lm_eval_task=None
)

# Generate prompts
prompts, prompt_lengths = generate_prompts(args)
```

