# Qwen3-32B Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-32B` is also supported on hardware:

- [BH LoudBox](Qwen3-32B_p150x8.md)
- [WH LoudBox/QuietBox](Qwen3-32B_t3k.md)

## Quickstart - Deploy Qwen3-32B Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Qwen3-32B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-e867533-8f36910 \
  --model Qwen3-32B \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [qwen3-32b-galaxy](https://github.com/tenstorrent/tt-metal/tree/e867533/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `e867533` |
| vLLM Commit | `8f36910` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-e867533-8f36910` |

## Performance Benchmark Sweeps for Qwen3-32B on galaxy

#### vLLM Text-to-Text Performance Benchmark Sweeps for Qwen3-32B on galaxy

|   Source    |  ISL  | OSL  | Concurrency | N Req | TTFT (ms) | TPOT (ms) | Tput User (TPS) | Tput Decode (TPS) | Tput Prefill (TPS) | E2EL (ms) | Req Tput (RPS) | Total Token Throughput (tokens/duration) |
|-------------|-------|------|-------------|-------|-----------|-----------|-----------------|-------------------|--------------------|-----------|----------------|------------------------------------------|
| openai-chat |   128 |  128 |           1 |     8 |      81.0 |      33.6 |           29.76 |              29.8 |             1579.7 |    4348.7 |          0.23  |                                    58.87 |
| openai-chat |   128 |  128 |          32 |   256 |     635.7 |      35.8 |           27.9  |             892.7 |             6443.5 |    5188.3 |          5.635 |                                  1442.62 |
| openai-chat |   128 | 1024 |           1 |     4 |      83.5 |      33.6 |           29.73 |              29.7 |             1532.4 |   34495.7 |          0.029 |                                    33.4  |
| openai-chat |   128 | 1024 |          32 |   128 |     612.8 |      32.4 |           30.86 |             987.4 |             6683.9 |   33765.7 |          0.948 |                                  1091.7  |
| openai-chat |  1024 |  128 |           1 |     4 |     174.2 |      33.6 |           29.74 |              29.7 |             5878.8 |    4444.9 |          0.225 |                                   259.16 |
| openai-chat |  1024 |  128 |          32 |   128 |    4321.3 |      34.0 |           29.38 |             940.3 |             7582.9 |    8643.3 |          3.291 |                                  3791.31 |
| openai-chat |  2048 |  128 |           1 |     4 |     305.2 |      34.2 |           29.26 |              29.3 |             6710.6 |    4645.0 |          0.215 |                                   468.44 |
| openai-chat |  2048 |  128 |          32 |   128 |    8091.6 |      37.3 |           26.84 |             859.0 |             8099.3 |   12822.9 |          2.285 |                                  4972.54 |
| openai-chat |  4096 |  128 |           1 |     4 |     457.6 |      34.7 |           28.78 |              28.8 |             8951.4 |    4869.9 |          0.205 |                                   867.32 |
| openai-chat |  4096 |  128 |          32 |   128 |   12604.6 |      40.9 |           24.48 |             783.3 |            10398.7 |   17793.2 |          1.686 |                                  7122.67 |
| openai-chat |  8192 |  128 |           1 |     2 |     765.9 |      36.2 |           27.63 |              27.6 |            10696.0 |    5361.6 |          0.187 |                                  1551.71 |
| openai-chat |  8192 |  128 |          32 |    64 |   22082.6 |      45.8 |           21.82 |             698.3 |            11871.1 |   27902.2 |          1.147 |                                  9540.35 |
| openai-chat | 16384 |  128 |           1 |     2 |    1456.1 |      38.7 |           25.84 |              25.8 |            11251.9 |    6371.7 |          0.157 |                                  2591.36 |
| openai-chat | 16384 |  128 |          31 |    62 |   37037.8 |      70.8 |           14.12 |             437.8 |            13713.1 |   46030.6 |          0.604 |                                  9968.07 |
| openai-chat | 32768 |  128 |           1 |     1 |    3162.5 |      44.1 |           22.68 |              22.7 |            10361.3 |    8763.0 |          0.114 |                                  3753.84 |
| openai-chat | 32768 |  128 |          15 |    15 |   37545.3 |      65.6 |           15.24 |             228.6 |            13091.4 |   45878.8 |          0.26  |                                  8548.28 |
| openai-chat | 65536 |  128 |           1 |     1 |    8242.0 |      54.2 |           18.44 |              18.4 |             7951.4 |   15129.1 |          0.066 |                                  4340.15 |
| openai-chat | 65536 |  128 |           7 |     7 |   42184.0 |      80.8 |           12.38 |              86.7 |            10875.0 |   52443.1 |          0.1   |                                  6547.32 |

Note: all metrics are means across benchmark run unless otherwise stated.
> ISL: Input Sequence Length (tokens)
> OSL: Output Sequence Length (tokens)
> Concurrency: number of concurrent requests (batch size)
> N Req: total number of requests (sample size, N)
> TTFT: Time To First Token (ms)
> TPOT: Time Per Output Token (ms)
> Tput User: Throughput per user (TPS)
> Tput Decode: Throughput for decode tokens, across all users (TPS)
> Tput Prefill: Throughput for prefill tokens (TPS)
> E2EL: End-to-End Latency (ms)
> Req Tput: Request Throughput (RPS)

### Performance Benchmark Targets Qwen3-32B on galaxy

#### Text-to-Text Performance Benchmark Targets Qwen3-32B on galaxy

| ISL | OSL | Concurrency | TTFT (ms) | Tput User (TPS) | Tput Decode (TPS) | Functional TTFT Check | Functional Tput User Check | Complete TTFT Check | Complete Tput User Check | Target TTFT Check | Target Tput User Check | Functional TTFT (ms) | Functional Tput User (TPS) | Complete TTFT (ms) | Complete Tput User (TPS) | Target TTFT (ms) | Target Tput User (TPS) |
|-----|-----|-------------|-----------|-----------------|-------------------|-----------------------|----------------------------|---------------------|--------------------------|-------------------|------------------------|----------------------|----------------------------|--------------------|--------------------------|------------------|------------------------|
| 128 | 128 |           1 |      81.0 |           29.76 |              29.8 | PASS ✅               | PASS ✅                    | FAIL ⛔             | FAIL ⛔                  | FAIL ⛔           | FAIL ⛔                |               300.00 |                      15.60 |              60.00 |                    78.00 |            30.00 |                 156.00 |

Note: all metrics are means across benchmark run unless otherwise stated.
> ISL: Input Sequence Length (tokens)
> OSL: Output Sequence Length (tokens)
> Concurrency: number of concurrent requests (batch size)
> TTFT: Time To First Token (ms)
> Tput User: Throughput per user (TPS)
> Tput Decode: Throughput for decode tokens, across all users (TPS)



### Test Results for Qwen3-32B on galaxy

### LLM API Test Metadata

| Attribute | Value |
| --- | --- |
| **Endpoint URL** | `http://127.0.0.1:8000/v1/chat/completions` |
| **Test Timestamp** | N/A |

### Parameter Conformance Summary

| Test Case | Status | Summary |
| --- | :---: | --- |
| `test_determinism_parameters` | ✅ PASS | 3/3 passed |
| `test_logprobs` | ❌ FAIL | 0/1 passed |
| `test_max_tokens` | ✅ PASS | 2/2 passed |
| `test_n` | ✅ PASS | 2/2 passed |
| `test_non_uniform_seeding` | ❌ FAIL | 0/1 passed |
| `test_penalties` | ✅ PASS | 9/9 passed |
| `test_seed_reproducibility` | ✅ PASS | 1/1 passed |
| `test_stop` | ✅ PASS | 2/2 passed |

### Detailed Test Results

| Test Case | Parametrization | Status | Message |
| --- | --- | :---: | --- |
| `test_determinism_parameters` | `test_determinism_parameters[temperature-0.0]` | ✅ PASSED |  |
| `test_determinism_parameters` | `test_determinism_parameters[top_k-1]` | ✅ PASSED |  |
| `test_determinism_parameters` | `test_determinism_parameters[top_p-0.01]` | ✅ PASSED |  |
| `test_logprobs` | `test_logprobs` | ❌ FAILED | Traceback: requests.exceptions.HTTPError: API Error: 500 Server Error: Internal Server Error for url: http://127.0.0.1:8000/v1/chat/completions. Response: {'error': {'message': 'list index out of range', 'type': 'Internal Server Error', 'param': None... |
| `test_max_tokens` | `test_max_tokens[10]` | ✅ PASSED |  |
| `test_max_tokens` | `test_max_tokens[5]` | ✅ PASSED |  |
| `test_n` | `test_n[2]` | ✅ PASSED |  |
| `test_n` | `test_n[3]` | ✅ PASSED |  |
| `test_non_uniform_seeding` | `test_non_uniform_seeding` | ❌ FAILED | Traceback: AssertionError: Determinism Failed for seed=0.   Expected 1 unique output, found 3.   Outputs: {'<think>\nOkay, the user wants a list of 10 random colors. Let me think about how to approach this. First, I need to generate colors in a way t... |
| `test_penalties` | `test_penalties[frequency_penalty-1.2-natural_repetition-messages1]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[frequency_penalty-1.2-repeat_trap-messages0]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[presence_penalty-1.2-natural_repetition-messages1]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[presence_penalty-1.2-repeat_trap-messages0]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[presence_penalty-1.2-semantic_repetition-messages2]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[repetition_penalty-1.5-natural_repetition-messages1]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[repetition_penalty-1.5-repeat_trap-messages0]` | ✅ PASSED |  |
| `test_penalties` | `test_penalties[repetition_penalty-1.5-semantic_repetition-messages2]` | ✅ PASSED |  |
| `test_seed_reproducibility` | `test_seed_reproducibility` | ✅ PASSED |  |
| `test_stop` | `test_stop[stop_seq0]` | ✅ PASSED |  |
| `test_stop` | `test_stop[stop_seq1]` | ✅ PASSED |  |

---

## GALAXY_T3K Configuration

### Quickstart - Deploy on WH Galaxy

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Qwen3-32B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-e95ffa5-48eba14 \
  --model Qwen3-32B \
  --tt-device galaxy_t3k
```

**via run.py command**

```bash
python3 run.py --model Qwen3-32B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-e95ffa5-48eba14` |
