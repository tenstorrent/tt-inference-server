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

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal/README.md) inference engine.

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Qwen3-32B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-bac8b34-7c6685a \
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
| Implementation Code | [qwen3-32b-galaxy](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `bac8b34` |
| vLLM Commit | `7c6685a` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-bac8b34-7c6685a` |

## Release Report

Models CI job: https://github.com/tenstorrent/tt-shield/actions/runs/22597865268/job/65476351959

### Metadata: Qwen3-32B on galaxy
```json
{
    "report_id": "id_qwen3-32b-galaxy_Qwen3-32B_galaxy_2026-03-03_01-53-04",
    "model_name": "Qwen3-32B",
    "model_id": "id_qwen3-32b-galaxy_Qwen3-32B_galaxy",
    "model_spec_json": "/home/ubuntu/actions-runner/_work/tt-shield/tt-shield/tt-inference-server/workflow_logs/run_specs/tt_model_spec_2026-03-03_00-00-49_id_qwen3-32b-galaxy_Qwen3-32B_galaxy_release_z26fQDA2.json",
    "model_repo": "Qwen/Qwen3-32B",
    "model_impl": "qwen3-32b-galaxy",
    "inference_engine": "vLLM",
    "device": "galaxy",
    "server_mode": "docker",
    "tt_metal_commit": "5fd3a04c7bae502122b7e0465dc6167ec2609567",
    "vllm_commit": "38dee8c",
    "run_command": "python run.py --model Qwen3-32B --device galaxy --workflow release --docker-server"
}
```

### Performance Benchmark Sweeps for Qwen3-32B on galaxy

#### vLLM Text-to-Text Performance Benchmark Sweeps for Qwen3-32B on galaxy

|  ISL  | OSL  | Concurrency | N Req | TTFT (ms) | TPOT (ms) | User Output Tput (tok/s/user) | Output Tput (tok/s) | Input Tput (tok/s) | E2EL (ms) | Req Tput (RPS) | Total Tput (tokens/duration) |
|-------|------|-------------|-------|-----------|-----------|-----------------|-------------------|--------------------|-----------|----------------|------------------------------------------|
|   128 |  128 |           1 |     8 |      70.8 |      22.8 |           43.9  |              43.9 |             1807.4 |    2964.1 |          0.337 |                                    86.36 |
|   128 |  128 |          32 |   256 |     570.7 |      21.3 |           46.86 |            1499.4 |             7176.6 |    3281.2 |          9.748 |                                  2495.48 |
|   128 | 1024 |           1 |     4 |      73.5 |      22.8 |           43.9  |              43.9 |             1741.9 |   23376.6 |          0.043 |                                    49.28 |
|   128 | 1024 |          32 |   128 |     668.6 |      21.5 |           46.62 |            1491.8 |             6125.8 |   22612.2 |          1.414 |                                  1628.54 |
|  1024 |  128 |           1 |     4 |     164.3 |      22.9 |           43.7  |              43.7 |             6231.6 |    3070.5 |          0.326 |                                   375.15 |
|  1024 |  128 |          32 |   128 |    4264.4 |      23.1 |           43.27 |            1384.7 |             7684.0 |    7199.3 |          4.045 |                                  4660.34 |
|  2048 |  128 |           1 |     4 |     295.7 |      23.3 |           42.9  |              42.9 |             6926.0 |    3256.1 |          0.307 |                                   668.24 |
|  2048 |  128 |          32 |   128 |    8155.7 |      25.5 |           39.26 |            1256.4 |             8035.7 |   11390.2 |          2.61  |                                  5678.78 |
|  4096 |  128 |           1 |     4 |     449.2 |      23.9 |           41.78 |              41.8 |             9118.1 |    3488.7 |          0.287 |                                  1210.68 |
|  4096 |  128 |          32 |   128 |   12720.3 |      28.6 |           35.01 |            1120.4 |            10304.2 |   16347.7 |          1.853 |                                  7827.92 |
|  8192 |  128 |           1 |     2 |     758.9 |      25.3 |           39.52 |              39.5 |            10794.8 |    3972.2 |          0.252 |                                  2094.41 |
|  8192 |  128 |          32 |    64 |   21839.5 |      35.1 |           28.49 |             911.6 |            12003.2 |   26297.4 |          1.133 |                                  9427.43 |
| 16384 |  128 |           1 |     2 |    1454.0 |      27.8 |           35.92 |              35.9 |            11267.9 |    4989.9 |          0.2   |                                  3308.92 |
| 16384 |  128 |          31 |    62 |   37614.7 |      56.1 |           17.81 |             552.2 |            13502.8 |   44743.9 |          0.626 |                                 10334.57 |
| 32768 |  128 |           1 |     1 |    3177.5 |      33.5 |           29.86 |              29.9 |            10312.5 |    7430.7 |          0.135 |                                  4426.83 |
| 32768 |  128 |          15 |    15 |   37515.1 |      55.2 |           18.12 |             271.9 |            13101.9 |   44522.3 |          0.271 |                                  8922.67 |
| 65536 |  128 |           1 |     1 |    8303.9 |      43.9 |           22.76 |              22.8 |             7892.2 |   13883.2 |          0.072 |                                  4729.63 |
| 65536 |  128 |           7 |     7 |   41851.8 |      70.4 |           14.2  |              99.4 |            10961.3 |   50793.0 |          0.103 |                                  6774.04 |

Note: all metrics are means across benchmark run unless otherwise stated.
> ISL: Input Sequence Length (tokens)
> OSL: Output Sequence Length (tokens)
> Concurrency: number of concurrent requests (batch size)
> N Req: total number of requests (sample size, N)
> TTFT: Time To First Token (ms)
> TPOT: Time Per Output Token (ms)
> User Output Tput: Throughput per user (tok/s)
> Output Tput: Throughput for decode tokens, across all users (tok/s)
> Input Tput: Throughput for prefill tokens (tok/s)
> E2EL: End-to-End Latency (ms)
> Req Tput: Request Throughput (RPS)

### Performance Benchmark Targets Qwen3-32B on galaxy

#### Text-to-Text Performance Benchmark Targets Qwen3-32B on galaxy

| ISL | OSL | Concurrency | TTFT (ms) | User Output Tput (tok/s/user) | Output Tput (tok/s) | Functional TTFT Check | Functional User Output Tput Check | Complete TTFT Check | Complete User Output Tput Check | Target TTFT Check | Target User Output Tput Check | Functional TTFT (ms) | Functional User Output Tput (tok/s/user) | Complete TTFT (ms) | Complete User Output Tput (tok/s/user) | Target TTFT (ms) | Target User Output Tput (tok/s/user) |
|-----|-----|-------------|-----------|-----------------|-------------------|-----------------------|----------------------------|---------------------|--------------------------|-------------------|------------------------|----------------------|----------------------------|--------------------|--------------------------|------------------|------------------------|
| 128 | 128 |           1 |      70.8 |            43.9 |              43.9 | PASS ✅               | PASS ✅                    | FAIL ⛔             | FAIL ⛔                  | FAIL ⛔           | FAIL ⛔                |               300.00 |                      15.60 |              60.00 |                    78.00 |            30.00 |                 156.00 |

Note: all metrics are means across benchmark run unless otherwise stated.
> ISL: Input Sequence Length (tokens)
> OSL: Output Sequence Length (tokens)
> Concurrency: number of concurrent requests (batch size)
> TTFT: Time To First Token (ms)
> User Output Tput: Throughput per user (tok/s)
> Output Tput: Throughput for decode tokens, across all users (tok/s)

### Accuracy Evaluations for Qwen3-32B on galaxy

| model     | device | task_name       | accuracy_check | score | ratio_to_reference | gpu_reference_score | ratio_to_published | published_score                                                                                             |
|-----------|--------|-----------------|----------------|-------|--------------------|---------------------|--------------------|-------------------------------------------------------------------------------------------------------------|
| Qwen3-32B | galaxy | r1_aime24       | FAIL ⛔         | 66.67 | 0.83               | [80.00](TBD)        | 0.82               | [81.40](https://qwenlm.github.io/blog/qwen3/)                                                               |
| Qwen3-32B | galaxy | r1_math500      | PASS ✅         | 94.00 | 0.98               | [96.10](TBD)        | 0.98               | [96.10](https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct) |
| Qwen3-32B | galaxy | r1_gpqa_diamond | PASS ✅         | 70.00 | 1.05               | [66.80](TBD)        | 1.05               | [66.80](https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct) |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

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
| `test_non_uniform_seeding` | `test_non_uniform_seeding` | ❌ FAILED | Traceback: AssertionError: Determinism Failed for seed=0.   Expected 1 unique output, found 3.   Outputs: {'<think>\nOkay, the user wants a list of 10 random colors. Let me think about how to approach this. First, I need to figure out what they mean ... |
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
