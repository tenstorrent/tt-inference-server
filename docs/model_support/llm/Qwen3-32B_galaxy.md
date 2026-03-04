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

## Release Report

Models CI job: https://github.com/tenstorrent/tt-shield/actions/runs/21076491467/job/60621726246#step:11:10

## Tenstorrent Model Release Summary: Qwen3-32B on galaxy

### Metadata: Qwen3-32B on galaxy
```json
{
    "report_id": "id_qwen3-32b-galaxy_Qwen3-32B_galaxy_2026-01-16_20-41-44",
    "model_name": "Qwen3-32B",
    "model_id": "id_qwen3-32b-galaxy_Qwen3-32B_galaxy",
    "model_spec_json": "/home/ubuntu/actions-runner/_work/tt-shield/tt-shield/tt-inference-server/workflow_logs/run_specs/tt_model_spec_2026-01-16_18-47-46_id_qwen3-32b-galaxy_Qwen3-32B_galaxy_release_B-ri7fc5.json",
    "model_repo": "Qwen/Qwen3-32B",
    "model_impl": "qwen3-32b-galaxy",
    "inference_engine": "vLLM",
    "device": "galaxy",
    "server_mode": "docker",
    "tt_metal_commit": "a9b09e0b611da6deb4d8972e8296148fd864e5fd",
    "vllm_commit": "a186bf4",
    "run_command": "python run.py --model Qwen3-32B --device galaxy --workflow release --docker-server"
}
```

### Performance Benchmark Sweeps for Qwen3-32B on galaxy

#### vLLM Text-to-Text Performance Benchmark Sweeps for Qwen3-32B on galaxy

|  ISL  | OSL  | Concurrency | N Req | TTFT (ms) | TPOT (ms) | User Output Tput (tok/s/u) | Output Tput (tok/s) | Input Tput (tok/s) | E2EL (ms) | Req Tput (RPS) | Total Tput (tok/s) |
|-------|------|-------------|-------|-----------|-----------|-----------------|-------------------|--------------------|-----------|----------------|------------------------------------------|
|   128 |  128 |           1 |     8 |      70.2 |      22.7 |           44.07 |              44.1 |             1824.1 |    2951.9 |          0.339 |                                    86.72 |
|   128 |  128 |          32 |   256 |     592.6 |      21.5 |           46.51 |            1488.4 |             6912.1 |    3323.1 |          9.625 |                                  2464.11 |
|   128 | 1024 |           1 |     4 |      72.0 |      22.8 |           43.94 |              43.9 |             1777.2 |   23352.1 |          0.043 |                                    49.33 |
|   128 | 1024 |          32 |   128 |     585.2 |      21.5 |           46.49 |            1487.6 |             6998.8 |   22591.2 |          1.416 |                                  1631.65 |
|  1024 |  128 |           1 |     4 |     163.2 |      22.7 |           44.09 |              44.1 |             6276.0 |    3043.3 |          0.329 |                                   378.5  |
|  1024 |  128 |          32 |   128 |    4259.1 |      23.0 |           43.51 |            1392.3 |             7693.7 |    7177.9 |          4.032 |                                  4645.08 |
|  2048 |  128 |           1 |     4 |     294.4 |      23.2 |           43.14 |              43.1 |             6956.8 |    3238.5 |          0.309 |                                   671.88 |
|  2048 |  128 |          32 |   128 |    8048.6 |      25.9 |           38.57 |            1234.1 |             8142.6 |   11341.5 |          2.628 |                                  5717.83 |
|  4096 |  128 |           1 |     4 |     448.1 |      23.8 |           41.99 |              42.0 |             9141.1 |    3472.7 |          0.288 |                                  1216.27 |
|  4096 |  128 |          32 |   128 |   12445.1 |      30.0 |           33.38 |            1068.0 |            10532.0 |   16250.3 |          1.856 |                                  7841.81 |
|  8192 |  128 |           1 |     2 |     754.9 |      25.3 |           39.59 |              39.6 |            10851.5 |    3963.2 |          0.252 |                                  2099.18 |
|  8192 |  128 |          32 |    64 |   21755.5 |      34.8 |           28.73 |             919.4 |            12049.5 |   26175.9 |          1.139 |                                  9472.55 |
| 16384 |  128 |           1 |     2 |    1445.4 |      27.7 |           36.14 |              36.1 |            11335.5 |    4959.4 |          0.202 |                                  3329.28 |
| 16384 |  128 |          31 |    62 |   36890.6 |      60.6 |           16.51 |             511.9 |            13767.8 |   44582.3 |          0.628 |                                 10372.79 |
| 32768 |  128 |           1 |     1 |    3152.1 |      33.2 |           30.13 |              30.1 |            10395.7 |    7367.5 |          0.136 |                                  4464.84 |
| 32768 |  128 |          15 |    15 |   37272.4 |      54.8 |           18.25 |             273.8 |            13187.2 |   44230.9 |          0.273 |                                  8982.27 |
| 65536 |  128 |           1 |     1 |    8243.9 |      43.5 |           23.0  |              23.0 |             7949.6 |   13766.1 |          0.073 |                                  4769.84 |
| 65536 |  128 |           7 |     7 |   41608.3 |      70.2 |           14.25 |              99.7 |            11025.5 |   50521.6 |          0.104 |                                  6810.19 |


Note: all metrics are means across benchmark run unless otherwise stated.
> ISL: Input Sequence Length (tokens)
> OSL: Output Sequence Length (tokens)
> Concurrency: number of concurrent requests (batch size)
> N Req: total number of requests (sample size, N)
> TTFT: Time To First Token (ms)
> TPOT: Time Per Output Token (ms)
> User Output Tput: Interactivity, throughput per user (tok/s/u)
> Output Tput: Throughput for output decode tokens, across all users (tok/s)
> Input Tput: Throughput for input prefill tokens (tok/s)
> E2EL: End-to-End Latency (ms)
> Req Tput: Request Throughput (RPS)

### Performance Benchmark Targets Qwen3-32B on galaxy

#### Text-to-Text Performance Benchmark Targets Qwen3-32B on galaxy

| ISL | OSL | Concurrency | TTFT (ms) | Tput User (TPS) | Tput Decode (TPS) | Functional TTFT Check | Functional Tput User Check | Complete TTFT Check | Complete Tput User Check | Target TTFT Check | Target Tput User Check | Functional TTFT (ms) | Functional Tput User (TPS) | Complete TTFT (ms) | Complete Tput User (TPS) | Target TTFT (ms) | Target Tput User (TPS) |
|-----|-----|-------------|-----------|-----------------|-------------------|-----------------------|----------------------------|---------------------|--------------------------|-------------------|------------------------|----------------------|----------------------------|--------------------|--------------------------|------------------|------------------------|
| 128 | 128 |           1 |      72.9 |           48.69 |              48.7 | PASS ✅               | PASS ✅                    | FAIL ⛔             | FAIL ⛔                  | FAIL ⛔           | FAIL ⛔                |               300.00 |                      15.60 |              60.00 |                    78.00 |            30.00 |                 156.00 |

Note: all metrics are means across benchmark run unless otherwise stated.
> ISL: Input Sequence Length (tokens)
> OSL: Output Sequence Length (tokens)
> Concurrency: number of concurrent requests (batch size)
> TTFT: Time To First Token (ms)
> Tput User: Throughput per user (TPS)
> Tput Decode: Throughput for decode tokens, across all users (TPS)

### Benchmark Performance Results for Qwen3-32B on galaxy

**Metric Definitions:**
> - **ISL**: Input Sequence Length (tokens)
> - **OSL**: Output Sequence Length (tokens)
> - **Concur**: Concurrent requests (batch size)
> - **N**: Total number of requests
> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)
> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)
> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)
> - **Tok/s**: Output token throughput
> - **Req/s**: Request throughput


### Accuracy Evaluations for Qwen3-32B on galaxy

| Model     | HW     | Benchmark       | Status | Target | TT Score (norm.) | TT Score        | Ref Score (norm.) | Ref Score                                                                                                   |
|-----------|--------|-----------------|--------|--------|------------------|-----------------|-------------------|-------------------------------------------------------------------------------------------------------------|
| Qwen3-32B | galaxy | r1_aime24       | PASS ✅ | 80.00  | 1.00             | [80.00](TBD)    | 0.98              | [81.40](https://qwenlm.github.io/blog/qwen3/)                                                               |
| Qwen3-32B | galaxy | r1_math500      | FAIL ⛔ | 85.00  | 0.88             | [96.10](TBD)    | 0.88              | [96.10](https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct) |
| Qwen3-32B | galaxy | r1_gpqa_diamond | PASS ✅ | 65.00  | 0.97             | [66.80](TBD)    | 0.97              | [66.80](https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct) |

> Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

### Test Results for Qwen3-32B on galaxy

### LLM API Test Metadata

| Attribute | Value |
| --- | --- |
| **Endpoint URL** | `http://127.0.0.1:8000/v1/chat/completions` |
| **Test Timestamp** | 2026-01-16 20:34:26 UTC |

### Parameter Conformance Summary

| Test Case | Status | Summary |
| --- | :---: | --- |
| `test_determinism_parameters` | ✅ PASS | 3/3 passed |
| `test_logprobs` | ✅ PASS | 1/1 passed |
| `test_max_tokens` | ✅ PASS | 2/2 passed |
| `test_n` | ✅ PASS | 2/2 passed |
| `test_non_uniform_seeding` | ✅ PASS | 1/1 passed |
| `test_penalties` | ✅ PASS | 9/9 passed |
| `test_seed_reproducibility` | ✅ PASS | 1/1 passed |
| `test_stop` | ✅ PASS | 2/2 passed |

### Detailed Test Results

| Test Case | Parametrization | Status | Message |
| --- | --- | :---: | --- |
| `test_determinism_parameters` | `test_determinism_parameters[temperature-0.0]` | ✅ PASSED |  |
| `test_determinism_parameters` | `test_determinism_parameters[top_k-1]` | ✅ PASSED |  |
| `test_determinism_parameters` | `test_determinism_parameters[top_p-0.01]` | ✅ PASSED |  |
| `test_logprobs` | `test_logprobs` | ✅ PASSED |  |
| `test_max_tokens` | `test_max_tokens[10]` | ✅ PASSED |  |
| `test_max_tokens` | `test_max_tokens[5]` | ✅ PASSED |  |
| `test_n` | `test_n[2]` | ✅ PASSED |  |
| `test_n` | `test_n[3]` | ✅ PASSED |  |
| `test_non_uniform_seeding` | `test_non_uniform_seeding` | ✅ PASSED |  |
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
