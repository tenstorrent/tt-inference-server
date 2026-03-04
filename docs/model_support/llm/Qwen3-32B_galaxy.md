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
|   128 |  128 |           1 |     8 |      72.9 |      20.5 |           48.69 |              48.7 |             1756.6 |    2681.2 |          0.373 |                                    95.09 |
|   128 |  128 |          32 |   256 |     568.4 |      21.8 |           45.78 |            1464.9 |             7206.8 |    3342.6 |          9.559 |                                  2437.53 |
|   128 | 1024 |           1 |     4 |      77.8 |      21.0 |           47.6  |              47.6 |             1644.5 |   21568.6 |          0.046 |                                    53.36 |
|   128 | 1024 |          32 |   128 |     523.4 |      22.7 |           44.0  |            1408.0 |             7825.8 |   23773.8 |          1.346 |                                  1548.9  |
|  1024 |  128 |           1 |     4 |     171.1 |      21.4 |           46.82 |              46.8 |             5985.7 |    2883.8 |          0.347 |                                   399.07 |
|  2048 |  128 |           1 |     4 |     311.7 |      22.1 |           45.23 |              45.2 |             6571.4 |    3119.7 |          0.32  |                                   697.02 |
|  2048 |  128 |          32 |   128 |    9106.5 |      26.0 |           38.39 |            1228.6 |             7196.6 |   12414.4 |          2.576 |                                  5603.5  |
|  2048 | 2048 |          32 |    64 |    9083.1 |      26.4 |           37.87 |            1211.8 |             7215.2 |   63137.1 |          0.507 |                                  2075.26 |
|  3000 |   64 |          32 |   128 |   14467.7 |      26.1 |           38.28 |            1224.8 |             6635.5 |   16113.6 |          1.985 |                                  6080.32 |
|  3072 |  128 |           1 |     4 |     481.4 |      22.4 |           44.7  |              44.7 |             6381.2 |    3322.4 |          0.301 |                                   962.73 |
|  4000 |   64 |          32 |   128 |   14465.4 |      27.4 |           36.5  |            1167.9 |             8848.7 |   16191.6 |          1.975 |                                  8026.23 |
|  4096 |  128 |           1 |     2 |     484.1 |      23.1 |           43.33 |              43.3 |             8461.6 |    3415.2 |          0.293 |                                  1236.33 |
|  8000 |   64 |          16 |    32 |   12107.2 |      28.2 |           35.43 |             566.8 |            10572.2 |   13885.6 |          1.152 |                                  9287.49 |
|  8192 |  128 |           1 |     2 |     800.3 |      24.8 |           40.34 |              40.3 |            10235.6 |    3948.7 |          0.253 |                                  2106.48 |
| 16000 |   64 |           8 |    16 |   11837.1 |      29.9 |           33.5  |             268.0 |            10813.5 |   13717.8 |          0.583 |                                  9365.38 |

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

|-----------|--------|-----------------|----------------|-------|--------------------|---------------------|--------------------|-------------------------------------------------------------------------------------------------------------|
| Qwen3-32B | galaxy | r1_aime24       | PASS ✅         | 80.00 | 1.00               | [80.00](TBD)        | 0.98               | [81.40](https://qwenlm.github.io/blog/qwen3/)                                                               |
| Qwen3-32B | galaxy | r1_math500      | FAIL ⛔         | 85.00 | 0.88               | [96.10](TBD)        | 0.88               | [96.10](https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct) |
| Qwen3-32B | galaxy | r1_gpqa_diamond | PASS ✅         | 65.00 | 0.97               | [66.80](TBD)        | 0.97               | [66.80](https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct) |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

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
