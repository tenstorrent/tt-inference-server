# Benchmarking Tools

The `benchmarks` workflow can drive four client tools against the same server and the same sweep. This guide covers what each tool does, how the harness invokes it, and why their numbers differ. What the sweep contains, how results are graded, and how to read a report are in the [Performance Benchmarks reference](../reference_config/benchmarking/README.md).

| Tool | `--tools` | Client | Version | Best for |
|---|---|---|---|---|
| [vLLM `vllm bench serve`](#vllm-vllm-bench-serve) | `vllm` (default) | Python venv | `vllm==0.13.0`, pinned | Release grading, regression tracking |
| [AIPerf](#aiperf) | `aiperf` | Python venv | unpinned | Full percentile distribution, per-request records, goodput |
| [GenAI-Perf](#genai-perf) | `genai` | NVIDIA Triton SDK Docker image | `nvcr.io/nvidia/tritonserver:25.11-py3-sdk` | Comparison with Triton-based results |
| [GuideLLM](#guidellm) | `guidellm` | Python venv | unpinned | Multi-turn, custom-dataset and omni-modal scenarios |

```bash
python3 run.py --model Qwen/Qwen3-32B --tt-device t3k --workflow benchmarks --tools aiperf
```

Each Python client gets its own virtual environment under `.workflow_venvs/`, created on first use; `--reset-venvs` rebuilds them. For an unpinned tool, record its version (`.workflow_venvs/<venv>/bin/pip show aiperf`) with the results.

## vLLM `vllm bench serve`

Driver: [`llm_module/drivers/vllm.py`](../llm_module/drivers/vllm.py). Each sweep point runs:

```text
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model <model>
  --dataset-name random --random-input-len <ISL> --random-output-len <OSL>
  --max-concurrency <C> --num-prompts <N> --percentile-metrics ttft,tpot,itl,e2el
  --save-result --save-detailed --result-filename <file>
  --host <host> --port <port> --extra-body '{"truncate_prompt_tokens": <ISL>}'
```

- **Output length.** vLLM 0.13.0 forces ignore-EOS for the `random` dataset on OpenAI-compatible backends, so every request generates exactly OSL tokens.
- **Warmup.** One unmeasured readiness request per point; `--num-warmups` defaults to 0.
- **Percentiles.** Mean, median, P99 and standard deviation for TTFT, TPOT, ITL and E2E latency (`--metric-percentiles` adds more). `--save-detailed` keeps per-request values in the result JSON.
- **Goodput.** `--goodput ttft:<ms> tpot:<ms> e2el:<ms>` is passed when a [requirements document](../reference_config/benchmarking/README.md#goodput-and-requirements-documents) sets SLOs for the point.
- **Remote endpoints** (`--server-url https://…`) use `--base-url`, a bearer-token header and no prompt truncation.
- The [fixed-workload protocol](../reference_config/benchmarking/README.md#fixed-workload-protocol) runs this tool through a timing adapter with a separately pinned client and tokenizer.

## AIPerf

Driver: [`llm_module/drivers/aiperf.py`](../llm_module/drivers/aiperf.py). [AIPerf](https://github.com/ai-dynamo/aiperf) is NVIDIA's OpenAI-compatible benchmarking client from the ai-dynamo project. Each sweep point runs:

```text
python -m aiperf profile --model <model> --tokenizer <model> --endpoint-type chat --streaming
  --url <url> --concurrency <C> --request-count <N>
  --synthetic-input-tokens-mean <ISL> --synthetic-input-tokens-stddev 0
  --output-tokens-mean <OSL> --output-tokens-stddev 0 --artifact-dir <dir>
```

- **Output length is not guaranteed.** Without `--extra-inputs ignore_eos:true`, requests may stop early at EOS, so OSL can be shorter than requested. The harness does not pass it today.
- **Warmup.** None; AIPerf runs no warmup unless `--warmup-request-count` or `--warmup-duration` is set, and the harness sets neither.
- **Percentiles.** P1 to P99 for every metric, plus per-request records.
- **Goodput.** `run.py --goodput "time_to_first_token:<ms> inter_token_latency:<ms> request_latency:<ms>"` applies to every point. Keys are AIPerf metric tags with values in their display units.
- **Artifacts.** `aiperf_artifacts/bench_<isl>_<osl>_<conc>_n<N>/` in the workflow output: `profile_export_aiperf.json` (aggregates), `profile_export.jsonl` (one record per request), `profile_export_aiperf.csv`.

## GenAI-Perf

Driver: [`llm_module/drivers/genai_perf.py`](../llm_module/drivers/genai_perf.py). Runs `genai-perf profile` in the NVIDIA Triton SDK container (`docker run --net host`, about 10 GB on first pull) with the same synthetic-input options as AIPerf. It passes no warmup and no ignore-EOS option, so expect a cold first point and variable output lengths.

## GuideLLM

Driver: [`llm_module/drivers/guidellm.py`](../llm_module/drivers/guidellm.py). With `--tools guidellm`, [GuideLLM](https://github.com/vllm-project/guidellm) runs the synthetic ISL/OSL sweep at fixed concurrency like the other tools. It also runs dataset-driven scenarios defined in [`llm_module/guidellm_scenarios.py`](../llm_module/guidellm_scenarios.py): multi-turn chat, custom datasets and omni-modal (text, image, video, audio) workloads.

## Understanding metric differences

The tools agree on what they send but not on where they stop each clock. Compare numbers from one tool only, and name the tool when quoting them.

**TTFT.** vLLM's server streams an empty role announcement before the first token:

```text
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}   <- vLLM client stops TTFT here
data: {"choices":[{"delta":{"content":"The"}}]}                    <- AIPerf stops TTFT here
```

`vllm bench serve` (`openai-chat` backend) stops the clock at the first streamed chunk that has any `choices`; AIPerf stops at the first chunk carrying content, reasoning or tool-call text. On gemma-3-4b-it on N300 at 128/128, concurrency 1, vLLM measured 73 ms and AIPerf 93 ms for the same server. The fixed-workload protocol stops at the first non-empty content event, like AIPerf.

**Decode metrics** share definitions under different names:

| Quantity | vLLM | AIPerf | Definition |
|---|---|---|---|
| Per-request decode time per token | TPOT | `inter_token_latency` | (E2E latency − TTFT) / (output tokens − 1), averaged over requests |
| Gap between streamed chunks | ITL | `inter_chunk_latency` | Every chunk gap, pooled across requests |
| Per-user decode throughput (`tput_user`) | `1000 / mean TPOT` (derived by the harness) | `output_token_throughput_per_user` | Tokens/s seen by one user |

TPOT agreed within about 5% between the two tools in the measurement above. `tput_user` can differ slightly between them when TPOT varies across requests, because the mean of reciprocals is not the reciprocal of the mean.

**Throughput.** Input, output and total token throughput are aggregate over the whole run (all concurrent requests), in tokens/s. Request throughput is completed requests per second.

**Percentiles.** At the default sweep's request counts (as few as one request per point at long ISL), a P99 is effectively the maximum and a mean can be dominated by one cold request. Use the [acceptance procedure](../reference_config/benchmarking/README.md#recommended-procedure-for-acceptance-runs) when percentiles matter.

## Running a tool outside the harness

These stand-alone commands reproduce one sweep point against a running server, with the warmup and exact-length options the harness does not yet pass. Set `API_KEY` to the server's bearer token (the harness derives it from `JWT_SECRET`; omit the header with `--no-auth` servers).

vLLM (install `vllm==0.13.0` in a separate venv):

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --host localhost --port 8000 --model Qwen/Qwen3-32B --dataset-name random --random-input-len 2048 --random-output-len 128 --ignore-eos --max-concurrency 32 --num-prompts 128 --num-warmups 32 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --extra-body '{"truncate_prompt_tokens": 2048}' --header "Authorization=Bearer $API_KEY" --save-result --save-detailed --result-filename qwen3-32b_t3k_isl2048_osl128_c32.json
```

AIPerf:

```bash
aiperf profile --model Qwen/Qwen3-32B --tokenizer Qwen/Qwen3-32B --url http://localhost:8000 --endpoint-type chat --streaming --synthetic-input-tokens-mean 2048 --synthetic-input-tokens-stddev 0 --output-tokens-mean 128 --output-tokens-stddev 0 --extra-inputs ignore_eos:true --concurrency 32 --request-count 128 --warmup-request-count 32 --api-key "$API_KEY" --artifact-dir artifacts/qwen3-32b_t3k_isl2048_osl128_c32
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| First point has very high TTFT | Server still capturing traces, or a cold client. Wait for `Background trace capture completed successfully` in the container log before benchmarking |
| `Connection refused` | Check `curl http://localhost:8000/health` and `docker logs <container>` |
| `401 Unauthorized` | Set `JWT_SECRET` to the value the server was started with, or start the server with `--no-auth` |
| Python dependency errors in a client venv | `--reset-venvs` |
| GenAI-Perf `docker pull` output | Informational; the first pull is large |
| AIPerf output length below OSL | Requests stopped at EOS; see [AIPerf](#aiperf) |

## References

- vLLM: [`vllm bench serve` CLI (v0.13.0)](https://docs.vllm.ai/en/v0.13.0/cli/bench/serve/), [benchmark CLI guide](https://docs.vllm.ai/en/v0.13.0/benchmarking/cli/), [`serve.py`](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/benchmarks/serve.py) and [`endpoint_request_func.py`](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/benchmarks/lib/endpoint_request_func.py) (metric definitions and timestamps)
- AIPerf: [repository](https://github.com/ai-dynamo/aiperf), [metrics reference](https://docs.nvidia.com/aiperf/reference/ai-perf-metrics-reference), [warmup](https://docs.nvidia.com/aiperf/tutorials/load-patterns-scheduling/warmup-phase-configuration), [goodput](https://docs.nvidia.com/aiperf/tutorials/metrics-analysis/benchmark-goodput-with-ai-perf)
- GenAI-Perf: [documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html)
- GuideLLM: [repository](https://github.com/vllm-project/guidellm)
- tt-metal: [LLM tech report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) (prefill versus decode, tracing), [models performance table](https://github.com/tenstorrent/tt-metal/blob/main/models/README.md)
