# Custom LongBench benchmark

`--benchmark custom-longbench` selects the bundled LongBench prompts for the
GLM-5.3 `benchmarks` workflow. The default random benchmark and the other
evaluation workflows retain their existing selection and settings.

## Dataset

The repository includes
[`datasets/custom-longbench/custom-longbench.jsonl`](../datasets/custom-longbench/custom-longbench.jsonl).
No separate dataset export is required. The adjacent
[dataset README](../datasets/custom-longbench/README.md) describes the public
LongBench sources, truncation, curation, all 368 selected original sample IDs,
and input-length validation.

The bundle covers all 12 raw input lengths, from 128 to 255,872 tokens.
It provides 64 prompts at 128 tokens; 16 each at 10,000, 196,608 and 255,872;
and 32 at each original length.
Use a GLM-5.3 endpoint with the matching tokenizer and chat template.

## Run

Use an idle GLM-5.3 endpoint with prefix caching disabled on both Prefill and
Decode. Choose a fresh `CACHE_ROOT` for each run. The command adds two arguments
to the standard benchmark:

```bash
export API_KEY=EMPTY_TOKEN  # Replace for an authenticated endpoint.
export CACHE_ROOT=/absolute/path/to/new/custom-results
python run.py \
  --model GLM-5.3 \
  --workflow benchmarks \
  --benchmark custom-longbench \
  --dataset-path "$PWD/datasets/custom-longbench/custom-longbench.jsonl" \
  --device super_cluster \
  --server-url http://SERVER:PORT \
  --skip-system-sw-validation \
  --dev-mode
```

The selector splits the bundle into per-length vLLM JSONL files containing only
`prompt`. It keeps the original output length, concurrency, and request count
for every matching sweep point. Insufficient rows fail before requests are sent;
implicit oversampling is rejected. Input lengths absent from the bundle are
reported and skipped.

This bundle matches all 28 GLM-5.3 conditions, including the 128-, 10000-,
196608-, and 255872-token inputs. Output length, concurrency and request count
are identical to the default random sweep: 373 requests and 112,256 output
tokens in total. Each prompt comes from one sufficiently long source sample.

For each point, the existing vLLM driver receives options such as:

```text
--backend openai-chat --endpoint /v1/chat/completions
--dataset-name custom --dataset-path /path/to/custom-longbench-isl-8192.jsonl
--custom-output-len 1024 --skip-chat-template --disable-shuffle --ignore-eos
```

`--skip-chat-template` leaves the single template application to the server.
`--custom-output-len` preserves the selected output budget. `--ignore-eos`
matches vLLM 0.13.0's automatic behavior for random datasets. Sampling remains
greedy and streamed, with the same request count and concurrency. Performance
acceptance targets for random inputs are not applied to custom-input results.

## Compare with random inputs

Omit `--benchmark` and `--dataset-path` to run the original benchmark, using
another fresh output directory. Compare only matching conditions on the same
server without other traffic. Report actual output tokens and request failures.
Raw input length is measured before the server applies its chat template.

For both random and custom-longbench, verify that each server role reports
`enable_prefix_caching=False` in `vllm:cache_config_info`, and retain the native
prefix-cache hit counter deltas (expected zero). Curation alone does not prevent
reuse when a row is replayed. The client's `--prefix-cache` option selects a
separate benchmark suite; it does not change the server's cache setting.
