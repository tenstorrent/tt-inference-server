# Custom LongBench input

368 prompts covering all 28 GLM-5.3 benchmark conditions:
64 at 128 tokens; 16 each at 10,000, 196,608 and 255,872;
32 each at 1K, 2K, 4K, 8K, 16K, 32K, 64K and 128K (K = 1,024).
[samples.csv](samples.csv) lists every selected original ID in JSONL order.

## Preparation

- **128–16K:** 240 samples from [LongBench V1](https://huggingface.co/datasets/zai-org/LongBench/blob/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip).
- **32K–255,872:** 128 samples from [LongBench V2](https://huggingface.co/datasets/zai-org/LongBench-v2/blob/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json).

Each prompt uses one source sample and its [upstream template](https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e).
Prompts retain their original English or Chinese text.
Overlength prompts retain their beginning and end, following official middle
truncation ([V1 code](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py#L56-L62),
[V2 code](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py#L24-L35)).
Our preparation additionally applies bounded corrections for exact token counts.
No padding, rewriting, translation, or ID insertion is applied during curation.
Lengths were verified with GLM-5.3.

## Selection

The original eight lengths use candidate pools shuffled with seed 0, selected
longest first (32 per length). Their 256 rows are unchanged. For the four added
lengths, process longest first, sort all original source IDs and shuffle with
seed 0 per length. Take eligible rows, excluding reused source IDs and matching
first 128 GLM-5.3 chat-template tokens. The maximum shared prefix is 112 tokens.

## Use

Follow the [benchmark command](../../docs/custom_longbench.md) with
`--dataset-path datasets/custom-longbench/custom-longbench.jsonl`.
Each condition uses the first `num_prompts` rows of its input length.
Disable server prefix caching. This is a throughput workload, not an accuracy suite.
