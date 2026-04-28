# External-Server DeepSeek Evals

Runners for evaluating a remote DeepSeek-R1-0528 server exposing OpenAI-compatible
endpoints (no local inference server). These scripts live under `evals/scripts/`
and drive `lm_eval` directly through the `.workflow_venvs/.venv_evals_common`
environment.

## Quick start

After cloning this repo, request a TT Console inference key at
<https://console.tenstorrent.com/dashboard/inference/keys> and save it once:

```bash
printf '%s\n' 'sk-tt-...' > ~/.tt-api-token
chmod 600 ~/.tt-api-token
```

Then run a smoke eval:

```bash
./evals/scripts/run_aime24_short_external.sh
```

Or run the full 30-question AIME24 eval:

```bash
./evals/scripts/run_aime24_all_external.sh
```

To estimate DeepSeek-style AIME24 pass@1 with 16 samples per problem:

```bash
./evals/scripts/run_aime24_pass1x16_external.sh
```

To inspect AIME24 generation lengths for any run output directory:

```bash
./evals/scripts/run_aime24_length_report.sh eval_results/r1_aime24_YYYYMMDD_HHMMSS
```

On first run, the script automatically creates
`.workflow_venvs/.venv_evals_common` and installs `lm_eval`. That one-time
bootstrap can take 5 to 15+ minutes; later runs reuse the same environment.

## Scripts

Only `run_*.sh` scripts are intended as user-facing entrypoints. `helper_*.sh`
scripts are shared internals used by those entrypoints.

| Script | Purpose | Default scope |
|---|---|---|
| `evals/scripts/run_deepseek_external.sh quick` | quick smoke mode | AIME24 short pass@1 x16 + MMLU-Pro sample (32/category) |
| `evals/scripts/run_deepseek_external.sh single` | single benchmark mode | AIME24 full (30) |
| `evals/scripts/run_deepseek_external.sh suite` | reporting-grade DeepSeek-aligned suite | AIME24 pass@1 x16 + GPQA Diamond + MATH-500 + LiveCodeBench + full MMLU-Pro |
| `evals/scripts/run_aime24_all_external.sh` | `r1_aime24` | 30 (full AIME24) |
| `evals/scripts/run_aime24_short_external.sh` | `r1_aime24_short` | 5 (IDs 60, 69, 75, 84, 86) |
| `evals/scripts/run_aime24_pass1x16_external.sh` | DeepSeek-style AIME24 pass@1 estimate | 16 full AIME24 runs, then aggregate |
| `evals/scripts/run_aime24_length_report.sh [OUTPUT_DIR]` | generation-length report versus GPU reference ranges | latest AIME output if `OUTPUT_DIR` is omitted |
| `evals/scripts/run_mmlu_pro_external.sh` | `mmlu_pro` | full 12,032 by default, or `LIMIT`/`MMLU_PRO_LIMIT` |

The short subset picks the AIME24 problems with the lowest mean GPU-reference
token counts — the quickest cases to validate correctness end-to-end. See
<https://github.com/tenstorrent/tt-metal/issues/37857#issuecomment-4116812760>
for the data behind the pick.

## Benchmark sizes

The current dataset sizes in the installed harness are:

| Task | Questions / prompts |
|---|---:|
| `r1_aime24` | 30 |
| `r1_gpqa_diamond` | 198 |
| `r1_math500` | 500 |
| `livecodebench` | `code_generation_lite` split shipped by the harness |
| `ifeval` | 541 |
| `mmlu_pro` | 12,032 |

`MMLU-Pro` is the outlier. Full `MMLU-Pro` is much larger than the rest of the
reasoning suite, so `suite` mode is intentionally long-running and intended for
reporting rather than quick iteration.

## Environment setup

The scripts manage their default eval environment automatically. If
`.workflow_venvs/.venv_evals_common/bin/lm_eval` is missing, the first run
bootstraps it in place. Override with `LM_EVAL_VENV=/path/to/venv` or
`LM_EVAL_BIN=/path/to/lm_eval` only if you want to use a preexisting install.

## Default environment

| Variable | Purpose | Example |
|---|---|---|
| `BASE_URL` *or* `DEPLOY_URL` (+ `SERVICE_PORT`) | Server root. Defaults to TT Console. `BASE_URL` wins if set; otherwise `DEPLOY_URL:SERVICE_PORT` is used (`SERVICE_PORT` defaults to `443`). Trailing `/v1` or `/v1/...` is stripped. | `https://console.tenstorrent.com` |
| `OPENAI_API_KEY` or `VLLM_API_KEY` | Auth for the remote server. TT Console requires a `sk-tt-*` key from <https://console.tenstorrent.com/dashboard/inference/keys>. Defaults to `~/.tt-api-token` when present. | `sk-tt-...` |
| `VLLM_MODEL` | Model name sent in the request body. Defaults to `deepseek-ai/DeepSeek-R1-0528`. | `deepseek-ai/DeepSeek-R1-0528` |
| `TOKENIZER_MODEL` | HuggingFace repo used to tokenize prompts for `--completions-api`. Chat-completions mode does not load a tokenizer. Defaults to `VLLM_MODEL`. | `deepseek-ai/DeepSeek-R1-0528` |
| `MAX_CONCURRENT` | Max simultaneous API requests. Defaults to `15`; decrease if the endpoint starts returning gateway timeouts or interrupted streams. | `5`, `15` |
| `MAX_GEN_TOKS` | Generation token cap passed as `max_tokens`. Defaults to `65535`, matching the 64K DeepSeek-R1-0528 model-card evaluation setup. | `65535` |
| `TEMPERATURE` | Sampling temperature. Defaults to `0.6`, matching the DeepSeek-R1-0528 model-card evaluation setup. | `0.6` |
| `TOP_P` | Nucleus sampling `top_p`. Defaults to `0.95`, matching the DeepSeek-R1-0528 model-card evaluation setup. | `0.95` |
| `MODEL_SEED` | API model seed. Normally unset; the pass@1 runner sets a distinct seed per run. | `1234000` |
| `STREAM` | Request streaming API responses. Defaults to `1` while TT Console has idle timeout issues. Set `STREAM=0` only when deliberately testing non-streaming behavior. | `STREAM=0` |
| `OUTPUT_DIR` | Destination for `lm_eval` results. Defaults to `./eval_results/<task>_<timestamp>/` or `./eval_results/deepseek_external_<mode>_<timestamp>/`. | `/tmp/deepseek_suite` |

Additional mode-specific knobs:

| Variable | Purpose | Default |
|---|---|---|
| `QUICK_MMLU_PRO_LIMIT` | `mmlu_pro` sample size in `quick` mode | `32` |
| `MMLU_PRO_LIMIT` | `mmlu_pro` sample size for `run_mmlu_pro_external.sh` | full run |
| `AIME_PASS1_RUNS` | Number of repeated AIME runs for `run_aime24_pass1x16_external.sh`, quick mode, and suite mode | `16` |

## TT Console key setup

If you use the default TT Console endpoint, request an inference key at
<https://console.tenstorrent.com/dashboard/inference/keys> and save it locally:

```bash
printf '%s\n' 'sk-tt-...' > ~/.tt-api-token
chmod 600 ~/.tt-api-token
```

Alternatively, export it for the current shell:

```bash
export OPENAI_API_KEY='sk-tt-...'
```

When targeting TT Console, the scripts fail early if no key is available or if
`OPENAI_API_KEY`/`VLLM_API_KEY` is set to a non-`sk-tt-*` key without
`~/.tt-api-token` available as a fallback.

## Example invocations

### Quick mode

```bash
./evals/scripts/run_deepseek_external.sh quick
```

This runs:

- `r1_aime24_short` pass@1 x16 (5 questions per run)
- `mmlu_pro --limit 32` (32 examples per MMLU-Pro category, 448 requests total)

### Single mode

```bash
./evals/scripts/run_deepseek_external.sh single
```

This runs:

- `r1_aime24` (full 30-question benchmark)

### AIME24 pass@1 x16

```bash
./evals/scripts/run_aime24_pass1x16_external.sh
```

This runs full AIME24 16 times with distinct API seeds, then writes:

- `r1_aime24_pass1_summary.json`
- `r1_aime24_pass1_summary.md`

### AIME24 generation-length report

```bash
./evals/scripts/run_aime24_length_report.sh eval_results/r1_aime24_YYYYMMDD_HHMMSS
```

This tokenizes saved `samples_*.jsonl` generations and prints an aligned
Markdown table against the reference GPU ranges from
<https://github.com/tenstorrent/tt-metal/issues/38446#issuecomment-4073998777>.
If no output directory is provided, it uses the latest AIME output under
`eval_results/`.

For streaming runs, the local streaming patch preserves DeepSeek reasoning
chunks in a `<think>...</think>` block before the visible answer, so the report
counts total generated reasoning-plus-answer tokens while the scorer still
extracts the final answer cleanly.

Useful options:

```bash
./evals/scripts/run_aime24_length_report.sh OUTPUT_DIR --show-reference-tt
./evals/scripts/run_aime24_length_report.sh OUTPUT_DIR --output OUTPUT_DIR/aime24_length_report.md
```

### Suite mode

```bash
./evals/scripts/run_deepseek_external.sh suite
```

This runs:

- `r1_aime24` pass@1 x16
- `r1_gpqa_diamond`
- `r1_math500`
- `livecodebench`
- `mmlu_pro` (full 12,032-question run)

DeepSeek-R1-0528 reports AIME 2024, GPQA-Diamond, and LiveCodeBench as
`Pass@1`, and their model card says benchmarks requiring sampling use 16
responses per query. Suite mode currently applies that repeated-sampling
aggregation to AIME24 only; GPQA-Diamond, MATH-500, and LiveCodeBench still run
once each until equivalent repeat aggregators are added.

### Full MMLU-Pro

```bash
./evals/scripts/run_mmlu_pro_external.sh
```

To run only a subset:

```bash
./evals/scripts/run_mmlu_pro_external.sh 128
```

### Host + port

```bash
DEPLOY_URL=https://my-deepseek.example.com \
SERVICE_PORT=443 \
OPENAI_API_KEY='sk-...' \
./evals/scripts/run_deepseek_external.sh single
```

## What the scripts do

- `helper_external_lm_eval.sh` is the shared helper underneath all wrappers.
- Resolve the server URL and strip any trailing `/v1`(`/...`) before re-appending
  the required endpoint path.
- Use `local-chat-completions` by default because TT Console exposes
  `/v1/chat/completions` for DeepSeek-R1-0528. Pass `--completions-api` to
  `helper_external_lm_eval.sh` only for non-Console servers that expose
  `/v1/completions`.
- In chat-completions mode, do not pass a HuggingFace tokenizer to `lm_eval`;
  prompts are sent as OpenAI chat messages, and avoiding tokenizer loading also
  avoids irrelevant DeepSeek RoPE config warnings from Transformers.
- Apply `--apply_chat_template`, `--trust_remote_code`,
  `--confirm_run_unsafe_code`, and log samples to the output directory.
- Default to concise logging: warnings/errors, tqdm progress, final scores, and
  the output directory. Pass `--show-config` to print full task configs,
  `--print-command` to print the full `lm_eval` command, or `--log-level INFO`
  to restore the harness's chatty informational logs.
- Pass DeepSeek-R1-0528 model-card generation defaults through CLI kwargs:
  `max_gen_toks=65535`, `temperature=0.6`, and `top_p=0.95`. Task YAML still
  owns task-specific stop strings such as `until`; use `--gen-kwargs` only when
  deliberately overriding or adding settings.
- Streaming is enabled by default. This adds `stream=true` to
  generation kwargs and loads `evals/scripts/lm_eval_streaming_patch/` through
  `PYTHONPATH`. Disable it with `STREAM=0` or `--no-stream`; non-streaming runs
  keep using the stock installed `lm_eval`.
  Streaming prevents idle connection timeouts, but it does not increase server
  capacity. The default is `MAX_CONCURRENT=15`; lower it if long reasoning evals
  hit gateway timeouts or interrupted streams.
  For chat streams that split `delta.reasoning` from `delta.content`, the patch
  preserves reasoning as a `<think>...</think>` block before the visible answer
  so sample logs retain total generation length while the scorer still extracts
  the final answer cleanly.
- Redact `sk-tt-*` API keys from terminal output and JSON/JSONL artifacts after
  the run, because upstream aiohttp exception reprs include request headers.
- Suppress the harmless `Cannot determine EOS string to pass to stop sequence`
  warning in terminal output. The server handles EOS termination; this warning
  only means `lm_eval` could not append an extra tokenizer-derived client-side
  stop string because chat-completions mode intentionally avoids tokenizer
  loading.
- Suppress known `lm_eval` noise for upstream AIME `acc` metadata and the
  generic chat-template warning. The current external wrappers use
  chat-completions with `generate_until` tasks, so the chat-template warning is
  expected and not a scoring risk.
- For the short AIME run, add `--include_path evals/custom_tasks/r1_aime24_short`
  so the custom task YAML + filter are discovered.

`suite` intentionally prefers `livecodebench` over `ifeval` because
`livecodebench` appears on current DeepSeek-R1-0528 benchmark tables, while
`ifeval` does not.

## Known warnings

- `Chat template formatting change affects loglikelihood and multiple-choice
  tasks`: expected for chat-completions mode and suppressed by the wrapper.
  `local-chat-completions` requires `--apply_chat_template` so requests are
  sent as chat messages. AIME and the current MMLU-Pro task are
  `generate_until`, not loglikelihood scoring.
- `[Task: r1_aime24] metric acc is defined...`: harmless upstream `lm_eval`
  task metadata noise and suppressed by the wrapper. The scored metric for AIME
  is `exact_match`; the local short AIME task removes this unused `acc` entry.
- `504 Gateway Time-out`: usually means TT Console's load balancer timed out
  waiting for a long non-streaming request or an overloaded queued request.
  Retry with lower concurrency, for example `MAX_CONCURRENT=1
  ./evals/scripts/run_aime24_all_external.sh`. If individual non-streaming
  requests still time out, keep streaming enabled.

## Output

Each run writes `lm_eval` results (JSON + per-sample logs) under
`OUTPUT_DIR`. The aggregate score is printed to stdout at the end and also
lives in `OUTPUT_DIR/.../results*.json`. Common keys for the current suite are:

The wrapper prints the exact debug paths up front:

- `Debug samples`: per-question records with the prompt/document, generated
  output, gold answer, and scoring details.
- `Results JSON`: aggregate metrics plus metadata for the run.

In the samples JSONL, the most useful debugging fields are:

- `arguments.gen_args_0.arg_0`: rendered input prompt/messages sent to the API.
- `arguments.gen_args_0.arg_1`: generation kwargs, including `max_gen_toks`.
- `resps` / `filtered_resps`: raw and post-filtered model output.
- `target` and metric fields such as `exact_match`: gold answer and score.

- `r1_aime24`: `exact_match,none`
- `r1_gpqa_diamond`: `exact_match,none`
- `r1_math500`: `exact_match,none`
- `livecodebench`: `acc,none`
- `mmlu_pro`: `exact_match,custom-extract`

If you run `ifeval` separately through `helper_external_lm_eval.sh`, the main
metrics are `prompt_level_strict_acc,none` and `inst_level_strict_acc,none`.

## Custom-task internals (short subset)

`evals/custom_tasks/r1_aime24_short/` contains:

- `r1_aime24_short.yaml` — clone of `r1_aime24.yaml` with `process_docs`
  pointing at the filter below.
- `utils.py` — re-exports `process_results_math` from the upstream
  `r1_evals` task and defines `filter_short_ids`, which keeps rows whose `id`
  is in `{60, 69, 75, 84, 86}`.

To change which questions are in the short set, edit `SHORT_AIME24_IDS` in
`utils.py`.
