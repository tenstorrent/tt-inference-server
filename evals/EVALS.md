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

Then run a smoke suite:

```bash
./evals/scripts/run_deepseek_external.sh smoke
```

For a very light live-endpoint check, run sanity mode:

```bash
./evals/scripts/run_deepseek_external.sh sanity
```

Or run a broad quick suite:

```bash
./evals/scripts/run_deepseek_external.sh quick
```

For reporting-grade results, run the full suite:

```bash
./evals/scripts/run_deepseek_external.sh full
```

Reuse `OUTPUT_DIR=/path/to/results` to resume a suite at the first benchmark
that does not already have results. To inspect AIME24 generation lengths for any
run output directory:

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
| `evals/scripts/run_deepseek_external.sh sanity` | gentle live-endpoint sanity check | 5 short AIME24 questions, `MAX_CONCURRENT=1`, `MAX_GEN_TOKS=32768`; exits 0 only on 5/5 |
| `evals/scripts/run_deepseek_external.sh smoke` | tiny end-to-end suite flow check | one or two examples from each final-table benchmark |
| `evals/scripts/run_deepseek_external.sh quick` | broad reduced suite | limited slices from each final-table benchmark |
| `evals/scripts/run_deepseek_external.sh full` | reporting-grade suite | full final-table benchmarks with AIME24/AIME25 pass@1 x16 |
| `evals/scripts/run_deepseek_external.sh suite` | alias for `full` | same as `full` |
| `evals/scripts/run_deepseek_external.sh single` | single benchmark mode | AIME24 once, plus derived majority row |
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
| `r1_aime25` | 30 |
| `r1_gpqa_diamond` | 198 |
| `r1_math500` | 500 |
| `mmlu_generative` | MMLU subjects from `cais/mmlu` |
| `mmlu_pro` | 12,032 |
| `gsm8k` | 1,319 |
| `arc_challenge_chat` | 1,172 |
| `humaneval_instruct` | 164 |
| `mbpp_instruct` | 500 |

Full `MMLU-Pro`, MMLU, and the code benchmarks are long-running. Use `quick`
for iteration and `full` only when you need reporting-grade numbers.

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
| `MAX_CONCURRENT` | Max simultaneous API requests. Defaults to `30`; decrease if the endpoint starts returning gateway timeouts or interrupted streams. | `5`, `30` |
| `MAX_GEN_TOKS` | Generation token cap passed as `max_tokens`. Defaults to `65535`, matching the 64K DeepSeek-R1-0528 model-card evaluation setup. | `65535` |
| `TEMPERATURE` | Sampling temperature. Defaults to `0.6`, matching the DeepSeek-R1-0528 model-card evaluation setup. | `0.6` |
| `TOP_P` | Nucleus sampling `top_p`. Defaults to `0.95`, matching the DeepSeek-R1-0528 model-card evaluation setup. | `0.95` |
| `MODEL_SEED` | API model seed. Normally unset; the pass@1 runner sets a distinct seed per run. | `1234000` |
| `STREAM` | Request streaming API responses. Defaults to `0`; set `STREAM=1` only when deliberately falling back to streaming behavior. | `STREAM=1` |
| `OUTPUT_DIR` | Destination for `lm_eval` results. Defaults to `./eval_results/deepseek_external_<mode>_<timestamp>/`. Reuse the same directory to resume completed benchmarks. | `/tmp/deepseek_suite` |

Additional mode-specific knobs:

| Variable | Purpose | Default |
|---|---|---|
| `AIME_PASS1_LIMIT` | Problem limit for direct `run_aime24_pass1x16_external.sh` use | full run |
| `MMLU_PRO_LIMIT` | `mmlu_pro` sample size for `run_mmlu_pro_external.sh` | full run |
| `AIME_PASS1_RUNS` | Number of repeated AIME runs for direct `run_aime24_pass1x16_external.sh` use | `16` |
| `AIME_PASS1_PARALLEL_RUNS` | Number of repeated AIME sample runs to launch in parallel for direct pass@1 use | `1` |

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

### Smoke mode

```bash
./evals/scripts/run_deepseek_external.sh smoke
```

This exercises the final-table suite flow with tiny limits. It runs AIME24 and
AIME25 with two samples over one problem each, then one or two examples from the
other benchmarks. The runner prints a compact `Benchmark | Measured | Reference`
table and writes `summary.md` / `summary.json` under `OUTPUT_DIR`.

Resume a run by reusing the same output directory:

```bash
OUTPUT_DIR=eval_results/deepseek_smoke ./evals/scripts/run_deepseek_external.sh smoke
```

### Sanity mode

```bash
./evals/scripts/run_deepseek_external.sh sanity
```

This runs the same five short AIME24 questions used by
`run_aime24_short_external.sh`, but forces `MAX_CONCURRENT=1` and
`MAX_GEN_TOKS=32768` for a gentle live-endpoint check. It exits with code `0`
only when all five questions are answered correctly. Any lower score exits with
code `1` and prints the output directory for debugging. Unlike the broader
suites, sanity mode always reruns its benchmark so it reflects the current live
endpoint rather than cached results.

### Quick mode

```bash
./evals/scripts/run_deepseek_external.sh quick
```

This runs the same benchmark set as `full`, but with reduced limits: AIME24 and
AIME25 use two samples over 16 problems each, MMLU uses one example per subject,
MMLU-Pro uses 16 examples per category, and the remaining tasks use 16 examples.

### Single mode

```bash
./evals/scripts/run_deepseek_external.sh single
```

This runs full AIME24 once and reports pass@1 plus the derived majority row.

### AIME24 pass@1 x16

```bash
./evals/scripts/run_aime24_pass1x16_external.sh
```

This runs full AIME24 16 times with distinct API seeds, then writes:

- `r1_aime24_pass1_summary.json`
- `r1_aime24_pass1_summary.md`

Set `AIME_PASS1_TASK=r1_aime25` and
`AIME_PASS1_INCLUDE_PATH=evals/custom_tasks/r1_aime25` to run the local AIME25
task directly.

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

### Full / Suite mode

```bash
./evals/scripts/run_deepseek_external.sh full
```

This runs:

- `r1_aime24` pass@1 x16 and majority@1
- `r1_aime25` pass@1 x16 and majority@1
- derived AIME24+25 pass@1 and majority@1
- `r1_math500`
- `mmlu_generative`
- `mmlu_pro`
- `gsm8k`
- `arc_challenge_chat`
- `humaneval_instruct`
- `r1_gpqa_diamond`
- `mbpp_instruct`

`suite` is accepted as an alias for `full`.
Reference scores in the summary table come from the BitSculpt DeepSeek-R1-0528
report, with AIME24/AIME25 split rows from the sampled AIME reference artifact:
<https://github.com/tenstorrent/bit_sculpt/blob/main/results/deepseek-r1-0528/reports/r1_0528_compression_campaign_2026-05.md> and
<https://github.com/tenstorrent/bit_sculpt/blob/main/results/deepseek-r1-0528/r1_30_archive/reference_evals/aime_sampled_baseline.json>.

To run only selected benchmarks, pass comma-separated benchmark keys:

```bash
./evals/scripts/run_deepseek_external.sh quick --benchmarks aime24_pass1,math500,mmlu_pro
```

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

- `run_deepseek_external.sh` is a thin entrypoint for
  `evals/deepseek_external_runner.py`, which owns suite selection, benchmark
  resume checks, and the final summary table.
- `helper_external_lm_eval.sh` is the shared helper underneath each individual
  `lm_eval` invocation.
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
  deliberately overriding or adding settings. This 64K generation budget is
  applied uniformly by the external helper to every benchmark in the suite.
- Non-streaming requests are the default (`STREAM=0`) and use
  `MAX_CONCURRENT=30`. Set `STREAM=1` or pass `--stream` to add `stream=true`
  to generation kwargs and enable the streaming response parser from
  `evals/scripts/lm_eval_streaming_patch/`.
  Streaming can help with idle connection timeouts, but it does not increase
  server capacity. If long reasoning evals hit gateway timeouts or interrupted
  streams, lower `MAX_CONCURRENT`.
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
- For custom AIME runs, add `--include_path` for local task directories such as
  `evals/custom_tasks/r1_aime24_short` and `evals/custom_tasks/r1_aime25`.
- Load `evals/scripts/lm_eval_streaming_patch/` through `PYTHONPATH` for
  external and managed `lm_eval` runs. Besides streaming compatibility, this
  patches the upstream MMLU-Pro `custom-extract` regex to accept common
  DeepSeek-R1 final-answer formats such as `**Answer: D**`,
  `Final Answer: D`, and `\boxed{\text{D}}`.

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
  requests still time out, retry with `STREAM=1`.

## Output

Each suite run writes `lm_eval` results (JSON + per-sample logs) under
`OUTPUT_DIR/<benchmark-key>/`. The compact aggregate table is printed to stdout
at the end and also written to `OUTPUT_DIR/summary.md` and
`OUTPUT_DIR/summary.json`. Individual `lm_eval` scores live in
`OUTPUT_DIR/<benchmark-key>/.../results*.json`. Common metric keys are:

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
- `r1_aime25`: `exact_match,none`
- `r1_gpqa_diamond`: `exact_match,none`
- `r1_math500`: `exact_match,none`
- `mmlu_generative`: `exact_match,get_response`
- `mmlu_pro`: `exact_match,custom-extract`
- `gsm8k`: `exact_match,strict-match`
- `arc_challenge_chat`: `exact_match,remove_whitespace`
- `humaneval_instruct`: `pass_at_1,create_test`
- `mbpp_instruct`: `pass_at_1,extract_code`

If you run `ifeval` separately through `helper_external_lm_eval.sh`, the main
metrics are `prompt_level_strict_acc,none` and `inst_level_strict_acc,none`.

## Custom-task internals

`evals/custom_tasks/r1_aime24_short/` contains:

- `r1_aime24_short.yaml` — clone of `r1_aime24.yaml` with `process_docs`
  pointing at the filter below.
- `utils.py` — re-exports `process_results_math` from the upstream
  `r1_evals` task and defines `filter_short_ids`, which keeps rows whose `id`
  is in `{60, 69, 75, 84, 86}`.

To change which questions are in the short set, edit `SHORT_AIME24_IDS` in
`utils.py`.

`evals/custom_tasks/r1_aime25/` contains an R1-style AIME25 task over
`math-ai/aime25` using the same prompt and math scorer shape as `r1_aime24`.
