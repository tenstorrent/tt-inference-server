# GPU references on Google Colab

This produces `gpu_reference_score` values for `reference_config/evals/eval_config.py`. On a Colab GPU VM, upstream vLLM serves each model, and TTIS's own `run.py --workflow evals --tt-device gpu` evaluates it. That means the same task configs, context, concurrency and HF revision as the Tenstorrent run, so the reference is like-for-like. (In #5343 we found that the Open LLM Leaderboard's log-likelihood `leaderboard_mmlu_pro` is not comparable to TTIS's generative `mmlu_pro`.)

## Quick start

```bash
uv tool install google-colab-cli && colab sessions     # one-time sign-in (paste the code back)
scripts/gpu_reference_colab/colab_gpu_reference.sh --gpu H100,A100 \
    upstage/SOLAR-10.7B-Instruct-v1.0 meta-llama/Llama-3.2-1B Qwen/Qwen1.5-0.5B-Chat
```

Results go to `workflow_logs/gpu_reference_colab/<session>/`. `summary.txt` holds the per-task scores, and `results/<org>__<name>/provenance.json` records how each score was produced.

To record a reference:

1. Review the run. `status.json` should say `ok`, and `provenance.json` should show the expected sha, revision and GPU.
2. Post the scores and provenance to a GitHub issue. Existing references use issue #5353 and append to it.
3. Set each task's `gpu_reference_score` to its score and `gpu_reference_score_ref` to that issue's URL. Leave the tolerances alone.

## Options

| Option | Default | Meaning |
|---|---|---|
| `MODEL...` | (required) | HF repo ids, run one after another on one VM |
| `--gpu G[,G...]` | `H100` | GPU types in preference order (T4, L4, G4, H100, A100) |
| `--high-mem` | off | high-RAM machine shape |
| `--ttis-ref REF` | `HEAD` | TTIS commit the VM checks out. It must be pushed to `origin`. |
| `--out DIR` | `workflow_logs/gpu_reference_colab/<session>` | local results directory (git-ignored) |
| `--keep` | off | do not stop the VM on exit |
| `--session NAME` | `gpuref-<sha[:8]>` | session name. Re-running with the same name re-attaches. |
| `--poll-minutes N` | `5` | minutes between status polls |
| `--min-balance CU` | off | stop, keeping partial results, once `colab usage` falls below this |
| `--vllm-version V` | per model | override every model's serving stack vLLM (see Serving stack per model) |
| `--transformers-version V` | per model | override every model's transformers pin |
| `--task MODEL=T[,T]` | all tasks | explicit runs: run only these lm-eval tasks of MODEL, each in full (e.g. only the tasks missing a reference); repeatable. Passed to run.py as `--eval-samples '{"T": null}'` and recorded in provenance.json (`tasks_selected`) |
| `--retry-capacity M,H` | off | on 503 (no GPU) or 412 (concurrent-session limit), retry every M minutes for up to H hours, then exit 75 |
| `--detach` | off | run in the background; see Unattended runs |
| `--dry-run` | off | local preflight, then print every `colab` command instead of running it |
| `--sweep` and its sweep options | off | see "Sweep" below |

The driver runs these steps in order: preflight, start the VM, upload the token, launch the runner detached (`setsid`), poll, then download and summarise. An `EXIT` trap stops the VM on every exit path unless you pass `--keep`.

Each poll is a short kernel execution, so polling also keeps the session alive.

Checks and parsers live in `gpuref.py`, and the code the driver sends with `colab exec` lives in `snippets/`. `remote_runner.sh` runs on the VM: it clones TTIS at the exact sha, installs pinned vLLM, and serves and evaluates one model at a time.

**Re-attach and stop.** Re-run the same command to re-attach: it attaches to a running runner, downloads a finished one, or relaunches a dead one. Use `--keep` if you want to detach. `colab sessions` lists the VMs and `colab stop -s <session>` stops one by hand.

## Sweep

```bash
scripts/gpu_reference_colab/colab_gpu_reference.sh --sweep --dry-run      # targets, why, GPU group
scripts/gpu_reference_colab/colab_gpu_reference.sh --sweep --min-balance 4000 \
    --retry-capacity 15,8 --tt-reports <dir with Shield report JSONs> --detach   # unattended
```

`--sweep` reads its targets from the catalog; no model list lives in code.

**What gets selected.** An EvalConfig task is a target if either:

- it has neither `published_score` nor `gpu_reference_score` (ungated); or
- it sets `gpu_reference_requested="<why>"`. This is a reviewed request to measure it, even over an existing reference.

`--refresh` also re-measures tasks that already have a GPU reference.

`--refresh-quetzal origin/main,<PR head>,...` audits every lm-eval task of every model that has a Quetzal P300X2 row, here or on those refs, whatever its current references (use `.` for this branch only). The plan prints the estimated hours and CU per group. Set `--min-balance` to your budget floor.

The report compares the Shield TT score against both the current and the proposed reference, using TTIS's own ratio and noise rules: gpuref calls `binomial_noise_n` and `noise_aware_check` from `workflow_module/eval_noise.py`, the module `test_module/llm_tests/llm_eval_tests.py` grades with. So `leaderboard_math_hard` (an unweighted mean of 7 subtasks) uses TTIS's effective n, not the sample count, and non-binomial tasks are graded by the ratio rule alone. It lists every TT pass that would fail, and every failure that would pass.

**What gets skipped** (`--dry-run` shows the reason for each):

- tasks that are not lm-eval (`EVALS_COMMON`) tasks;
- models with no GPU spec and no TT row to derive one from;
- models whose bf16 weights fit no Colab GPU.

**GPU specs.** A target model without a GPU spec gets one derived from its TT row: its Quetzal row first, otherwise its default TT rows. It keeps the row's `max_concurrency`, `max_context` is capped at the native `max_position_embeddings`, and the HF revision is pinned. An explicit run (or a sweep) from a pushed commit with no GPU spec for the model also works without committing one: the runner derives the spec on the VM from that commit's TT row, merges it into the VM's clone only, and records it as `derived_gpu_spec` in `provenance.json`. So a model whose row lives on a dispatch branch is one command: `--ttis-ref <that branch> <model>`.

If the HF config has a `vision_config` (a multimodal checkpoint whose TT row serves the text decoder), the spec sets `limit_mm_per_prompt` to zero images and videos, so vLLM skips the vision encoder's memory profiling; text prompts are processed identically.

If the HF config carries `dual_chunk_attention_config` (the Qwen2.5-*-1M models), the spec adds `hf_overrides` that remove it. Quetzal lowers the HF transformers graph, which ignores that setting, so standard attention on the GPU is the like-for-like reference.

`--write-specs` merges by model: the models this run selected are added or rewritten, and every other generated entry is kept as it is. `--prune` also drops the generated entries of models this run did not select. A change to a derivation rule reaches every derived spec through a run that selects them all, such as `--refresh-quetzal`.

1. `--write-specs` merges the derived entries into a generated block at the end of `workflows/model_specs/dev/llm.yaml`.
2. Review that change, commit it and push it. The VM runs the pushed sha.

**Placement and sessions.** The memory preflight places each model:

- on `A100` if its bf16 weights fit a 40 GB A100, even if only with a `max_model_len` cap. There is no H100 fallback: an A100 bills about 5.3 CU/hr against about 18 for an H100, and the scores are valid on either;
- on `H100` if it needs 80 GB.

Each group runs in one session, named `gpuref-sweep-{a100,h100}-<sha8>`.

**Group fallback.** If a group gets no GPU (Colab returns 503, or 412 at the concurrent-session limit), its driver exits 75. The sweep defers that group, runs the others, and retries deferred groups at the end, waiting with `--retry-capacity` when it is set.

**Resume.** Each finished model is downloaded as soon as it completes. Re-running the same command, which is recorded in `workflow_logs/gpu_reference_colab/RESUME.txt` with its sha pinned, does two things:

- skips models whose latest `ok` run used exactly the same `vllm serve` arguments (revision, context, flags; `--max-num-seqs` aside), has a score for every target task, and has the same **eval fingerprint**: a sha256 over each lm-eval task's score-affecting `EvalTask` fields after GPU device overrides (`num_fewshot`, `gen_kwargs`, chat template and few-shot layout, thinking, ...), its scorer, the lm-eval commit and the vLLM version. The runner records the fingerprint of what actually ran (installed lm-eval and vLLM) in `provenance.json`. Runs recorded before fingerprints, or with another one, are re-run, and the report never proposes their scores;
- re-attaches to a session that is still running.

Resume is idempotent across commits: reuse is decided per model by the fingerprint, not by the sha.
- **At a newer commit** (`--ttis-ref <new sha>`), a model whose eval settings, lm-eval commit and vLLM version are unchanged keeps its finished run, and a model whose settings changed is re-run.
- **When not to:** runs recorded by tooling older than fingerprints have none, so a newer commit re-runs every one of them. The plan prints `NOTE: N model(s) have ok runs recorded without an eval fingerprint`. To keep such a sweep's results, finish it, and report on it, with the commit it started at.

`--min-balance` is checked before each group and at every poll. Set `GPUREF_RESULTS_ROOT` to keep results (and `RESUME.txt`) outside the checkout, for example when the checkout sits under a `/tmp` that is cleared on reboot.

**Outputs** (`workflow_logs/gpu_reference_colab/sweep-<sha8>/`, written by `gpuref.py publish` at the end of the sweep):

- `sweep_plan.json`;
- `sweep_summary.json`: GPU score, samples, TT score, cap and longest request per task;
- `sweep_5353.md`: the #5353 comment, including each stack (vLLM, transformers, server env);
- a proposed `eval_config.patch`. It records new references and clears `gpu_reference_requested`.
  - Where a task already has a GPU reference, the patch replaces it only if the new score differs by more than TTIS's noise rule (1.96 binomial SE at the old score). Otherwise it keeps the old reference and just clears the flag.
  - **It is never applied automatically**: review it, then `git apply` it.

Only whole models can be run, because `run.py` evaluates every task in a config. Any non-target tasks are measured too, but they are not proposed.

## Unattended runs

One command runs a whole sweep in the background, with nothing else to start or watch:

```
GPUREF_RESULTS_ROOT=~/gpuref-results scripts/gpu_reference_colab/colab_gpu_reference.sh --sweep \
    --refresh-quetzal origin/main --min-balance 4400 --retry-capacity 15,8 \
    --tt-reports ~/gpuref-results/shield_reports --detach
```

- **`--detach`** runs the driver in its own session and returns. Its output goes to `$GPUREF_RESULTS_ROOT/gpuref_run.log`.
- **`--retry-capacity 15,8`**: when Colab has no GPU free (503) or the concurrent-session limit is hit (412), the driver tries again every 15 minutes for up to 8 hours. Then it exits 75, and a sweep moves on (see Group fallback).
- **Status, for people and agents.** Read these; there is no need to tail the driver:
  - `$GPUREF_RESULTS_ROOT/gpuref_status.log`: one line per state change of every driver, as `<time> state=... session=... done=3/26 phase=...`.
  - `<session dir>/driver_status.json`: the driver's latest state. States are `STARTING`, `WAITING_CAPACITY`, `RUNNING`, `DONE`, `FAILED`, `NO_CAPACITY` and `LOW_BALANCE`.
  - `sweep-<sha8>/driver_status.json`: the sweep's state, from `SWEEP_PLANNING` and `SWEEP_GROUP` / `DEFERRED` to `SWEEP_DONE` or `SWEEP_INCOMPLETE`.
  - `<session dir>/results/status.json`: per-model status from the VM.
- **Exit status:** 0 when everything finished, 75 for no GPU capacity (retry later), 1 otherwise. Partial results are always kept.

**Serving stack per model.** Each model gets a stack: a vLLM version, an optional transformers pin, and a server environment. It comes from `gpuref.py serving_stack` and is chosen by the model's HF config `architectures`:
- **Default:** vLLM 0.13.0 (TTIS's own client pin), transformers unpinned, `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **`ARCH_STACKS`:** maps an architecture vLLM 0.13.0 cannot serve to the first vLLM release that supports it, and to the transformers the TT side lowers it with. Today:
  - Muse-Glimmer: vLLM 0.28.0, transformers 5.15.0;
  - Gemma 4: vLLM 0.20.0, transformers 5.5.3 (the version vLLM 0.20.0 is tested with; 5.15.0 fails to start it).

The plan prints each model's stack. The runner builds one venv per stack on first use, so one sweep handles mixed stacks in a single session. The serve plan, `provenance.json` (`stack`, installed `packages`, `vllm_env`) and the eval fingerprint all record the stack. A run's fingerprint is kept only if the installed packages are the requested stack.

To add an architecture, add one `ARCH_STACKS` entry citing the vLLM PR that added it. `--vllm-version` / `--transformers-version` override every model's stack, for a one-off explicit run.

## Publishing results

`gpuref.py publish` renders the #5353 comment and the proposed patch from results. It never posts or applies anything unless asked:

```
# a sweep (strict: only runs whose fingerprint matches the plan)
python3 scripts/gpu_reference_colab/gpuref.py publish --plan $R/sweep-<sha8>/sweep_plan.json \
    --results-root $R --tt-reports $R/shield_reports
# explicit runs, patching the eval_config.py of the branch that will carry them
python3 scripts/gpu_reference_colab/gpuref.py publish --models meta-models/Muse-Glimmer-30B \
    --results-root $R --out-dir $R/muse --eval-config <that branch's eval_config.py>
```

After the references are approved, add **`--post`**. It posts `sweep_5353.md` to #5353 with `gh` and re-renders `eval_config.patch` so every reference cites that comment.
- It refuses a body containing a token-like string.
- It refuses a second post; `publish.json` records the URL.
- The patch still has to be applied by hand.

## HF token

The driver reads the token from `HF_TOKEN` or `~/.cache/huggingface/token`. A token is required if any model is gated, and the preflight checks that it can read each gated repo.

1. The token is written to a mode-0600 file in a mode-0700 `mktemp -d` directory, using the `printf` builtin or `cp`.
2. `colab upload` sends that file to `/content/gpuref`, and the local copy is then deleted.
3. A snippet with no token in it moves the file to `~/.cache/huggingface/token` (mode 0600). The runner never reads it.

The token never appears on a command line, in a log, or in exec code.

## GPU choice, capacity and model size

Colab answers `503 Service Unavailable` when it has no capacity. Each `--gpu` type gets 3 attempts with 120 s and 240 s backoff before the driver moves to the next type. Any other refusal moves on at once.

**Cost.** An A100 bills about 5.3 CU/hr and an H100 about 18 CU/hr (measured 2026-10-09). The scores are valid on either GPU, so the sweep runs every model whose weights fit an A100 on an A100 only. Check the rate with `colab usage`. Batch 1 took 2.4 h and 11.6 CU: SOLAR-10.7B, Llama-3.2-1B and Qwen1.5-0.5B on an A100, with most of the time spent on MMLU-Pro CoT.

The scores are bf16 vLLM references and valid on either GPU. The GPU you actually got is recorded in `provenance.json` and in the `gpu` column of the summary.

| bf16 model size | `--gpu` | Notes |
|---|---|---|
| up to ~14B | `H100,A100` | An A100 (40 GB) holds the weights. Long contexts may only fit there with a cap. |
| ~14B to 32B | `H100` | 80 GB. 27-32B models at long contexts get a `max_model_len` cap. |
| 70B+ | not on Colab | Needs a multi-GPU machine. `remote_runner.sh` is host-agnostic. |

Before provisioning, the preflight estimates bf16 weights plus the KV cache for one `max_context` sequence plus about 4 GiB of overhead. The estimate uses the Hub's parameter count and `config.json`. It compares that against 90% of each listed GPU's memory and then:

- refuses a model that fits no listed GPU, with the reason;
- drops GPU types a model cannot use fully, when another listed type can.

If vLLM still refuses `max_context`, the runner retries once at vLLM's own estimate. It records `max_model_len_cap` in `provenance.json` and a note in `status.json`, so review those runs. The prefill chunk (`--max-num-batched-tokens`) is capped at 16384 on GPU. The TT default equals `max_context`, and at 131072 that made vLLM's memory profiling run out of memory on an A100. The chunk size only changes how work is scheduled, not the scores. `max_num_seqs` is never reduced: vLLM queues what does not fit.

For models up to about 8B parameters (9e9 or fewer, from the Hub's safetensors count or `config.json`), the runner serves with `--max-num-seqs 256` and exports `TT_EVAL_CLIENT_CONCURRENCY=256`, so the eval client sends 256 concurrent requests instead of the TT row's `max_concurrency` (often 32), which leaves an A100 mostly idle. From 9e9 to 16e9 parameters it uses as many sequences as the GPU's spare memory holds at 2048 KV tokens each (measured on Qwen3-0.6B `mmlu_pro`), at most 128. That is about 126 for Qwen3-14B on an 80 GB H100; on an A100-40GB it is below the row's value, which then stays. So a 10-16B model whose A100 KV room is below its row's value is placed in the H100 group, where it serves up to 128 sequences at about the same CU in a third of the wall time. `provenance.json` records the reason as `placement`. Larger models keep the row's value. Prompts, sampling and scoring are unchanged. The values and the rule that chose them are recorded in `provenance.json` (`max_num_seqs`, `client_concurrency`, `concurrency_rule`), and resume ignores `--max-num-seqs`.

## Outputs

```
<out>/summary.txt                      status per model + score table
<out>/results/status.json              {"<repo>": {"status": "ok"|"failed", "note": ...}}
<out>/results/runner.log               VM runner log
<out>/results/<org>__<name>/           serve_plan.json, vllm_server.log, run_py.log, provenance.json
<out>/results/workflow_logs/           TTIS eval reports, lm-eval results and samples
```

`provenance.json` records:

- the GPU name, driver and memory;
- the vllm, torch and transformers versions;
- `ttis_sha`;
- the pinned and resolved HF revision;
- the exact `vllm serve` and `run.py` commands;
- `max_context` and `max_concurrency`, plus any `max_model_len_cap`;
- start and end times (UTC), status and exit code;
- the lm-eval commit, both pinned and installed.

## Adding a model

1. Add a `- device: GPU` / `default_impl: true` template to `workflows/model_specs/dev/llm.yaml`. For a Quetzal model, use a separate `impl: quetzal` template next to the other GPU entries, not a device inside the generated row.
   - `max_context`: min(the TT row's `max_context`, the model's `max_position_embeddings`). With no TT row, use the native value.
   - `max_concurrency`: the TT row's value.
   - `vllm_args.revision` / `tokenizer_revision`: the TT row's pins.
2. Add an EvalConfig if the model has none.
3. Run `--dry-run`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Colab CLI not signed in` | Run `colab sessions` once in a terminal. |
| `'colab new' failed on every GPU` | No capacity or entitlement right now. Retry later, or add a GPU type. Check `colab sessions` for leftover VMs. |
| `not on any origin branch` | Push the commit, or pass `--ttis-ref`. |
| `no GPU spec` / `refused` in the preflight | Add the GPU entry (above), or pick a bigger GPU. 70B+ is out of scope. |
| `VM step '<name>' failed` | The printed output shows the traceback from the VM. |
| `vLLM did not become ready` | See `<model>/vllm_server.log`. |
| `run.py exited N` | See `<model>/run_py.log`. |
| `session ... is gone` | Colab reclaimed the VM and the results on it are lost. Re-run. |
