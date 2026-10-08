# GPU references on Google Colab

`colab_gpu_reference.sh` measures a model's TTIS eval scores on a Google Colab
GPU VM (H100 by default) and brings back the results, ready to become the
model's `gpu_reference_score` / `gpu_reference_score_ref` in
`reference_config/evals/eval_config.py`.

## Why

A Tenstorrent eval run is graded against its EvalConfig's reference score. A
published number is only a fair bar when it was measured the same way TTIS
measures. On #5343, reviewer fivanovicTT showed that the Open LLM Leaderboard
`leaderboard_mmlu_pro` numbers (5-shot log-likelihood multiple choice) are not
comparable to TTIS's `mmlu_pro` (5-shot generative chain of thought, letter
extraction), so those gates were dropped. A GPU run of the exact same TTIS task
config is a like-for-like reference: same task, same n-shot, same chat-template
setting, same context window, same concurrency, same HF revision. Only the
hardware and serving stack differ. If a TT score is far below its GPU reference,
the TT serving is the suspect. If the GPU number is just as low, the model is.

The VM does what [docs/gpu_workflows.md](../../docs/gpu_workflows.md) describes
for any GPU host. It serves the model with upstream vLLM on
`http://127.0.0.1:8000/v1` under its HF repo id, then runs TTIS's own
`python run.py --workflow evals --tt-device gpu --model <repo> --dev-mode`
against it. These are full evals: no `--limit-samples-mode`, no smoke caps.

## Prerequisites

- A Colab plan with H100 access (Colab Pro/Pro+ or pay-as-you-go compute units).
- `uv` (https://docs.astral.sh/uv/) and `git`. The driver is bash and runs on
  Linux or macOS. Its Python helper needs `python3` with PyYAML, or else it
  falls back to `uv run --with pyyaml`.
- The Colab CLI, which you install and sign in to once:

  ```bash
  uv tool install google-colab-cli
  colab sessions        # prints a sign-in URL; paste the code back once
  ```

  The token is cached at `~/.config/colab-cli/token.json`. The driver never
  starts the interactive sign-in itself. If the token is missing or expired,
  it stops and asks you to run `colab sessions` again.
- The models must have a `- device: GPU` entry with `default_impl: true` in
  `workflows/model_specs/dev/llm.yaml` (see "Adding a model" below).
- The TTIS commit the VM runs must be pushed to `origin`, because the VM clones
  the public repo at that exact sha.

### HF token

A token is required when any model is gated (for example
`meta-llama/Llama-3.2-1B`). The driver takes it from `HF_TOKEN`, or else from
`~/.cache/huggingface/token` (what `hf auth login` writes). In the preflight,
the driver checks whether each repo is gated and whether the token can read it.

The token never appears on a command line, in a log, or in code sent with
`colab exec`:

1. It is written into a `mktemp -d` directory with mode 0700 and a file with
   mode 0600, using the `printf` builtin or `cp` from the token file.
2. That file goes up with `colab upload` and the local copy is deleted.
3. A small `colab exec` snippet, which contains no token, moves it to
   `~/.cache/huggingface/token` on the VM, with mode 0600 and a 0700 directory.
   The upload lands in `/content/gpuref` first because Jupyter's contents API
   can refuse hidden paths such as `~/.cache`.

On the VM, huggingface_hub, vLLM and lm-eval read the token from that file. The
runner never reads it, exports it or prints it.

## Usage

```bash
scripts/gpu_reference_colab/colab_gpu_reference.sh \
    upstage/SOLAR-10.7B-Instruct-v1.0 meta-llama/Llama-3.2-1B Qwen/Qwen1.5-0.5B-Chat
```

| Option | Default | Meaning |
|---|---|---|
| `MODEL...` | (required) | HF repo ids, run one after another on one VM |
| `--gpu GPU[,GPU...]` | `H100` | Colab GPU types in order of preference (T4, L4, G4, H100, A100), e.g. `H100,A100`; see "GPU choice and capacity" |
| `--high-mem` | off | request the high-RAM machine shape |
| `--ttis-ref REF` | `HEAD` | TTIS commit for the VM; resolved locally to a full sha, which must be on an `origin` branch |
| `--out DIR` | `workflow_logs/gpu_reference_colab/<session>` | local results directory (`workflow_logs/` is git-ignored) |
| `--keep` | off | leave the session running on exit (no `colab stop`) |
| `--session NAME` | `gpuref-<sha[:8]>` | Colab session name; re-running with the same name attaches |
| `--poll-minutes N` | `5` | minutes between status polls |
| `--vllm-version V` | runner pin (`0.13.0`) | override the vLLM pin, only to work around a VM problem |
| `--dry-run` | off | run the local preflight, then print every `colab` command instead of running it |

For the first three models, `--gpu H100,A100` is a good choice. H100
capacity on Colab comes and goes.

Run it from a TTIS checkout. The preflight reads that checkout's catalog, and
the VM reads the `--ttis-ref` commit's catalog. The driver warns if the two
differ.

## GPU choice and capacity

Colab GPU capacity varies by hour and by account. When it has none, the assign
call returns `503 Service Unavailable`. `--gpu` takes an ordered preference
list, and the driver walks it:

- each type gets 3 attempts with 120 s, then 240 s backoff while it keeps
  answering 503;
- any other refusal (quota or entitlement) moves on to the next type
  immediately;
- the driver gives up only when every listed type has failed.

The type you got is logged, and the exact GPU (`nvidia-smi` name, driver and
memory) goes into each `provenance.json` and into the `gpu` column of
`summary.txt`.

The scores are bf16 vLLM references with greedy or seeded decoding and the
same task configs, so they are valid on either an H100 or an A100. Small
run-to-run differences come from kernel and batching nondeterminism, not from
the GPU type. Still, cite the GPU from `provenance.json` with the number.

Before creating a session, the preflight estimates each model's bf16 serving
memory from the Hub's parameter count and `config.json`: weights, plus the KV
cache for one `max_context` sequence, plus about 4 GiB of overhead. It checks
that against 90% of each listed GPU's memory (H100 80 GB, A100 40 GB, L4, T4).
A type that cannot hold a model is dropped from the list. For the first three
models: SOLAR-10.7B needs about 25 GiB (weights 20 GiB, KV 0.75 GiB per 4K
sequence), Llama-3.2-1B about 10 GiB at 128K, and Qwen1.5-0.5B about 8 GiB at
32K. All of them fit a 40 GB A100. With 32 concurrent sequences, vLLM queues
whatever the KV pool cannot hold, which costs throughput but not correctness.

### Which models go on which GPU

References are bf16 only, because quantized references are out of scope. Each
Colab runtime has one GPU.

| bf16 model size | GPU | Notes |
|---|---|---|
| up to ~14B (weights up to ~28 GB) | `--gpu H100,A100` | Fits a 40 GB A100 at typical contexts. The preflight drops A100 if a long `max_context` does not fit there. |
| ~14B to 32B | `--gpu H100` only | A 40 GB A100 cannot hold the weights. For 27B-32B, 80 GB leaves little KV room, so long contexts may need a `max_model_len` cap (below). |
| 70B and above | out of scope on Colab | About 140 GB of bf16 weights needs a multi-GPU machine. `remote_runner.sh` is host-agnostic, so it could later be pointed at one over SSH. |

The preflight enforces this table before anything is provisioned. It reads the
parameter count from the Hub's safetensors metadata, falling back to
`config.json`. A model whose bf16 weights do not fit the largest GPU in
`--gpu` is refused with an explanation. A run that mixes small and 27B-32B
models drops A100 from its list. If no listed GPU can hold every model, split
the models into separate runs.

For a model that fits H100 only with a cap (for example Qwen2.5-32B at a 32K
`max_context`, which needs about 73 GiB against 72 GiB usable), the preflight
prints the expected cap.

`max_num_seqs` is never reduced. vLLM queues the requests its KV pool cannot
hold, which affects throughput, not results.

If vLLM still refuses `max_context` on the GPU it got, the runner retries once
at vLLM's own estimated maximum length. It records `max_model_len_cap` in
`provenance.json` and a note in `status.json`. lm-eval still sizes prompts to
the spec's `max_context`, so review a capped run before using its numbers.

## What happens

1. **Preflight (local).** Resolves `--ttis-ref` to a sha and checks it is
   pushed. Resolves each model's GPU `DeviceModelSpec` with TTIS's own loader.
   This is the same lookup `run.py --dev-mode --tt-device gpu` does. Then it
   checks HF gating and token access, and checks that `colab` is installed and
   signed in.
2. **Session.** `colab new -s <session> --gpu <type> [--high-mem]`, walking
   the `--gpu` preference list. If
   `colab sessions` already lists `<session>`, the driver reads the runner state
   on it instead (see "Resume").
3. **Upload.** A `prepare` snippet creates `/content/gpuref` and
   `~/.cache/huggingface`. Then the HF token goes up (see above), followed by
   `remote_runner.sh` and `gpuref.py`.
4. **Detached runner.** A `launch` snippet starts `remote_runner.sh` with
   `subprocess.Popen(..., start_new_session=True)`, which is `setsid`. The
   runner has its own session and process group, with no controlling terminal
   or stdin, so a dropped CLI connection or a finished `colab exec` does not
   stop it. On the VM, the runner:
   - clones TTIS at the exact sha (`git fetch --depth 1 origin <sha>`) and
     verifies `HEAD`;
   - makes sure `python3 -m venv` works (installing `python3.X-venv` if it
     doesn't), along with PyYAML and uv. After that, `run.py` bootstraps its own
     venvs, as the workflows user guide describes;
   - installs pinned `vllm==0.13.0` into its own venv, away from Colab's torch;
   - for each model:
     1. reads the GPU spec through `gpuref.py serve-plan`, which uses the TTIS
        loader;
     2. starts `vllm serve <repo> --served-model-name <repo> --port 8000
        --max-model-len <max_context> --max-num-seqs <max_concurrency> --dtype
        bfloat16` plus the spec's `vllm_args`, minus TT-only keys;
     3. waits up to 45 min for `/v1/models` to list the repo;
     4. runs `run.py --workflow evals --tt-device gpu --model <repo>
        --dev-mode`;
     5. stops the server and writes `provenance.json`.

     If a model fails, the runner records the failure and moves on to the next
     model.
   - writes `DONE` (every model OK) or `FAILED` (setup failed, or any model
     failed).
5. **Poll.** Every `--poll-minutes`, a short `status` snippet prints the
   phase, GPU utilisation and the runner's log tail. Colab keeps a session
   alive while its kernel is active, and these polls are kernel executions, so
   they keep the VM up. After 6 failed polls in a row, the driver checks
   `colab sessions` to see whether the VM is gone.
6. **Download.** On `DONE` or `FAILED`, a `pack` snippet tars
   `/content/gpuref/results`. The driver runs `colab download`, extracts the
   tarball into `--out`, and prints a per-model, per-task score table parsed
   from the TTIS eval report JSON (`gpuref.py summary`). It also saves the table
   to `summary.txt`.
7. **Stop.** An `EXIT` trap runs `colab stop -s <session>` on every exit path
   (success, failure, Ctrl-C) unless `--keep` is set. The driver exits 0 only
   on `DONE`.

`--dry-run` runs step 1 except the `colab` checks, then prints steps 2 to 7 as
`+ colab ...` lines.

## Outputs

```
<out>/
  gpuref-results.tar.gz
  summary.txt                      # the score table printed at the end
  results/
    status.json                    # {"<repo>": {"status": "ok"|"failed", "note": ...}}
    runner.log                     # the VM runner's log
    <org>__<name>/
      serve_plan.json              # resolved GPU spec and the exact vllm serve argv
      vllm_server.log
      run_py.log                   # run.py --workflow evals output
      provenance.json
    workflow_logs/                 # TTIS workflow_logs from the VM
      reports_output/evals/        # report_*.md and data/report_data_*.json (the graded scores)
      ...                          # lm-eval results_*.json and per-sample logs, run logs
```

`provenance.json` records:

- `gpu`: name, driver and memory, from `nvidia-smi`
- `packages`: vllm, torch and transformers versions
- `ttis_sha`
- `model_revision_pinned` (from the spec) and `model_revision_sha` (resolved on the Hub)
- the exact `vllm_serve_command` and `run_py_command`
- `max_context` and `max_concurrency`
- `started_at` / `finished_at` (UTC)
- `status` and `evals_exit_code`
- `lm_eval_commit_pinned` (from `requirements/evals-common.txt`) and
  `lm_eval_commit_installed` (from the evals venv's `direct_url.json`)

## Resume, attach and stop

- **Re-attach.** Run the same command again. The session name defaults to
  `gpuref-<sha[:8]>`, so the second invocation finds the session in
  `colab sessions` and reads the runner state on the VM:
  - `RUNNING`: the driver attaches and polls.
  - `DONE` or `FAILED`: the driver goes straight to download.
  - No runner, or a runner that died without a marker: the driver launches a
    new one.

  Use `--keep` if you mean to detach. Without it, Ctrl-C stops the VM.
- **Inspect.** `colab sessions`, `colab status -s <session>`, and
  `echo 'print(open("/content/gpuref/runner.log").read()[-4000:])' | colab exec -s <session>`.
- **Stop by hand.** `colab stop -s <session>`. Finish with `colab sessions`,
  which should report no active sessions. An idle VM still burns compute units.

## Time and cost

Rough H100 timings for the first three models, all at TTIS's full task sizes
(ifeval 541 prompts, MATH-Hard 1,324, MMLU-Pro 12,032; 32 concurrent):

| Step | Wall time |
|---|---|
| VM setup (clone, vLLM install, first `run.py` venvs) | ~15-25 min |
| SOLAR-10.7B-Instruct (4K context) | ~1.5-2.5 h |
| Llama-3.2-1B (base, long CoT generations) | ~0.5-1.5 h |
| Qwen1.5-0.5B-Chat | ~0.5-1 h |

Most of the time goes to the MMLU-Pro chain-of-thought generations. Check the
balance with `colab usage` before and after. While a session runs, its `Usage
rate:` line shows the H100's compute units per hour. Total cost is about rate x
wall hours. A 40 GB A100 measured about 5.3 CU/hr (2026-10-08). An A100 run
takes noticeably longer than the H100 timings above.

## Turning results into references

1. Read the score for each task from `summary.txt`, or from
   `results/workflow_logs/reports_output/evals/data/report_data_*.json`
   (`sections[].data.score`, already in the task's unit).
2. Review the result before you use it. Check that `status.json` says `ok`,
   that `provenance.json` has the expected sha, revision and vLLM version, and
   that `run_py.log` shows no truncation or server errors.
3. In `reference_config/evals/eval_config.py`, set the task's
   `gpu_reference_score` to that number and its `gpu_reference_score_ref` to a
   durable citation. Use a GitHub issue or PR comment that attaches
   `provenance.json` and the report JSON. The existing references use this
   form, for example
   `https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051`.
   You can also give a link to the committed provenance file at a fixed sha.
4. Leave the tolerances alone. Never choose a reference to make a known TT
   score pass. The GPU number stands on its own.

## Adding a model

Add a GPU entry to the model's dev spec (`workflows/model_specs/dev/llm.yaml`).
`run.py --dev-mode --tt-device gpu` must resolve exactly one default GPU leaf
for that repo. For a Quetzal row, add a separate `impl: quetzal` template
rather than a device inside the generated row (see the GPU block after the
Quetzal rows):

- `max_context` = min(the TT row's `max_context`, the model's native
  `max_position_embeddings`), so lm-eval truncates the same way;
- `max_concurrency` = the TT row's (lm-eval's `num_concurrent`);
- `vllm_args.revision` / `tokenizer_revision` = the TT row's pins.

Then check with `--dry-run`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `the Colab CLI is not signed in` | Run `colab sessions` once in a terminal and finish the sign-in. |
| `'colab new' failed on every GPU` / `Allocation refused` | No quota, entitlement or capacity for any listed type right now (each type gets 3 attempts on `503 Service Unavailable`). Retry later, or add a type such as `--gpu H100,A100`. Run `colab stop` on any leftover sessions. |
| `is not on any origin branch` | Push the branch, or pass `--ttis-ref` with a pushed commit. |
| `no GPU spec: No model spec matches ... device='GPU'` | Add the GPU entry (above). |
| `needs an HF token` / `cannot read` | Set `HF_TOKEN` or run `hf auth login`, and accept the model's license on the Hub. |
| `VM step 'launch' failed` | The snippet raised on the VM. The printed output has the traceback. |
| `vLLM did not become ready` in `status.json` | See `<model>/vllm_server.log` (out of memory, unsupported architecture, download error). |
| `run.py exited N` | See `<model>/run_py.log`. A connection refused there means the server died mid-run (check `vllm_server.log`). |
| `session ... is gone` | Colab reclaimed the VM (idle or quota). Results on the VM are lost, so re-run. |
| `colab sessions` shows a `[?]` session | An assignment with no local record. Stop it from the Colab UI, or re-create local state and `colab stop`. |
