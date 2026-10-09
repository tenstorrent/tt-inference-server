# GPU Report Generation

The `reports` workflow (and the upstream `benchmarks` / `evals` workflows) can target `--tt-device gpu` to consume or produce data against a non-Tenstorrent (CUDA/NVIDIA) inference backend. This guide covers the extra steps required for the GPU path — they differ from the standard Tenstorrent device flow.

## Required: add a GPU `DeviceModelSpec`

Find the `ModelSpecTemplate` for your model in `workflows/model_spec.py` and add a GPU entry to `device_model_specs`. Example pattern (matches the existing Llama-3.1-8B-Instruct entry — grep `model_spec.py` for `DeviceTypes.GPU` to see it in context):

```python
DeviceModelSpec(
    device=DeviceTypes.GPU,
    max_concurrency=32,
    max_context=128 * 1024,
    default_impl=True,
),
```

Setting `default_impl=True` lets the runtime resolver pick this device entry without requiring an explicit `--impl` flag on the CLI.

Optionally, add GPU performance reference values to `model_performance_reference[<model>]["gpu"]` for performance-target rendering. Without them you'll see `No performance targets found for model '<model>' on device 'gpu'` warnings and blank target columns — non-blocking.

## Running workflows on GPU

There is an implicit dependency for any workflow that drives inference on GPU (`benchmarks`, `evals`, `release`): **you must bring-your-own-server and point the workflows to it.** The pipeline has no GPU server-launch automation — start a vLLM (or OpenAI-compatible) server yourself on `http://127.0.0.1:8000/v1` before invoking these workflows. The `reports` workflow alone is file-based and does not require a running server.

### Benchmarks / Evals
Note: needs running vLLM

```bash
python run.py --workflow benchmarks --tt-device gpu --model <Model> 
python run.py --workflow evals      --tt-device gpu --model <Model> 
```

### Reports only — no server required

If benchmark and/or eval output files already exist on disk for this model+device, you can render reports without any running server:

```bash
python run.py --workflow reports --tt-device gpu --model <Model> 
```

The workflow reads files from:

- `workflow_logs/benchmarks_output/benchmark_*_gpu_*.json` (benchmark reports)
- `workflow_logs/evals_output/eval_id_<impl>_<model>_gpu/` (eval reports)

If a directory or matching file is missing, that report section logs `Skipping.` and the workflow continues.



### Release workflow

```bash
python run.py --workflow release --tt-device gpu --model <Model>
```

Runs `evals` → `benchmarks` → `tests` → `reports` in sequence, all against the external vLLM. Use this for full release certification once individual workflows verify correctly.

## Collecting GPU references on Google Colab

[`scripts/gpu_reference_colab/`](../scripts/gpu_reference_colab/README.md) automates the bring-your-own-server flow above on a Google Colab GPU VM, using the [Colab CLI](https://github.com/googlecolab/google-colab-cli). For each model, the VM serves the HF repo with pinned upstream vLLM under its repo id and runs `run.py --workflow evals --tt-device gpu --dev-mode` at a pushed TTIS sha. These are full evals with no limits. The VM then sends back the eval reports and a `provenance.json`. The result is a like-for-like `gpu_reference_score`: the same task configs, context window, concurrency and HF revision as the Tenstorrent run.

- **Prerequisites:** a Colab plan with H100 access, `uv`, the model's `- device: GPU` / `default_impl: true` entry in `workflows/model_specs/dev/llm.yaml`, and an HF token (`HF_TOKEN` or `~/.cache/huggingface/token`) for gated repos.
- **One-time sign-in:** `uv tool install google-colab-cli && colab sessions`, then complete the sign-in.
- **Run:** `scripts/gpu_reference_colab/colab_gpu_reference.sh --gpu H100,A100 upstage/SOLAR-10.7B-Instruct-v1.0 meta-llama/Llama-3.2-1B Qwen/Qwen1.5-0.5B-Chat`. `--gpu` is an ordered preference list, because Colab GPU capacity varies. The scores are bf16 vLLM references and are valid on either GPU, and the GPU the run got is recorded in `provenance.json`. Add `--dry-run` to see every `colab` command first. The script polls a detached runner and stops the VM on exit unless you pass `--keep`, and re-running the same command re-attaches.
- **Sweep:** `colab_gpu_reference.sh --sweep [--dry-run] [--emit-patch]` picks its targets from the catalog instead of from a model list. It selects every ungated EvalConfig task, plus every task flagged with `gpu_reference_requested="<why>"`.
  - **GPU specs:** a missing GPU spec is derived from the model's TT row; `--write-specs` writes it.
  - **Placement:** each model goes into an A100-capable group or an H100-only group.
  - **Resume:** finished models are downloaded as they complete, and re-running the same command resumes.
  - **Outputs:** `sweep_summary.json`, a #5353 block, and a proposed `eval_config.patch` that is never applied automatically.
- **Model sizes:** references are bf16 on one Colab GPU. Models up to about 14B can use `--gpu H100,A100`, and 27-32B models need `--gpu H100`. 70B is out of scope on Colab because it needs a multi-GPU machine. The preflight refuses a model whose bf16 weights do not fit the largest listed GPU, and it reports any `max_model_len` cap.
- **Time and cost:** on one H100 (an A100 is slower), about 15-25 min of setup plus roughly 0.5-2.5 h per model. Most of that is MMLU-Pro chain-of-thought generation. Compute units are the H100 rate (`colab usage` while it runs) times wall hours.
- **Outputs:** `workflow_logs/gpu_reference_colab/<session>/` holds `summary.txt` (per-model, per-task scores), the TTIS eval reports, and `provenance.json`. The provenance file records the GPU, driver, vLLM/torch/transformers, TTIS sha, model revision, exact `vllm serve` args, timestamps and the lm-eval commit.
- **Using the results:** set the task's `gpu_reference_score` in `reference_config/evals/eval_config.py` to the reported score. Set `gpu_reference_score_ref` to a GitHub issue or PR comment that attaches `provenance.json` and the report JSON, or to the committed provenance file at a fixed sha. Leave the tolerances unchanged.

See the [README](../scripts/gpu_reference_colab/README.md) for the step-by-step flow, resume and stop, the outputs layout and troubleshooting.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Model:=<M> does not support device:=gpu` | Model missing `DeviceModelSpec(device=DeviceTypes.GPU, ...)` | Add the entry to the model's `ModelSpecTemplate` in `workflows/model_spec.py`, with `default_impl=True`. |
| `Error code: 404 — The model '<repo>/<name>' does not exist` (during evals/benchmarks) | vLLM is running but registered under a different name | `curl /v1/models` to inspect; restart vLLM with the correct `--served-model-name`. |
| `NotImplementedError: GPU support for running inference server not implemented yet` | Passed `--docker-server` or `--local-server` with `--tt-device gpu` | Drop those flags — start vLLM yourself instead. |
| Connection refused on port 8000 (during evals/benchmarks/release) | No server running | Start vLLM before launching the workflow, or use `--workflow reports` only (file-based, no server). |
| `No performance targets found for model '<M>' on device 'gpu'` | No GPU entry in `model_performance_reference` | Optional. Add an entry to populate the targets column, or ignore. |

## See also

- [Workflows User Guide — Reports](workflows_user_guide.md#reports)
- [`workflows/model_spec.py`](../workflows/model_spec.py) — model and device spec definitions
- [`workflows/validate_setup.py`](../workflows/validate_setup.py) — GPU server restriction enforcement
