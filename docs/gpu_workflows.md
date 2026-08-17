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
