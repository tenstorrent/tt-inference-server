# Running Models with `run.py --local-server`

This guide explains how to run a vLLM-backed model directly on the host with:

```bash
python3 run.py --model <model-name> --tt-device <device> --workflow server --local-server
```

`--local-server` is intended for development with local `tt-metal` and vLLM source trees. It avoids Docker, starts the same `vllm-tt-metal/src/run_vllm_api_server.py` wrapper used by the container image, and keeps all logs, model weights, and TT caches on the host filesystem.

## Repository Setup

The examples below assume this layout, but the paths can be anywhere:

```text
~/tt-inference-server
~/tt-metal
~/vllm
```

Use absolute paths or normal shell home expansion such as `~/tt-metal`. Do not use `~./tt-metal`.

### 1. Prepare `tt-metal`

Clone and build `tt-metal` from source, then create its Python environment:

```bash
git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules ~/tt-metal
cd ~/tt-metal
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
```

For `--local-server`, the `tt-metal` checkout must contain:

```text
~/tt-metal/python_env/bin/python
~/tt-metal/build/lib/
```

You can sanity-check the environment with:

```bash
~/tt-metal/python_env/bin/python -c "import ttnn; print('ttnn import OK')"
```

### 2. Prepare vLLM

Clone the vLLM source tree you want to develop against and install it into the `tt-metal` Python environment:

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm
~/tt-metal/python_env/bin/pip install -e ~/vllm
```

If you are working from Tenstorrent's vLLM fork or a model-specific commit, check out the matching commit before installing. The expected vLLM commit for a released model is recorded in the model spec (`workflows/model_spec.py` and generated `model_spec.json`) as `vllm_commit`.

If you are using upstream vLLM plus the out-of-tree TT plugin, install the plugin into the same `tt-metal` environment:

```bash
~/tt-metal/python_env/bin/pip install --no-deps -e ~/tt-inference-server/tt-vllm-plugin
```

The plugin currently uses vLLM V1, so export:

```bash
export VLLM_USE_V1=1
```

Verify that vLLM sees the TT platform plugin:

```bash
~/tt-metal/python_env/bin/python -c "from importlib.metadata import entry_points; print(list(entry_points(group='vllm.platform_plugins')))"
```

You should see an entry named `tt`. If `VLLM_PLUGINS` is set to an empty string, vLLM disables plugins; unset it or include `tt`.

### 3. Prepare `tt-inference-server`

Clone `tt-inference-server` and run commands from its repository root:

```bash
git clone https://github.com/tenstorrent/tt-inference-server.git ~/tt-inference-server
cd ~/tt-inference-server
```

`run.py` bootstraps its own workflow virtual environments with `uv`; you normally do not need to install Python packages into this repo manually. For local server mode, `run.py` also installs `vllm-tt-metal/requirements.txt` into the `tt-metal` Python environment before launching the server.

Set secrets as environment variables or in `.env`:

```bash
export HF_TOKEN=hf_...
export JWT_SECRET=my-secret-string
```

`HF_TOKEN` is needed for gated Hugging Face models. `JWT_SECRET` enables bearer-token auth. For local development without auth, pass `--no-auth`.

## Running a Local Server

Minimal command with separate local `tt-metal` and vLLM checkouts:

```bash
cd ~/tt-inference-server
export VLLM_USE_V1=1

python3 run.py \
  --model Mistral-Small-3.1-24B-Instruct-2503 \
  --tt-device t3k \
  --workflow server \
  --local-server \
  --tt-metal-home ~/tt-metal \
  --vllm-dir ~/vllm
```

If your vLLM checkout lives at `~/tt-metal/vllm`, you can omit `--vllm-dir`; `run.py` defaults it to `<tt-metal-home>/vllm`.

Useful development flags:

```bash
# Disable auth for local testing.
--no-auth

# Use a different OpenAI API port.
--service-port 8888

# Skip background trace capture if traces are already generated or not needed.
--disable-trace-capture

# Skip host system software validation while iterating.
--skip-system-sw-validation

# Point at an existing HF cache or pre-downloaded weights.
--host-hf-cache ~/.cache/huggingface
--host-weights-dir /path/to/model/weights
```

Only one of `--host-volume`, `--host-hf-cache`, and `--host-weights-dir` can be used at a time.

When `--workflow server` succeeds, `run.py` exits after starting the server process. The server keeps running in the background until you kill it.

## What Happens Internally

Local server mode has two main processes:

```text
python3 run.py ...
  -> ~/tt-metal/python_env/bin/python vllm-tt-metal/src/run_vllm_api_server.py ...
       -> runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

### `run.py`

`run.py` is the orchestration entrypoint:

1. Parses CLI arguments and resolves `--model`, `--tt-device`, `--impl`, and `--engine`.
2. Exports the model catalog to `model_spec.json`.
3. Bootstraps workflow tooling with `workflows/bootstrap_uv.py`.
4. Resolves the model/device pair into a concrete `ModelSpec` using `workflows/model_spec.py`.
5. Applies CLI overrides such as `--override-tt-config` and `--vllm-override-args`.
6. Writes a runtime model spec JSON under `workflow_logs/runtime_model_specs/`.
7. Runs validation from `workflows/validate_setup.py`:
   - Confirms `--local-server` is allowed only for vLLM-backed model specs.
   - Confirms `--tt-metal-home` or `TT_METAL_HOME` is set.
   - Optionally runs system software validation unless `--skip-system-sw-validation` is set.
   - Confirms `import vllm` works with `<tt-metal-python-venv-dir>/bin/python`.
   - Confirms required paths exist: `python_env/bin/python`, `build/lib`, the vLLM source directory, and `vllm-tt-metal/src/run_vllm_api_server.py`.
8. Calls `setup_host()` to create or resolve host storage directories.
9. Calls `workflows/run_local_server.py::run_local_server()`.

For `--workflow server`, `run.py` does not run benchmarks or evals after the server starts.

### `workflows/run_local_server.py`

`run_local_server()` prepares and launches the host subprocess:

1. Creates a log file in `workflow_logs/local_server/`.
2. Installs `vllm-tt-metal/requirements.txt` into the `tt-metal` Python environment with `uv pip install --python <tt-metal-python>`.
3. Builds the local-server environment:
   - `APP_DIR=<tt-inference-server>`
   - `TT_METAL_HOME=<tt-metal>`
   - `PYTHON_ENV_DIR=<tt-metal>/python_env` unless overridden by `--tt-metal-python-venv-dir`
   - `vllm_dir=<vllm-dir>`
   - `PYTHONPATH=<tt-inference-server>:<tt-metal>:...:<vllm-dir>`
   - `LD_LIBRARY_PATH=<tt-metal>/build/lib:...`
   - `CACHE_ROOT`, `TT_CACHE_PATH`, and `TT_METAL_LOGS_PATH`
   - `RUNTIME_MODEL_SPEC_JSON_PATH=<workflow_logs/runtime_model_specs/...json>`
   - `MODEL_WEIGHTS_DIR`, `HF_HOME`, and `HOST_HF_HOME` when a host cache or weights directory is used
4. Builds the subprocess command:

```bash
<tt-metal>/python_env/bin/python \
  <tt-inference-server>/vllm-tt-metal/src/run_vllm_api_server.py \
  --model <hf-model-repo> \
  --tt-device <device> \
  [--no-auth] \
  [--disable-trace-capture] \
  [--service-port <port>]
```

5. Starts the subprocess with `subprocess.Popen()`, redirects stdout/stderr to the local-server log, and waits briefly to ensure it did not fail immediately.
6. For `--workflow server`, leaves the subprocess running and logs the PID and kill command.

### `vllm-tt-metal/src/run_vllm_api_server.py`

This wrapper adapts a resolved tt-inference-server model spec into a vLLM OpenAI API server:

1. Parses wrapper arguments such as `--model`, `--tt-device`, `--no-auth`, `--disable-trace-capture`, and `--service-port`.
2. Loads the runtime model spec from `RUNTIME_MODEL_SPEC_JSON_PATH`.
3. Ensures weights are available unless `MODEL_WEIGHTS_DIR` already points at existing weights.
4. Registers TT model classes with vLLM's `ModelRegistry`.
5. Applies model-specific runtime environment variables from the model spec, such as `VLLM_TARGET_DEVICE=tt`, `MESH_DEVICE`, `ARCH_NAME`, and model-specific TT settings.
6. Sets TT Metal timeout and triage environment variables unless disabled.
7. Sets up auth:
   - `--no-auth` disables vLLM API key auth.
   - Otherwise `JWT_SECRET` is encoded into `VLLM_API_KEY`.
8. Creates model weight symlinks and sets `HF_MODEL`.
9. Merges model-spec `vllm_args` with CLI passthrough args and rewrites `sys.argv` as a normalized `vllm serve` command.
10. Optionally starts background trace capture requests.
11. Starts vLLM in-process with:

```python
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

At this point vLLM owns the process and serves the OpenAI-compatible API on the configured port.

## Logs and Runtime State

Important files are under `workflow_logs/`:

```text
workflow_logs/run_logs/run_<timestamp>_<model-id>_<workflow>_<uuid>.log
workflow_logs/runtime_model_specs/runtime_model_spec_<timestamp>_<model-id>_<uuid>.json
workflow_logs/local_server/vllm_local_<timestamp>_<model>_<device>_<workflow>.log
```

The run log shows the CLI summary, validation, generated runtime model spec path, local-server environment exports, subprocess command, PID, and stop command.

The local-server log is the server's stdout/stderr. If startup fails, look there first.

To stop a server started by `--workflow server`, use the PID logged by `run.py`:

```bash
kill <pid>
```

You can also check for a listening port:

```bash
curl http://localhost:8000/v1/models
```

If auth is enabled, include the bearer token generated from `JWT_SECRET` by the client tooling, or use `--no-auth` for local testing.

## Running Client Workflows Against the Server

After a local server is running, you can run client-side workflows without `--local-server`:

```bash
python3 run.py --model Mistral-Small-3.1-24B-Instruct-2503 --tt-device t3k --workflow benchmarks --limit-samples-mode smoke-test
python3 run.py --model Mistral-Small-3.1-24B-Instruct-2503 --tt-device t3k --workflow evals --limit-samples-mode smoke-test
```

If the server uses a non-default port, export `SERVICE_PORT` or pass `--service-port` consistently.

## Troubleshooting

**`--local-server requires --tt-metal-home or TT_METAL_HOME to be set`**

Pass `--tt-metal-home ~/tt-metal` or export `TT_METAL_HOME=/path/to/tt-metal`.

**`Missing required python venv interpreter`, `tt-metal build/lib`, or `vLLM source dir`**

Build `tt-metal`, create `python_env`, and pass the correct `--vllm-dir`. The local server validates all required paths before launch.

**`requires the vllm Python package to be installed in the tt-metal python environment`**

Install vLLM into the same interpreter that local server will use:

```bash
~/tt-metal/python_env/bin/pip install -e ~/vllm
```

**vLLM says `Failed to infer device type`**

The TT platform plugin is probably not installed or not being loaded. Install `tt-vllm-plugin` into `tt-metal/python_env`, set `VLLM_USE_V1=1`, and verify the `tt` platform plugin appears in `vllm.platform_plugins`.

**`ImportError: cannot import name 'current_platform' from 'vllm.platforms'`**

This usually indicates a circular import or incompatible vLLM/plugin combination. Make sure the plugin version matches the vLLM source tree you installed.

**`ValueError: numpy.dtype size changed`**

The `tt-metal` Python environment has incompatible binary wheels, often pandas/scikit-learn versus NumPy. Reinstall compatible versions in `tt-metal/python_env` or rebuild the environment.

**`unrecognized arguments: --override-tt-config`**

That flag exists in Tenstorrent's vLLM fork, not upstream vLLM. Use a compatible vLLM checkout or make sure the local server wrapper filters Tenstorrent-only model-spec keys before invoking upstream vLLM.
