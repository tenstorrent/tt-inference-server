# Running vLLM parameter tests

How to run vLLM parameters specific tests for development and debugging.

### step 1: first create venv by running workflow
This will fail out if no server is running.
```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow tests
```

### step 2: run server
You can run the online vLLM server locally via: https://github.com/tenstorrent/vllm/blob/dev/examples/server_example_tt.py

Please make sure to set the runtime arguments the same as in tt-inference-server, if there are changes to runtime args those must be reflected in code.

You can run directly using tt-inference-server docker as an alternative to running locally and managing your own tt-metal and vLLM builds, for example:
```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow server --docker-server --dev-mode
```

### step 3: using venv_tests_run_script venv for local dev
```bash
cd $TT_INFERENCE_SERVER_REPO_ROOT
source .workflow_venvs/.venv_tests_run_script/bin/activate

# add authorization env var if server was started with authorization
# note: if you used VLLM_API_KEY env var you can set that.
export JWT_SECRET=<my-secret>

# the example below runs the determinism tests, these test top_k or top_p, and temperature are working.
pytest tests/server_tests/test_cases/test_vllm_server_parameters.py -sv \
-k "test_determinism" \
--endpoint-url http://127.0.0.1:8000/v1/chat/completions \
--model-name Qwen/Qwen3-32B \
--model-backend tt-transformers \
--output-path ./workflow_logs/tests_output/test_my_output_path
```
The default pytest args are defined in `test_config.py` for each model, e.g.: https://github.com/tenstorrent/tt-inference-server/blob/dev/tests/test_config.py#L45


#### step 3[alternative]: run pytest binary without 'source'
```bash
cd $TT_INFERENCE_SERVER_REPO_ROOT

# add authorization env var if server was started with authorization
# note: if you used VLLM_API_KEY env var you can set that.
export JWT_SECRET=<my-secret>

# the example below runs the determinism tests, these test top_k or top_p, and temperature are working.
.workflow_venvs/.venv_tests_run_script/bin/pytest tests/server_tests/test_cases/test_vllm_server_parameters.py -sv \
-k "test_determinism" \
--endpoint-url http://127.0.0.1:8000/v1/chat/completions \
--model-name Qwen/Qwen3-32B \
--model-backend tt-transformers \
--output-path ./workflow_logs/tests_output/test_my_output_path
```

Example output:
```log
tstesco@xxxxxxxxxxx ~/software/tt-inference-server[tstesco/test-manual-run-doc]$ export JWT_SECRET=<my-secret>
tstesco@xxxxxxxxxxx ~/software/tt-inference-server[tstesco/test-manual-run-doc]$ .workflow_venvs/.venv_tests_run_script/bin/pytest tests/server_tests/test_cases/test_vllm_server_parameters.py -sv -k "test_determinism" --endpoint-url http://127.0.0.1:8000/v1/chat/completions --model-name Qwen/Qwen3-32B --model-backend tt-transformers --output-path ./workflow_logs/tests_output/test_my_output_path
================================================================================ test session starts =================================================================================
platform linux -- Python 3.10.19, pytest-8.3.5, pluggy-1.6.0 -- /home/tstesco/software/tt-inference-server/.workflow_venvs/.venv_tests_run_script/bin/python
cachedir: .pytest_cache
rootdir: /home/tstesco/software/tt-inference-server
configfile: pyproject.toml
plugins: anyio-4.12.0
collected 20 items / 17 deselected / 3 selected

tests/server_tests/test_cases/test_vllm_server_parameters.py::test_determinism_parameters[temperature-0.0] PASSED
tests/server_tests/test_cases/test_vllm_server_parameters.py::test_determinism_parameters[top_k-1] PASSED
tests/server_tests/test_cases/test_vllm_server_parameters.py::test_determinism_parameters[top_p-0.01] PASSED
Generating parameter_report.json...
parameter_report.json generated.


========================================================================= 3 passed, 17 deselected in 57.53s ==========================================================================
```

You will see outputs in where you specify `--output-path`, e.g. `$TT_INFERENCE_SERVER_REPO_ROOT/workflow_logs/tests_output/test_my_output_path/parameter_report.json`
