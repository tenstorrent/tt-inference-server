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

### step 2: using venv_tests_run_script venv for local dev
```bash
cd $TT_INFERENCE_SERVER_REPO_ROOT
source .workflow_venvs/.venv_tests_run_script/bin/activate

# add authorization env var if server was started with authorization
export JWT_SECRET=<my-secret>

pytest tests/server_tests/test_cases/test_vllm_server_parameters.py -sv \
-k "test_determinism" \
--endpoint-url http://127.0.0.1:8000/v1/chat/completions \
--model-name Qwen/Qwen3-32B \
--model-backend tt-transformers \
--output-path ./workflow_logs/tests_output/test_my_output_path

```
The default pytest args are defined in `test_config.py` for each model, e.g.: https://github.com/tenstorrent/tt-inference-server/blob/dev/tests/test_config.py#L45


#### step 2: [alternative] run pytest binary without 'source'
```bash
cd $TT_INFERENCE_SERVER_REPO_ROOT

# add authorization env var if server was started with authorization
export JWT_SECRET=<my-secret>

.workflow_venvs/.venv_tests_run_script/bin/pytest tests/server_tests/test_cases/test_vllm_server_parameters.py -sv \
-k "test_determinism" \
--endpoint-url http://127.0.0.1:8000/v1/chat/completions \
--model-name Qwen/Qwen3-32B \
--model-backend tt-transformers \
--output-path ./workflow_logs/tests_output/test_my_output_path

# add authorization env var if server was started with authorization
export JWT_SECRET=<my-secret>
```

You will see outputs in where you specify `--output-path`, e.g. `$TT_INFERENCE_SERVER_REPO_ROOT/workflow_logs/tests_output/test_my_output_path/parameter_report.json`