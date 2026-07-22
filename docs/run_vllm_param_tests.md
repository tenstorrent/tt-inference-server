# Running vLLM parameter tests

How to run the vLLM parameter-conformance tests for development and debugging.

These are the LLM/VLM API parameter tests that run as part of `--workflow spec_tests`
(routed to the v2 engine). The suites live in the v2 package:
`tt-inference-server-v2/llm_module/test_vllm_chat_completions.py` and
`test_vllm_responses.py`. Models are mapped to suites in
`tt-inference-server-v2/test_module/test_suites/llm.json`.

### step 1: first create the venv by running the workflow
This will fail out if no server is running.
```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow spec_tests
```

### step 2: run server
You can run the online vLLM server locally via: https://github.com/tenstorrent/vllm/blob/dev/examples/server_example_tt.py

Please make sure to set the runtime arguments the same as in tt-inference-server, if there are changes to runtime args those must be reflected in code.

You can run directly using tt-inference-server docker as an alternative to running locally and managing your own tt-metal and vLLM builds, for example:
```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow server --docker-server --dev-mode
```

### step 3: run the suite directly against a running server
The `spec_tests` workflow runs the suite in a child pytest process; you can
reproduce that manually. The suite imports fixtures from `server_tests.conftest`
and `report_module`, so put both the repo root and the v2 package root on
`PYTHONPATH`:

```bash
cd $TT_INFERENCE_SERVER_REPO_ROOT
source .workflow_venvs/.venv_tests_run_script/bin/activate

# add authorization env var if server was started with authorization
# note: if you used VLLM_API_KEY env var you can set that.
export JWT_SECRET=<my-secret>

# the example below runs the determinism tests (top_k / top_p / temperature)
PYTHONPATH="$PWD:$PWD/tt-inference-server-v2" \
pytest tt-inference-server-v2/llm_module/test_vllm_chat_completions.py -sv \
  -k "test_determinism" \
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name Qwen/Qwen3-32B \
  --task-name vllm_chat_completions \
  --output-path ./workflow_logs/reports_output/spec_tests/test_my_output_path
```

The supported pytest options (`--endpoint-url`, `--model-name`, `--task-name`,
`--output-path`) are declared in `tt-inference-server-v2/llm_module/conftest.py`.

You will see outputs where you specify `--output-path`, e.g.
`$TT_INFERENCE_SERVER_REPO_ROOT/workflow_logs/reports_output/spec_tests/test_my_output_path/parameter_report_vllm_chat_completions.json`
