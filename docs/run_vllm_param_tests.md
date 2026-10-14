# Running vLLM parameter tests

How to run the vLLM parameter-conformance tests for development and debugging.

These are the LLM/VLM API parameter tests that run as part of `--workflow spec_tests`
(routed to the workflow engine). The suites live in `llm_module`:
`llm_module/test_vllm_chat_completions.py`,
`test_vllm_responses.py` and `test_tool_call_json_schema.py`. Models are mapped to suites in
`test_module/test_suites/llm.json`.

### step 1: first create the venv by running the workflow
This will fail out if no server is running.
```bash
python3 run.py --model Qwen/Qwen3-32B --device galaxy --workflow spec_tests
```

### step 2: run server
You can run the online vLLM server locally via: https://github.com/tenstorrent/vllm/blob/dev/examples/server_example_tt.py

Please make sure to set the runtime arguments the same as in tt-inference-server, if there are changes to runtime args those must be reflected in code.

You can run directly using tt-inference-server docker as an alternative to running locally and managing your own tt-metal and vLLM builds, for example:
```bash
python3 run.py --model Qwen/Qwen3-32B --device galaxy --workflow server --docker-server --dev-mode
```

### step 3: run the suite directly against a running server
The `spec_tests` workflow runs the suite in a child pytest process; you can
reproduce that manually. The suite imports fixtures from `test_fixtures.conftest`
and `report_module`, so put the repo root on
`PYTHONPATH`:

```bash
cd $TT_INFERENCE_SERVER_REPO_ROOT
source .workflow_venvs/.venv_tests_run_script/bin/activate

# add authorization env var if server was started with authorization
# note: if you used VLLM_API_KEY env var you can set that.
export JWT_SECRET=<my-secret>

# the example below runs the determinism tests (top_k / top_p / temperature)
PYTHONPATH="$PWD" \
pytest llm_module/test_vllm_chat_completions.py -sv \
  -k "test_determinism" \
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name Qwen/Qwen3-32B \
  --task-name vllm_chat_completions \
  --output-path ./workflow_logs/reports_output/spec_tests/test_my_output_path
```

The supported pytest options (`--endpoint-url`, `--model-name`, `--task-name`,
`--output-path`) are declared in `llm_module/conftest.py`.

You will see outputs where you specify `--output-path`, e.g.
`$TT_INFERENCE_SERVER_REPO_ROOT/workflow_logs/reports_output/spec_tests/test_my_output_path/parameter_report_vllm_chat_completions.json`

### Tool-call schema suite
`test_tool_call_json_schema.py` (run by `ToolCallSchemaConformanceTest`) takes
its settings as `--schema-*` options, also declared in `llm_module/conftest.py`.
It needs `jsonschema` in the venv.

```bash
PYTHONPATH="$PWD" \
pytest llm_module/test_tool_call_json_schema.py -q --tb=line \
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name MiniMaxAI/MiniMax-M3 \
  --task-name tool_call_json_schema \
  --output-path ./workflow_logs/reports_output/spec_tests/tool_call_schema \
  --schema-tool-choice auto --schema-exclude-ref
```

Per-case results go to `parameter_report_tool_call_json_schema.json` in the
output path; `--schema-max-cases N` runs a quick subset.
