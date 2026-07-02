# Prefill/Decode Tests — How to Run

`test_prefill_decode.py` is a smoke/integration suite for the Dynamo +
cpp_server **disaggregated** stack. It proves prefill/decode routing and
prefix-cache behavior end-to-end through the frontend: input length (ISL) vs the
`MAX_TOKENS_TO_PREFILL_ON_DECODE` threshold decides whether a prompt is prefilled
locally on the decode server or offloaded to the prefill server, and each test
asserts the expected cache HIT/MISS, `cached_tokens`, and TTFT/TPS. The 8 tests
cover health, local prefill, prefix-cache growth, shared prefixes, offload
routing, streaming TTFT, large (~55k) offloaded prompts, and multi-turn chat.

`test_07` (large-prompt prefix-cache TTFT, cold vs warm) is the main test — it
sends a 50k-token system prompt + 5k-token user prompt (cold), then a second 5k
user prompt reusing the same system (warm), exercising the full offloaded prefill
+ prefix-cache path. 

There are two ways to run it: against a self-contained **mock** stack that gets
brought up for you, or against a **real deployed** stack you point it at.

## Quick run (mock, self-contained)

Brings up the disaggregated **mock_pipeline** stack for you (decode + prefill
workers + frontend + etcd), runs the suite, and tears it down. No server to
deploy yourself.

Via `run.py`:

```bash
python run.py --workflow prefill_decode --served-model moonshotai/Kimi-K2.6
```

Standalone (lets you pick a single test, e.g. `test_07`):

```bash
cd tt-inference-server-v2/test_module/llm_tests/smoke_tests
MODEL=moonshotai/Kimi-K2.6 ./run_tests.sh -v -k 07
```

Requirements: standalone runs need `pytest` (and `datasets` for `test_08`);
`run.py` bootstraps its own venv automatically.

```bash
pip install pytest datasets
```

## Against a real deployed stack

Here you deploy real decode + prefill servers yourself and point the tests at
that Dynamo endpoint — no mock stack is launched. Deploy instructions:

https://docs.google.com/document/d/10XN_m7UbIHv_ynaSrZisy4LlDFIPrpft97Lmi2Tdhss/edit?tab=t.0#heading=h.5iy89axg7v5e

Then run the test file directly (defaults assume Dynamo at `localhost:8080`):

```bash
TARGET=http://<your-dynamo>:8080 MODEL=<served-model> \
  pytest -v tt-inference-server-v2/test_module/llm_tests/smoke_tests/test_prefill_decode.py
```

Env overrides: `TARGET`, `MODEL`, `DECODE_URL`, `PREFILL_URL`, `THRESHOLD`,
`API_KEY`.
