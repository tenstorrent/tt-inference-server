# AGENTS.md

Instructions for agentic coding tools (Claude Code, Cursor, Codex, etc.) working in `cpp_server`.

## Requirements and Issues

Requirements live in GitHub Issues at
[tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server/issues).

Issues relevant to this component are labeled with **both** `cpp-server` and
`Inference technologies`. Use `gh` to read them:

```bash
# List open issues for cpp_server
gh issue list --repo tenstorrent/tt-inference-server \
  --label cpp-server --label "Inference technologies"

# View a specific issue
gh issue view <number> --repo tenstorrent/tt-inference-server
```

Issues here are typically terse. If an issue doesn't give you enough context
to implement it (no acceptance criteria, ambiguous scope, unclear definition
of done), **ask the user before implementing** — don't infer scope from the
title alone.

## Build

```bash
./build.sh --blaze
```

## Run

```bash
./build/tt_media_server_cpp -p <PORT>
```

Logs default to `info`. For debug logs:

```bash
TT_LOG_LEVEL=debug ./build/tt_media_server_cpp -p <PORT>
```

### Log line identity

Every log line is prefixed with the process role so that mixed
decode/prefill/worker logs in a shared aggregator can be filtered by role
without relying on separate tee files:

```
[2026-06-03 20:55:07.660] [decode] [info] ...
[2026-06-03 20:55:07.881] [decode-worker0] [info] ...
```

The role is `LLM_MODE` (`decode` / `prefill` / `regular`) for the LLM service,
the service name (`image` / `embedding`) otherwise. Forked worker subprocesses
append `-worker<index>` (e.g. `decode-worker0`) so they stay distinguishable
from the HTTP node. Controlled by `LLM_MODE`.

## Test

**Unit + integration tests:**

```bash
cd build && ctest --output-on-failure
```

**Canonical integration test:**
[`tests/main_integration_test.cpp`](tests/main_integration_test.cpp) — gray-box round-trip
of the controller against a mocked worker. Start with
`HappyPath_RequestToMemoryToTaskToResponse` to understand the suite.

**Workload smoke test:** run a vLLM benchmark client against the running server.
Because the backend is a mock, throughput and latency numbers are not
meaningful — assert only that **no requests fail**.

```bash
uv venv
source .venv/bin/activate
uv pip install vllm

vllm bench serve \
  --model 'deepseek-ai/DeepSeek-R1-0528' \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --max-concurrency 64
```

## Recommended MCP servers

This project ships an `.mcp.json` recommending the
[Excalidraw](https://mcp.excalidraw.com) MCP server, useful for sketching
architecture diagrams during agent sessions. Tools that auto-load
`.mcp.json` (Claude Code, Cursor) will pick it up.
