# Qwen-3-1.5B-Instruct CPU Runner

CPU-only model runner for the `cpp_server` LLM engine, demonstrating how to
serve models that are **not implemented in tt-metal**.

## Architecture

```
cpp_server (C++)
  └── LLMRunner (scheduler + block manager)
        └── QwenModelRunner  (pybind11 wrapper)
              └── Qwen15BRunner  (Python, HuggingFace transformers)
                    └── AutoModelForCausalLM on CPU
```

## Files Added

| File | Purpose |
|------|---------|
| `tt_model_runners/qwen_runner.py` | Python runner: loads HF model, manages `DynamicCache`, samples tokens |
| `include/runners/qwen_model_runner.hpp` | C++ header: `QwenModelRunner` implements `IModelRunner` |
| `src/runners/qwen_model_runner.cpp` | C++ implementation: pybind11 glue, GIL handling, error recovery |

## Files Modified

| File | Change |
|------|--------|
| `include/config/types.hpp` | Added `QWEN` to `ModelRunnerType` enum |
| `src/runners/llm_runner/model_runner.cpp` | Added `makeQwenModelRunner()` factory branch |
| `src/services/model_service_registration.cpp` | Registered `QWEN` runner type in `RunnerRegistry` |
| `CMakeLists.txt` | Added `src/runners/qwen_model_runner.cpp` to `LLM_RUNNER_LIB_SOURCES` (unconditional, CPU-only) |

## How to Use

### 1. Set environment variables

```bash
export MODEL_SERVICE=llm
export LLM_DEVICE_BACKEND=qwen          # or any non-"llama" string
export HF_MODEL=Qwen/Qwen3-1.5B-Instruct
export TT_PYTHON_PATH=/path/to/tt-inference-server
```

### 2. Build

```bash
cd tt-media-server/cpp_server
./build.sh
```

No tt-metal installation required. The Qwen source is compiled unconditionally
because it has no tt-metal dependencies.

### 3. Run

```bash
./build/tt_media_server
```

The runner factory will select `QwenModelRunner` when `runner_type == QWEN`.

## Key Design Decisions

- **CPU-only**: Uses `transformers.AutoModelForCausalLM.from_pretrained(..., device_map="cpu")`. No tt-metal imports.
- **KV cache**: `DynamicCache` (past_key_values) managed per-sequence in Python. The C++ block_table is passed through but ignored.
- **Sampling**: Done in Python (`torch.softmax` + `torch.multinomial`). Supports temperature, top_p, top_k, repetition/presence/frequency penalties, and grammar masks (`allowed_token_ids`).
- **Parity with LlamaModelRunner**: The C++ wrapper is a near-clone of `LlamaModelRunner`, using the same pybind11 patterns, GIL acquisition, and error handling. This minimizes cognitive load for maintainers.

## Limitations / TODO

- **No batching**: Each sequence is processed independently. For a 1.5B model on CPU this is acceptable for a POC, but batched inference would improve throughput.
- **No paged KV cache**: `DynamicCache` grows unbounded. Long conversations will eventually OOM. For a POC this is fine; production would need a CPU paged-cache implementation or offloading.
- **No continuous batching**: The scheduler still calls `run()` per-step; the runner itself does not exploit continuous batching.
- **Memory**: `torch.float32` by default. Could use `float16` or `bfloat16` on CPU for lower memory, but CPU float16 matmuls are often slower.

## Testing

### Manual smoke test

```bash
# Start server
TT_VISIBLE_DEVICES=0 ./build/tt_media_server &

# Send a chat completion
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'
```

### Unit test (optional)

A minimal unit test could instantiate `QwenModelRunner` directly with a mock
`DecodeCallback` and verify that `run()` returns a token. This requires the
Python environment to have `transformers` and `torch` installed.
