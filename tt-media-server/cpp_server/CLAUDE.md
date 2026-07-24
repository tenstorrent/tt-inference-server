# CLAUDE.md

This file provides guidance when working in `cpp_server/` (TT Media Server C++ / Drogon).

## Project overview

High-performance C++ server with an OpenAI-compatible API, for benchmarking against the Python FastAPI stack. Features include LLM chat completions and embeddings, paged attention, prefix caching, and multiprocess workers with Boost.Interprocess IPC.

## Build and development

```bash
# Standard release build (tokenizer assets pre-fetched by build.sh)
./build.sh

./build.sh --debug
./build.sh --asan
./build.sh --tsan
```

Runtime model/backend selection uses `LLM_DEVICE_BACKEND` and related env vars (see `config/settings.hpp`).

## Testing

```bash
cd build && ctest --output-on-failure
```

Integration tests also live under `tests/` (Python and C++).

## C++ conventions (naming, format, lint)

Mirrors the Cursor rules in `.cursor/rules/` (`cpp-naming`, `cpp-format-lint`).

- **Naming** (per `.clang-tidy`, Google style): **never snake_case**. camelCase for
  functions/methods/variables/parameters/locals (incl. local `const`); PascalCase for
  classes/structs; `UPPER_CASE` for global/static/class constants and macros; private
  members are camelCase with a trailing `_` (e.g. `sessionManager_`). Don't rename
  externally visible strings (JSON fields, metric labels, env vars, wire values) for style.
- **Format:** `./format.sh` (clang-format; authoritative for layout).
- **Lint:** `./build.sh --clang-tidy` (lint == build; enforces the naming above).

## Dependencies and prerequisites

- **CMake** >= **3.24** (matches `CMakeLists.txt` `cmake_minimum_required`). On Ubuntu 22.04 and similar, distro `cmake` may be older; run `./install_dependencies.sh`, which installs a Kitware CMake under `/usr/local` when needed.
- **C++20** compiler (GCC 10+, Clang 12+)
- **Drogon** >= 1.8 (installed by `install_dependencies.sh` if missing)
- **Boost**, **JsonCpp**, **Rust** (for tokenizers-cpp), **Python 3** (interpreter + dev headers for pybind)

Optional: **Kafka** / `librdkafka-dev` via `./install_dependencies.sh --kafka` when building with `KAFKA_ENABLED=ON`.

## Docker / CI

CI runs `cpp_server/install_dependencies.sh` before C++ jobs; it must keep CMake at or above the `cmake_minimum_required` version in `CMakeLists.txt`.
