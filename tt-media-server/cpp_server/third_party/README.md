# Third-party dependencies

Optional dependencies are **fetched at configure time** via CMake `FetchContent` (no git submodules).
Fetched content lives under `build/_deps/<name>-src/` and is not committed.

## Tokenizer (ENABLE_TOKENIZER=ON)

- **Source:** [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) at a pinned commit (AHashMap fix).
- **Requires:** Rust toolchain. No `git submodule update` needed; CMake fetches when you configure with `-DENABLE_TOKENIZER=ON`.

## Adding a new third-party

1. In `third_party/CMakeLists.txt`: add an `if(OPTION_NAME)` block with `FetchContent_Declare(...)` and `FetchContent_MakeAvailable(...)`.
2. In the main `cpp_server/CMakeLists.txt`: declare the option and link the target when the option is ON.

Use FetchContent (or find_package) so the main repo stays small and versions are pinned in CMake.

## If you previously had the tokenizers-cpp submodule

To remove it and rely on FetchContent only (no sublink):

```bash
# From repo root
git submodule deinit -f tt-media-server/cpp_server/third_party/tokenizers-cpp
git rm -f tt-media-server/cpp_server/third_party/tokenizers-cpp
# Edit .gitmodules and remove the [submodule "tt-media-server/cpp_server/third_party/tokenizers-cpp"] section
git add .gitmodules
```

Then reconfigure with `-DENABLE_TOKENIZER=ON`; CMake will fetch tokenizers-cpp into `build/_deps/`.
