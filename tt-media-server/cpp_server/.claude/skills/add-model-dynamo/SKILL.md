---
name: add-model-dynamo
description: Checklist for onboarding a new LLM to the cpp_server inference backend so it serves through Dynamo — registering the model type, fetching tokenizer files, tokenizer static data (eos/stop/think tokens), and Dynamo discovery (reasoning + tool-call parsers, generation_config publishing). Use whenever a new model is being onboarded to cpp_server, a HuggingFace model is being wired into the Dynamo deploy, or a model "loads but generates wrong / isn't discoverable / isn't selectable".
---

# Onboarding a model to cpp_server (Dynamo)

## Touchpoints at a glance

| # | Where | What to add |
|---|-------|-------------|
| 0 | **ask the user** | the model's **full HuggingFace id** (e.g. `openai/gpt-oss-120b`, `MiniMaxAI/MiniMax-M2.7`) — required, do not guess |
| 1 | `include/config/types.hpp` | `ModelType` + `Model` enum values, `MODEL_MAPPINGS` (`Model`→HF id), `modelTypeFromDeviceBackend` short-name branch |
| 2 | `src/config/settings.cpp` | `modelType()` resolver: HF id → `ModelType` (else silently falls back to DeepSeek) |
| 3 | `scripts/fetch_tokenizers.sh` | download **all** needed files into `tokenizers/<hf-id>/` |
| 4 | `src/utils/tokenizers/tokenizer.cpp` | `tokenizerDirForModel`, `createTokenizer` (defaults to `DeepseekTokenizer`), `staticInfoFor` + a `StaticTokenizerInfo` (eos/stop/think token ids) |
| 5 | `src/dynamo/discovery.cpp` | `runtimeParsersForModelType` (reasoning + tool-call parser ids), `buildMdcJson` (publishes `generation_config.json`) |
| 6 | deploy + verify | `deploy.sh --hf-model-id <hf-id>`, confirm loads / registers / answers |

Reference: [tenstorrent/tt-inference-server#4143](https://github.com/tenstorrent/tt-inference-server/pull/4143) (and the GPT-OSS/MiniMax onboarding commits).

## How the pieces connect

The worker resolves behavior from `MODEL` (HF id) → `ModelType`
(`settings.cpp::modelType`), which selects the tokenizer dir, the tokenizer impl
(default `DeepseekTokenizer`), and the **static token info** (eos/stop/think ids).
The Dynamo frontend, separately, reads the **MDC** the worker publishes in
`src/dynamo/discovery.cpp` to learn the tokenizer files, the `generation_config.json`, and
which **reasoning/tool-call parsers** to apply. Both halves must agree.

The recurring failure mode: the model loads but a table wasn't updated — falls
back to DeepSeek (missing `modelType()` entry), wrong/no reasoning + tool output
(missing `runtimeParsersForModelType` branch), or the frontend hard-fails to load
the model because `eos_token_id` is absent (model carries it only in
`generation_config.json`, which `buildMdcJson` must publish).

## Checklist

1. **Get the full HF id from the user.** Everything keys off it; don't assume.

2. **Register the model** in `include/config/types.hpp`: add the value to both
   `enum class ModelType` and `enum class Model`, add `{Model::X, "<hf-id>"}` to
   `MODEL_MAPPINGS`, and add the `LLM_DEVICE_BACKEND` short-name branch to
   `modelTypeFromDeviceBackend` (e.g. `"gpt-oss" -> GPT_OSS_120B`). Then map the HF
   id in `src/config/settings.cpp` `modelType()`
   (`if (m == "<hf-id>") return ModelType::X;`) — without it the worker silently
   serves as DeepSeek.
   In src/config/settings.cpp, in resolveBlazeNumberOfPipelineStages, add the case for the model
   Ask the user for the number of stages, if he did not already specify this

3. **Fetch tokenizer files** in `scripts/fetch_tokenizers.sh` so
   `tokenizers/<hf-id>/` has **tokenizer.json, tokenizer_config.json, config.json,
   generation_config.json, chat_template.jinja** if jinja files exist. Steps 4–5 read eos ids out of
   `config.json` / `generation_config.json` and discovery publishes
   `generation_config.json`, so all must be present.

4. **Tokenizer** in `src/utils/tokenizers/tokenizer.cpp`:
   - `tokenizerDirForModel` → the HF dir (default falls back to DeepSeek).
   - `createTokenizer` → reuse `DeepseekTokenizer` for chat-template/tool-call
     behavior unless the model needs a dedicated impl (the **default-to-deepseek** rule).
   - Add a `StaticTokenizerInfo` (e.g. `gptOss120bInfo()`) and wire it into
     `staticInfoFor`. Set, **verifying ids against the fetched tokenizer**:
     - `eosTokenId` = `eos_token_id` from **config.json** (the single primary id),
     - `stopTokenIds` = the remaining ids from **generation_config.json** `eos_token_id`
       (often a list — e.g. gpt-oss `[200002, 199999, 200012]` → eos `200002`, stops `{199999, 200012}`),
     - for reasoning models, `thinkStartTokenId`/`thinkEndTokenId` (`<think>`/`</think>`);
       Harmony-style models (gpt-oss) have no think tokens.

5. **Dynamo discovery** in `src/dynamo/discovery.cpp`:
   - `runtimeParsersForModelType` → return `{reasoning_parser, tool_call_parser}`
     id strings for the model — these tell the frontend which parsers to apply
     (e.g. `gpt_oss → {"gpt_oss","harmony"}`, `minimax_m2 → {"minimax_append_think","minimax_m2"}`;
     DeepSeek default `{"deepseek_r1", nullptr}`).
   - `buildMdcJson` already publishes `generation_config.json` when present (so
     models like MiniMax that omit `eos_token_id` from `config.json` still load).
     Nothing to change unless the model needs extra MDC fields — just confirm the
     file is fetched (step 3).

6. **Deploy & verify** (see `run-dynamo-server`). `deploy.sh` serves any model via
   `--hf-model-id <hf-id>` — no script change needed. Then:

```bash
curl -s "http://dynamo-frontend:8000/v1/models"
curl -s "http://dynamo-frontend:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"<hf-id>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

Confirm: `docker logs tt-cpp-worker` shows the new tokenizer loaded and the worker
registered with etcd; a reasoning model reports `reasoning_tokens` in the final-chunk
usage; tool-call output parses (the `runtimeParsersForModelType` ids are correct);
and the frontend didn't reject the model for a missing `eos_token_id`.
