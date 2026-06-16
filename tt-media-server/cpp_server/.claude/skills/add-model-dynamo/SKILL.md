---
name: add-model-dynamo
description: Checklist for adding a new LLM to the cpp_server inference backend so it serves through Dynamo — tokenizer fetch, ModelType config, reasoning/chat-template handling, and the deploy flag. Use whenever a new model is being onboarded to cpp_server, a HuggingFace model is being wired into the Dynamo deploy, or a model "loads but generates wrong / isn't selectable via --<model>".
---

# Adding a model to cpp_server (Dynamo)

## Touchpoints at a glance

| # | Where | What to add |
|---|-------|-------------|
| 1 | `scripts/fetch_tokenizers.sh` | HF id + download → `tokenizers/<org>/<model>/` |
| 2 | `include/config/settings.hpp`, `src/config/settings.cpp` | `ModelType` enum value, HF-id→enum resolver (`~:465`), descriptor prefix (`~:48`), sampling/reasoning config (`~:483`) |
| 3 | `include/utils/tokenizers/tokenizer.hpp` | `thinkTokenIdsFor()` → reasoning span tokens (or "no token") |
| 4 | `dynamo_frontend/deploy.sh` (`~:265`) | worker env: `BLAZE_SOCKET_DESCRIPTOR_PREFIX`, `USE_DEEPSEEK_MD_FORMAT` |
| 5 | `dynamo_frontend/deploy.sh` (`:39`, `:133`, `:102`) | `<MODEL>_MODEL_ID`, `--<model>` flag, usage text |

## How the pieces connect

The cpp_server worker resolves its behavior from `MODEL` (HF id) → `ModelType`
(`config/settings.cpp`), which drives the tokenizer, the reasoning-span token ids,
and sampling. The Dynamo frontend routes by the same model name (registered in
etcd) and the deploy injects per-model worker env. The model catalog (tokenizer
tree) is **baked into both** the worker and frontend images and fetched by the
shared `scripts/fetch_tokenizers.sh`.

The recurring failure mode: the model loads but a registry table wasn't updated —
wrong/empty reasoning tokens (no `reasoning_tokens` in usage), wrong chat-template
(`USE_DEEPSEEK_MD_FORMAT`), or `--<model>` not selectable because the deploy flag
is missing.

## Read first

`tt-media-server/cpp_server/scripts/fetch_tokenizers.sh` (the model list) and an
existing model as the template: DeepSeek-R1 or Kimi-K2.6 for reasoning models,
Llama-3.1-8B-Instruct for a plain instruct model. `build.sh:175-181` shows how the
tokenizer fetch is wired into the build.

## Checklist

1. **Tokenizer files** → add the HF id + download logic to
   `scripts/fetch_tokenizers.sh` so `tokenizers/<org>/<model>/` gets
   `tokenizer.json`, `tokenizer_config.json`, `config.json`,
   `generation_config.json`. The frontend image bakes the same tree.
2. **`ModelType`** → in `include/config/settings.hpp` add the enum value; in
   `src/config/settings.cpp` map the HF id in the resolver (`~:465`,
   `if (m == "<org>/<Model>") return ModelType::<NEW>;`), add the descriptor-prefix
   string in `toString` (`~:48`), and add the sampling/reasoning branch (`~:483`).
   Without the resolver entry the worker silently falls back to `DEEPSEEK_R1_0528`.
3. **Reasoning span** → `thinkTokenIdsFor(ModelType)` in
   `include/utils/tokenizers/tokenizer.hpp`: return the `<think>`/`</think>` (or
   equivalent) token id pair for reasoning models, the "no token" sentinel
   otherwise. Drives `reasoning_tokens` accounting on the Dynamo path.
4. **Chat template / MD format** → add a branch to the `case "$HF_MODEL_ID"` block
   in `dynamo_frontend/deploy.sh:~265` with the right
   `BLAZE_SOCKET_DESCRIPTOR_PREFIX` and (if applicable) `USE_DEEPSEEK_MD_FORMAT=1`.
5. **Deploy flag** → in `dynamo_frontend/deploy.sh` add `<MODEL>_MODEL_ID="<org>/<Model>"`
   (`:39`), a `--<model>` arg setting `HF_MODEL_ID` (`:133`), and a usage line (`:102`).

## Verify end-to-end

Always exercise it through the deploy, not a bare worker:

```bash
./build.sh                       # re-fetches tokenizers, recompiles
cd ../../dynamo_frontend
LLM_DEVICE_BACKEND=mock ./deploy.sh --<model> --local-build --no-monitoring
FE=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' dynamo-frontend)
curl -s "http://${FE}:8000/v1/models"
curl -s "http://${FE}:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"<org>/<Model>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

- `docker logs tt-cpp-worker` must show the new tokenizer loaded (`[TokenizerUtil] Loaded tokenizer from: .../<model>/tokenizer.json`) and the worker registered with etcd.
- A reasoning model should report `reasoning_tokens` in the final-chunk usage; if it's 0/absent, step 3 is wrong.
- If `--<model>` errors as unknown, step 5 is missing.

## Related skills

`run-dynamo-server` (launch the stack) · `benchmark-dynamo` (load-test it).
