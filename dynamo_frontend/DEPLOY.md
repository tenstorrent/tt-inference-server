# Deploying Dynamo Frontend + cpp_server Worker

`deploy.sh` brings up a three-container, etcd-backed stack on the `dynamo-net`
Docker network — **etcd** (discovery), **cpp_server** (worker), and
**dynamo-frontend** (HTTP gateway) — then tails the worker's logs. Ctrl+C tears
all three down.

## Quick start

```bash
cd dynamo_frontend
./deploy.sh --deepseek      # default; or: --kimi
```

All images have defaults, so no image flags are required. Override individually
with `--etcd-image` / `--worker-image` / `--frontend-image` if needed.

**Local development** (run your freshly-built `cpp_server` binary inside the
worker instead of the image's, e.g. on a box without a Tenstorrent card — a
build made without `TT_METAL_HOME` runs the mock backend):

```bash
cd ../tt-media-server/cpp_server && ./build.sh        # produces build/tt_media_server_cpp
cd ../../dynamo_frontend && ./deploy.sh --deepseek --local-build
```

`--local-build` bind-mounts `<repo>/tt-media-server/cpp_server/build` over the
worker image and runs that binary — no `--cpp-server-dir` and no image rebuild.

## Options


| Flag                     | Default                                   | Purpose                                                                         |
| ------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------- |
| `--kimi` / `--deepseek`  | `--deepseek`                              | Model to serve (sets `HF_MODEL_ID` + the worker's blaze prefix / MD-format env) |
| `--hf-model-id <id>`     | `deepseek-ai/DeepSeek-R1-0528`            | Explicit HF model id (overrides `--kimi`/`--deepseek`)                          |
| `--local-build`          | off                                       | Run this repo's `cpp_server/build` binary in the worker                         |
| `--etcd-image <img>`     | `quay.io/coreos/etcd:v3.5.13`             | etcd image                                                                      |
| `--worker-image <img>`   | `…/tt-media-inference-server-blaze:<tag>` | cpp_server image                                                                |
| `--frontend-image <img>` | `…/tt-dynamo-frontend:<tag>`              | Dynamo frontend image                                                           |
| `--device-ids <ids>`     | `10,11,14,15,18,19,22,23`                 | `DEVICE_IDS` env on the worker                                                  |


Fixed (not flags): network `dynamo-net`, container names `etcd` / `tt-cpp-worker`
/ `dynamo-frontend`, frontend host port `8080`, `LLM_DEVICE_BACKEND=pipeline_manager`,
`MODEL_NAME=tt-cpp-server`.

`HF_TOKEN` (for gated models) and perf knobs (`DYN_TOKENIZER`, `RAYON_NUM_THREADS`,
`DYN_RUNTIME_*`, `RUST_LOG`, `DYN_TX_TRACE`, `DYN_ENABLE_ANTHROPIC_API`) are read
from the calling shell's environment if set.

## What it does, step by step

1. **Network** — creates `dynamo-net` if missing; all three containers join it
  and resolve each other by name.
2. **etcd** — starts it (publishing `:2379`) and waits until
  `etcdctl endpoint health` passes.
3. **Worker** — starts cpp_server with etcd discovery
  (`DYNAMO_ETCD_ENDPOINTS=http://etcd:2379`), the model-specific env, and
   `--device /dev/tenstorrent` when the card is present. Waits up to 60s for it
   to register `v1/instances/…`; on failure it dumps the worker logs and exits.
   With `--local-build`, the entrypoint runs the bind-mounted binary
   (`LD_PRELOAD`ing the image's `libtt_llm_engine.so.0` only if present).
4. **Frontend** — starts it against the same etcd and maps `:8000 → :8080`.
  `MODEL_PATH` points at the tokenizer tree **baked into the frontend image**
   (same `fetch_tokenizers.sh` the worker uses), so no tokenizer bind-mount is
   needed.
5. **Logs** — `docker logs -f tt-cpp-worker`, blocking until you Ctrl+C.
6. **Teardown** — `trap cleanup EXIT INT TERM` removes the three containers
  (the `dynamo-net` network is left in place).

## Verify (from a second shell)

```bash
curl -s http://localhost:8080/v1/models | jq          # owned_by: "TT Inc"

curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-R1-0528","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}' | jq
```

Use the `id` returned by `/v1/models` as the `model` (it's the HF id, e.g.
`deepseek-ai/DeepSeek-R1-0528` or `moonshotai/Kimi-K2.6`). Add `"stream": true`
with `curl -N` for streaming.

Inspect what the worker registered:

```bash
docker exec etcd etcdctl get --prefix --keys-only v1/
# v1/instances/default/backend/generate/<hex>
# v1/mdc/default/backend/generate/<hex>
```

## Troubleshooting

- **Worker never registers** — usually a device or model-env problem; the worker
log dump on timeout shows the cause. On a box without the card, use
`--local-build` with a mock build to exercise the discovery + frontend wiring.
- `**/v1/models` is empty** — frontend and worker aren't talking to the same etcd.
Check `docker exec dynamo-frontend curl -s http://etcd:2379/version` and that
both use namespace `default` / component `backend` / endpoint `generate`.
- **Frontend 404s the model** (`huggingface.co/api/models//home/...`) — the
frontend image is missing the baked tokenizer tree at
`/home/container_app_user/app/server/cpp_server/tokenizers/<hf-model-id>/`.
Confirm with
`docker exec dynamo-frontend ls /home/container_app_user/app/server/cpp_server/tokenizers/<hf-model-id>/`.
- **Frontend dials the worker and fails** (`Invalid TCP address` / `Connection refused`) — the worker advertises its `dynamo-net` IPv4 in `transport.tcp`, so
the frontend must share that network. Leave `DYN_TCP_RPC_HOST` unset so the
worker auto-detects a numeric IP. Check:
`docker exec etcd etcdctl get --prefix v1/instances/ -w json | jq -r '.kvs[].value | @base64d | fromjson | .transport.tcp'`.

