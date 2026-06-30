# Deploying Dynamo Frontend + cpp_server Worker

`deploy.sh` brings up an etcd-backed Dynamo stack on the `dynamo-net` Docker
network — **etcd** (discovery), **cpp_server** (worker), and
**dynamo-frontend** (HTTP gateway) — plus **Prometheus** and **Grafana** from
`tt-media-server/monitoring/`. With `--prefill-gateway`, it also starts a
C++ **PrefillGateway**, a managed prefill cpp_server worker, and configures the
Dynamo-registered cpp_server as decode so requests go through the gateway. It
then tails the decode worker's logs. Ctrl+C tears the managed containers down.

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
cd ../../dynamo_frontend
PROMETHEUS_HOST_PORT=9091 GRAFANA_HOST_PORT=3001 \
  ./deploy.sh --deepseek --local-build --llm-device-backend mock_pipeline
```

`--local-build` bind-mounts `<repo>/tt-media-server/cpp_server/build` over the
worker image and runs that binary — no `--cpp-server-dir` and no image rebuild.
`mock_pipeline` avoids the Blaze socket descriptor files that
`pipeline_manager` expects from a live Blaze runtime.

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
| `--llm-device-backend <name>` | `pipeline_manager`                   | `LLM_DEVICE_BACKEND` env on the worker                                          |
| `--prefill-gateway`      | off                                       | Start PrefillGateway and route decode prefill requests through it               |
| `--prefill-gateway-image <img>` | `tt-prefill-gateway:dev`          | PrefillGateway image; the default local image is built automatically if missing |
| `--prefill-gateway-prefill <host:port>` | none                       | External TCP prefill endpoint; repeatable and implies TCP transport             |
| `--prefill-gateway-prefill-bind <host:port>` | `0.0.0.0:7200`        | ZMQ prefill ROUTER bind endpoint                                                |
| `--no-monitoring`        | off                                       | Skip Prometheus + Grafana                                                       |


Fixed (not flags): network `dynamo-net`, container names `etcd` / `tt-cpp-worker`
/ `dynamo-frontend`, monitoring container names `dynamo-prometheus` /
`dynamo-grafana` / `dynamo-process-exporter`, optional gateway container name
`prefill-gateway`, optional managed prefill worker container name
`tt-cpp-prefill-worker`, frontend host port `8080`,
`LLM_DEVICE_BACKEND=pipeline_manager`, `MODEL_NAME=tt-cpp-server`.

`HF_TOKEN` (for gated models) and perf knobs (`ROUTER_MODE`, `DYN_TOKENIZER`,
`RAYON_NUM_THREADS`, `DYN_RUNTIME_*`, `RUST_LOG`, `DYN_TX_TRACE`,
`DYN_ENABLE_ANTHROPIC_API`) are read from the calling shell's environment if
set. `ROUTER_MODE` defaults to `kv` in this deployment so Dynamo frontend
timing metrics are emitted; override it if you need a different router.
`LLM_DEVICE_BACKEND` is also read from the environment and can be overridden
with `--llm-device-backend`.

Native prefill threshold policy is exposed through
`TT_NATIVE_PREFILL_POLICY=threshold` and
`TT_PREFILL_ON_DECODE_MAX_TOKENS` (defaulting to
`MAX_TOKENS_TO_PREFILL_ON_DECODE` or `1000`). These are currently a TT policy
contract for Dynamo router work: cpp_server also advertises the same threshold in
its MDC `runtime_config`. ai-dynamo 1.2.0.post1 does not consume the TT threshold
env directly yet.

Monitoring uses `tt-media-server/monitoring/docker-compose.yml`, attached to
`dynamo-net` via `TT_NET=dynamo-net`. By default Prometheus scrapes
`dynamo-frontend:8000` so Dynamo frontend metrics are available immediately.
Override `SERVER_TARGET`, `SERVER_SERVICE`, `GATEWAY_TARGET`, or
`GF_HOME_DASHBOARD` in the calling shell if you want a different scrape target
or dashboard. If another monitoring stack already uses the default ports, set
`PROMETHEUS_HOST_PORT` or `GRAFANA_HOST_PORT` before running `deploy.sh`.
When `--prefill-gateway` is enabled and `GATEWAY_TARGET` is not set,
Prometheus scrapes `prefill-gateway:9091`.

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
5. **PrefillGateway (optional)** — with `--prefill-gateway`, starts the
   gateway on `dynamo-net`. The default ZMQ mode binds `0.0.0.0:7200` for
   prefills and exposes metrics on container port `9091`; the script also
   starts one managed `LLM_MODE=prefill` cpp_server worker that connects to
   that bind endpoint. The Dynamo-registered worker runs as `LLM_MODE=decode`
   with `USE_PREFILL_GATEWAY=1` and
   `MAX_TOKENS_TO_PREFILL_ON_DECODE=0`, so decode requests route prefill work
   through the gateway.
6. **Monitoring** — starts Prometheus + Grafana, with Prometheus attached to
   `dynamo-net` and scraping the frontend's `/metrics`.
7. **Logs** — `docker logs -f tt-cpp-worker`, blocking until you Ctrl+C.
8. **Teardown** — `trap cleanup EXIT INT TERM` stops the monitoring compose
   stack and removes the Dynamo containers plus the optional PrefillGateway and
   managed prefill worker (the `dynamo-net` network is left in place).

## Verify (from a second shell)

```bash
curl -s http://localhost:8080/v1/models | jq          # owned_by: "TT Inc"

curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-R1-0528","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}' | jq
```

Open Prometheus at `http://localhost:${PROMETHEUS_HOST_PORT:-9090}` and Grafana
at `http://localhost:${GRAFANA_HOST_PORT:-3000}` (`admin` / `admin`).
Prometheus target health is also available directly:

```bash
curl -s "http://localhost:${PROMETHEUS_HOST_PORT:-9090}/api/v1/targets" | jq
```

Use the `id` returned by `/v1/models` as the `model` (it's the HF id, e.g.
`deepseek-ai/DeepSeek-R1-0528` or `moonshotai/Kimi-K2.6`). Add `"stream": true`
with `curl -N` for streaming.

Inspect what the worker registered:

```bash
docker exec etcd etcdctl get --prefix --keys-only v1/
# v1/instances/default/decode/generate/<hex>
# v1/mdc/default/decode/generate/<hex>
```

If you enabled `--prefill-gateway`, verify that Prometheus sees it:

```bash
docker exec dynamo-prometheus wget -qO- http://prefill-gateway:9091/metrics | head
docker exec dynamo-prometheus wget -qO- http://127.0.0.1:9090/api/v1/targets
```

You can also inspect the gateway health endpoint inside the Docker network:

```bash
docker exec dynamo-prometheus wget -qO- http://prefill-gateway:9092/health
```

## Troubleshooting

- **Worker never registers** — usually a device or model-env problem; the worker
log dump on timeout shows the cause. On a box without the card, use
`--local-build --llm-device-backend mock_pipeline` with a mock build to
exercise the discovery + frontend wiring.
- **Prometheus/Grafana port already allocated** — keep the existing monitoring
stack running and publish this deployment on alternate ports:
`PROMETHEUS_HOST_PORT=9091 GRAFANA_HOST_PORT=3001 ./deploy.sh --deepseek`.
- **PrefillGateway is up but unhealthy** — inspect
`docker logs prefill-gateway`, `docker logs tt-cpp-prefill-worker`, and
`docker exec dynamo-prometheus wget -qO- http://prefill-gateway:9092/health`.
Healthy gateway routing requires one decode connection and at least one
registered prefill worker.
- `**/v1/models` is empty** — frontend and worker aren't talking to the same etcd.
Check `docker exec dynamo-frontend curl -s http://etcd:2379/version` and that
both use namespace `default` / component `decode` / endpoint `generate`.
- **Frontend 404s the model** (`huggingface.co/api/models//home/...`) — the
frontend image is missing the baked tokenizer tree at
`/home/container_app_user/app/server/cpp_server/tokenizers/<hf-model-id>/`.
Confirm with
`docker exec dynamo-frontend ls /home/container_app_user/app/server/cpp_server/tokenizers/<hf-model-id>/`.
- **Frontend dials the worker and fails** (`Invalid TCP address` / `Connection refused`) — the worker advertises its `dynamo-net` IPv4 in `transport.tcp`, so
the frontend must share that network. Leave `DYN_TCP_RPC_HOST` unset so the
worker auto-detects a numeric IP. Check:
`docker exec etcd etcdctl get --prefix v1/instances/ -w json | jq -r '.kvs[].value | @base64d | fromjson | .transport.tcp'`.

