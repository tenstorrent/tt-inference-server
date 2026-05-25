# Deploying Dynamo Frontend + cpp_server Worker

`deploy.sh` orchestrates a three-container etcd-backed deployment: **etcd**
(discovery), **cpp_server** (worker), and **dynamo-frontend** (HTTP gateway).
On Ctrl+C the script tears everything down.

## Prerequisites

- A running Docker daemon and a `docker` CLI on `PATH`.
- The three images already present locally (or pullable):
  - **etcd**: e.g. `quay.io/coreos/etcd:v3.5.13` (off-the-shelf).
  - **cpp_server**: built from `tt-media-server/Dockerfile.blaze` (or one of
    the other `Dockerfile.*` variants).
  - **dynamo-frontend**: built from `dynamo_frontend/Dockerfile.frontend`.
- For the worker to actually drive a Tenstorrent card, `/dev/tenstorrent`
  must exist on the host. When it does, `deploy.sh` automatically passes
  `--device /dev/tenstorrent --cap-add=SYS_NICE`. When it does not, the
  worker still starts but cannot perform real inference — useful for
  smoke-testing the discovery + frontend wiring with `LLM_DEVICE_BACKEND=mock_pipeline`.

## Usage

```bash
./deploy.sh \
  --etcd-image <img> \
  --worker-image <img> \
  --frontend-image <img> \
  [options]
```

Example:

```bash
./deploy.sh \
  --etcd-image quay.io/coreos/etcd:v3.5.13 \
  --worker-image tt-media-server-cpp:blaze \
  --frontend-image dynamo-frontend
```

Show all options:

```bash
./deploy.sh --help
```

The script runs in the foreground, tailing the frontend's logs. Hit Ctrl+C
to stop and remove all three containers.

## Flags

Required:

| Flag | Purpose |
|---|---|
| `--etcd-image <img>` | etcd Docker image (e.g. `quay.io/coreos/etcd:v3.5.13`) |
| `--worker-image <img>` | cpp_server Docker image (e.g. `tt-media-server-cpp:blaze`) |
| `--frontend-image <img>` | Dynamo frontend Docker image (e.g. `dynamo-frontend`) |

Optional:

| Flag | Default | Purpose |
|---|---|---|
| `--network-name <name>` | `dynamo-net` | Docker network the three containers share |
| `--etcd-name <name>` | `etcd` | Container + DNS name for etcd |
| `--worker-name <name>` | `tt-cpp-worker` | Container + DNS name for the worker (also embedded into the instance JSON as `transport.tcp`) |
| `--frontend-name <name>` | `dynamo-frontend` | Container name for the frontend |
| `--frontend-host-port <port>` | `8080` | Host port mapped to the frontend's `:8000` |
| `--model-name <name>` | `tt-cpp-server` | `id` reported on `GET /v1/models` |
| `--hf-model-id <id>` | `deepseek-ai/DeepSeek-R1-0528` | Tokenizer repo the frontend pulls at boot |
| `--llm-device-backend <name>` | `mock_pipeline` | Backend the cpp_server runner selects |
| `--device-ids <ids>` | `(0)` | `DEVICE_IDS` env on the worker. Each top-level paren group becomes one parallel `LLMService` consumer thread. With the default `"(0)"` the worker only runs one in-flight generation at a time, regardless of how many requests the frontend pipelines. For benchmark concurrency on `mock_pipeline` use e.g. `"(0),(1),(2),(3),(4),(5),(6),(7)"`. |
| `--cpp-binary <path>` | _(unset)_ | Bind-mount a locally-built `tt_media_server_cpp` over the image's binary. Useful when the prebuilt CI image ships a stale binary. Path is resolved with `readlink -f`. |
| `--tokenizers-host-dir <dir>` | `<script_dir>/../tt-media-server/cpp_server/tokenizers` | Host directory that holds `<hf-model-id>/{config,tokenizer,tokenizer_config}.json`. Bind-mounted into the frontend at the path the worker advertises in the MDC. |
| `--skip-tokenizer-share` | _(off)_ | Skip the tokenizer bind-mount. Debug only — the frontend will be unable to load any model the worker advertises via the MDC. |

For gated HuggingFace models, export `HF_TOKEN` in the calling shell — it is
forwarded into the frontend container as an env var:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
./deploy.sh --etcd-image ... --worker-image ... --frontend-image ...
```

### Overriding the worker binary

If the prebuilt worker image's `tt_media_server_cpp` is stale (e.g. CI cache
drift), build it locally and bind-mount the fresh binary:

```bash
cd /localdev/ljovanovic/tt-inference-server/tt-media-server/cpp_server
./build.sh --blaze

cd /localdev/ljovanovic/tt-inference-server/dynamo_frontend
./deploy.sh \
  --etcd-image quay.io/coreos/etcd:v3.5.13 \
  --worker-image ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:<tag> \
  --frontend-image dynamo-frontend \
  --cpp-binary ../tt-media-server/cpp_server/build/tt_media_server_cpp
```

The binary is bind-mounted read-only over the image path
`/home/container_app_user/app/server/cpp_server/build/tt_media_server_cpp`,
so `run_cpp.sh` picks it up without any image rebuild.

## What the script does, step by step

1. **Network.** Creates `$NETWORK_NAME` (Docker bridge) if missing. All three
   containers join it, so `etcd`, `tt-cpp-worker`, and `dynamo-frontend`
   resolve each other by name.
2. **etcd.** Starts the etcd image with `--advertise-client-urls
   http://0.0.0.0:2379`. Polls `etcdctl endpoint health` for up to 30
   seconds and only continues once etcd reports `is healthy`. The host
   port `2379` is published so you can poke etcd with a local `etcdctl`
   if you need to.
3. **cpp_server worker.** Starts the worker with the matching
   `DYNAMO_DISCOVERY_BACKEND=etcd` and `DYNAMO_ETCD_ENDPOINTS=http://etcd:2379`.
   `DYN_TCP_RPC_HOST` is left **unset** so the worker auto-detects its own
   IPv4 address via `getifaddrs()` (see `detectAdvertiseHost()` in
   `dynamo_endpoint.cpp`). Dynamo's TCP dialer on the frontend parses
   `transport.tcp` with Rust's `SocketAddr::from_str`, which requires a
   numeric IP — overriding the auto-detection with the container name
   makes the frontend log `Invalid TCP address ...: invalid socket address
   syntax`. The script then polls `etcdctl get --prefix v1/instances/`
   for up to 60 seconds. If the worker exits before registration the
   script aborts and dumps the worker's recent logs.
4. **Tokenizer share.** The worker's MDC (Model Descriptor Card) advertises
   absolute paths inside its container (e.g.
   `/home/container_app_user/app/server/cpp_server/tokenizers/<model>/tokenizer.json`).
   Those paths do not exist in the frontend container, and Dynamo's loader
   falls back to treating the path as a HuggingFace repo id — yielding a 404
   such as `https://huggingface.co/api/models//home/container_app_user/...`.
   The script avoids this by bind-mounting the host source-checkout's
   `tt-media-server/cpp_server/tokenizers/` (overridable with
   `--tokenizers-host-dir`) **read-only** into the frontend at exactly
   `/home/container_app_user/app/server/cpp_server/tokenizers`. Before
   mounting, it verifies `<host_dir>/<hf-model-id>/{config,tokenizer,
   tokenizer_config}.json` are all present (the prebuilt worker image
   typically ships only the two `tokenizer*.json` files; the host source
   checkout has `config.json` too). The frontend's `MODEL_PATH` env var is
   also pointed at the mounted model dir so `entrypoint.sh` skips its own
   HuggingFace download. Pass `--skip-tokenizer-share` to disable this
   (debug only — discovery-driven loads will fail).
5. **Frontend.** Starts the Dynamo frontend with `DYN_DISCOVERY_BACKEND=etcd`
   and `ETCD_ENDPOINTS=http://etcd:2379` — the same etcd, same key prefix.
   Maps the container's `:8000` to the host's `$FRONTEND_HOST_PORT`.
6. **Logs.** `docker logs -f` against the frontend container. The script
   blocks here until you interrupt it.
7. **Teardown.** A `trap cleanup EXIT INT TERM` removes the three
   containers on Ctrl+C (or any abnormal exit). The network is kept; if
   you want to remove it too, run `docker network rm dynamo-net` after
   teardown. The bind-mounted host tokenizers dir is untouched.

## Testing the running stack

While `deploy.sh` is tailing the frontend, open a second shell:

```bash
curl -s http://localhost:8080/v1/models | jq
```

You should see a single model entry whose `owned_by` is `TT Inc`.

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tt-cpp-server",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16
  }' | jq
```

For streaming, add `"stream": true` and use `curl -N`.

You can also inspect what the worker registered:

```bash
docker exec etcd etcdctl get --prefix --keys-only v1/
```

Expected output:

```
v1/instances/default/backend/generate/<hex>
v1/mdc/default/backend/generate/<hex>
```

Both reappear every few seconds because the worker keep-alives its etcd
lease (see `discovery.cpp`'s `EtcdDiscoveryRegistration::keepAlive`).

## Failure modes

- **`docker: command not found`** — install the Docker CLI; see the top of
  this directory's `README.md`.
- **etcd container exits immediately** — check the image you passed. Some
  etcd images do not have `etcd` at `/usr/local/bin/etcd`; if so, override
  the entrypoint in the script or pass a different image.
- **Worker never registers** — most often a backend env-var mismatch. The
  log dump the script prints on timeout will usually show what failed
  (missing model files, device error, build mismatch). Re-run with
  `LLM_DEVICE_BACKEND=mock_pipeline` to isolate Dynamo-side problems from
  hardware-side problems.
- **Frontend `/v1/models` is empty** — frontend cannot see what the worker
  wrote. Confirm both point at the same etcd (`docker exec etcd etcdctl
  get --prefix v1/`), the same namespace/component/endpoint, and that
  `ETCD_ENDPOINTS` is reachable from inside the frontend container
  (`docker exec dynamo-frontend curl -s http://etcd:2379/version`).
- **Frontend dials worker and returns `Invalid TCP address ...: invalid
  socket address syntax`** — Dynamo's TCP dialer
  (`lib/runtime/src/pipeline/network/egress/tcp_client.rs::parse_address`)
  splits `transport.tcp` on the *first* `/` and parses the left half with
  Rust's `SocketAddr::from_str`, which only accepts a numeric `IP:port`.
  If you set `DYN_TCP_RPC_HOST` to a hostname (container name, DNS name)
  the auto-detected IP is replaced with a string the dialer rejects.
  Leave `DYN_TCP_RPC_HOST` unset so the worker auto-detects its IPv4 via
  `getifaddrs()`. Verify with
  `docker exec etcd etcdctl get --prefix v1/instances/ -w json | jq -r '.kvs[].value | @base64d | fromjson | .transport.tcp'`
  — you should see something like `172.18.0.X:37187/generate`.
- **Frontend dials worker and returns `Missing x-endpoint-path header for
  TCP request`** — the opposite mistake. Dynamo's egress TCP client
  derives the `x-endpoint-path` header from the trailing `/endpoint_name`
  segment of `transport.tcp`. If `tcp_address` is *only* `IP:port` (no
  slash), `parse_address` returns `(addr, None)` and the client refuses
  to send. The cpp_server should publish `IP:port/<endpoint>` — see
  `dc.tcp_address = ...` in `cpp_server/src/dynamo/dynamo_endpoint.cpp`.
- **Frontend dials worker and gets `Connection refused`** — the IP in
  `transport.tcp` is the worker's IP on `dynamo-net`, so the frontend must
  be on the same docker network. If you change `NETWORK_NAME` to `host`,
  set `DYN_TCP_RPC_HOST` to the host's external IPv4 explicitly (the
  auto-detect inside a host-networked container will be wrong).
- **Bench requests look strictly sequential even at high `--max-concurrency`,
  or — worse — bench freezes part-way through and the worker logs
  `[BlazeRunner] Output hang detected: no model output for 60001 ms`
  before self-terminating.** Three places to check, in order:
  1. **TCP read loop.** `DynamoServer::handle_connection` must off-thread
     `stream_response`; a stale CI binary that calls it synchronously
     will serialize every request on a connection (Dynamo's frontend
     pool only grows when an existing connection's 1024-slot channel
     buffer is full, so light load reuses one connection forever).
     Confirm with `ls -la <cpp-binary>` mtime after `--cpp-binary` is
     in play.
  2. **Single event loop.** The Dynamo path's `DynamoEndpoint` runs a
     pool of `trantor::EventLoopThreadPool` loops and round-robins
     incoming requests across them. The HTTP path
     (`llm_controller.cpp`) gets per-thread loops for free from
     drogon's IO-thread pool. A version that uses a single
     `EventLoopThread` for Dynamo will work fine at one in-flight
     request and hang under any real concurrency — the first slow
     session-create or stream callback blocks every subsequent
     request on the same loop until the worker's 60s watchdog fires.
     Look for `[DynamoEndpoint] Started N request-loop threads` in the
     worker logs to confirm the pool is up.
  3. **`LLMService` consumer count.** Even with the above two fixed,
     `LLMService` starts one consumer thread per entry in
     `DEVICE_IDS`. Default `"(0)"` means one in-flight generation no
     matter how many requests are queued. Pass
     `--device-ids "(0),(1),(2),(3),(4),(5),(6),(7)"` (or whatever
     width the backend supports) to widen the pipeline.
- **`ModelExpress download failed for model '/home/container_app_user/...'`
  / `404 Not Found for url ...huggingface.co/api/models//home/...`** — the
  worker's MDC paths are not visible inside the frontend container. Either
  the tokenizer-share step was skipped (`--skip-tokenizer-share`), the
  `--tokenizers-host-dir` host directory doesn't have all three of
  `config.json`, `tokenizer.json`, and `tokenizer_config.json` under
  `<hf-model-id>/`, or you changed `WORKER_TOKENIZER_DIR` inside
  `cpp_server` without updating `deploy.sh`. Confirm with
  `docker exec dynamo-frontend ls /home/container_app_user/app/server/cpp_server/tokenizers/<hf-model-id>/`.
  All three files must show up.
