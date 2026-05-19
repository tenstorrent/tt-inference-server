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
| `--hf-model-id <id>` | `meta-llama/Llama-3.1-8B-Instruct` | Tokenizer repo the frontend pulls at boot |
| `--hf-token <token>` | _(empty)_ | HuggingFace token for gated models |
| `--llm-device-backend <name>` | `mock_pipeline` | Backend the cpp_server runner selects |

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
   `DYN_TCP_RPC_HOST=$WORKER_NAME` ensures the address the worker writes
   into etcd's instance JSON resolves from inside the frontend container.
   The script then polls `etcdctl get --prefix v1/instances/` for up to
   60 seconds. If the worker exits before registration the script aborts
   and dumps the worker's recent logs.
4. **Frontend.** Starts the Dynamo frontend with `DYN_DISCOVERY_BACKEND=etcd`
   and `ETCD_ENDPOINTS=http://etcd:2379` — the same etcd, same key prefix.
   Maps the container's `:8000` to the host's `$FRONTEND_HOST_PORT`.
5. **Logs.** `docker logs -f` against the frontend container. The script
   blocks here until you interrupt it.
6. **Teardown.** A `trap cleanup EXIT INT TERM` removes the three
   containers on Ctrl+C (or any abnormal exit). The network is kept; if
   you want to remove it too, run `docker network rm dynamo-net` after
   teardown.

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
- **Frontend dials worker and gets `Connection refused`** — the
  `transport.tcp` baked into the instance JSON is wrong. The script sets
  `DYN_TCP_RPC_HOST=$WORKER_NAME`, which resolves on the shared docker
  network. If you change `NETWORK_NAME` to `host`, also change
  `DYN_TCP_RPC_HOST` to the host's external IP, not `127.0.0.1`.
