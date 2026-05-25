# Dynamo Frontend Docker

Lightweight Docker image for the NVIDIA Dynamo frontend, configured to use
**etcd v3** for backend discovery.

## Build

```bash
cd dynamo_frontend/
docker build -f Dockerfile.frontend -t dynamo-frontend .
```

## Run

```bash
docker run --network host \
  -e DYN_DISCOVERY_BACKEND=etcd \
  -e ETCD_ENDPOINTS=http://127.0.0.1:2379 \
  -e MODEL_NAME=my-model \
  -e HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  dynamo-frontend
```

`--network host` is recommended so the frontend can reach the cpp_server
worker's TCP transport (advertised in the instance JSON) without extra
port-mapping plumbing. A user-defined bridge network also works as long as
the worker, etcd, and the frontend are on the same network.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `mock-model` | Model name exposed via `/v1/models` |
| `MODEL_PATH` | `/app/model` | Path to model config/tokenizer files |
| `HF_MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace repo to download tokenizer from |
| `HF_TOKEN` | _(none)_ | HuggingFace token for gated models |
| `HTTP_PORT` | `8000` | HTTP listen port |
| `DYN_DISCOVERY_BACKEND` | `etcd` | Discovery backend (`etcd` or `file`) |
| `ETCD_ENDPOINTS` | `http://etcd:2379` | Comma-separated etcd v3 client URLs |
| `DYN_REQUEST_PLANE` | `tcp` | Request plane transport (`tcp` or `nats`) |
| `DYN_EVENT_PLANE` | `zmq` | Event plane transport (`zmq` or `nats`) |

## Notes

- **etcd is required**: both the frontend container and every cpp_server worker
  must point at the same etcd cluster (same `ETCD_ENDPOINTS`). Workers write
  `v1/instances/...` and `v1/mdc/...` keys under a lease; the frontend reads
  them to discover models and dial the worker's TCP transport.
- **Tokenizer download**: on first start, the entrypoint downloads
  `config.json`, `tokenizer.json`, and `tokenizer_config.json` from
  `HF_MODEL_ID`. Mount a volume at `$MODEL_PATH` to skip.
- **owned_by patch**: the build patches the Dynamo binary so `/v1/models`
  returns `"owned_by": "TT Inc"` instead of `"nvidia"`. Replacement must be
  exactly 6 characters.
- **File-store fallback**: set `DYN_DISCOVERY_BACKEND=file` and mount a
  shared directory (e.g. `-v /tmp/dynamo_store_kv:/tmp/dynamo_store_kv`) for
  single-host development. Not recommended for production.
