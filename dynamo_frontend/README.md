# Dynamo Frontend Docker

Lightweight Docker image for the NVIDIA Dynamo frontend, configured to use
**etcd v3** for backend discovery.

## Build

```bash
cd /path/to/tt-inference-server
docker build -f dynamo_frontend/Dockerfile.frontend -t dynamo-frontend .
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
| `HTTP_PORT` | `8000` | HTTP listen port |
| `ROUTER_MODE` | `kv` | Dynamo router mode; `kv` activates cache-aware worker selection |
| `DYN_DISCOVERY_BACKEND` | `etcd` | Discovery backend (`etcd` or `file`) |
| `ETCD_ENDPOINTS` | `http://etcd:2379` | Comma-separated etcd v3 client URLs |
| `DYN_REQUEST_PLANE` | `tcp` | Request plane transport (`tcp` or `nats`) |
| `DYN_EVENT_PLANE` | `zmq` | Event plane transport (`zmq` or `nats`) |
| `DYN_PREFILL_ON_DECODE_MAX_TOKENS` | `1000` | Threshold matching cpp_server `MAX_TOKENS_TO_PREFILL_ON_DECODE`; logged/exported for Dynamo router support |

## Notes

- **etcd is required**: both the frontend container and every cpp_server worker
  must point at the same etcd cluster (same `ETCD_ENDPOINTS`). Workers write
  `v1/instances/...` and `v1/mdc/...` keys under a lease; the frontend reads
  them to discover models and dial the worker's TCP transport.
- **Tokenizers are baked in**: the image contains the tokenizer tree at the
  absolute path advertised by cpp_server MDCs. Build with repo-root context so
  `tt-media-server/cpp_server/scripts/fetch_tokenizers.sh` is available.
- **owned_by patch**: the build patches the Dynamo binary so `/v1/models`
  returns `"owned_by": "TT Inc"` instead of `"nvidia"`. Replacement must be
  exactly 6 characters.
- **File-store fallback**: set `DYN_DISCOVERY_BACKEND=file` and mount a
  shared directory (e.g. `-v /tmp/dynamo_store_kv:/tmp/dynamo_store_kv`) for
  single-host development. Not recommended for production.
