# Dynamo Frontend Docker

Lightweight Docker image for the NVIDIA Dynamo frontend

## Build

```bash
cd docker/
docker build -f Dockerfile.frontend -t dynamo-frontend .
```

## Run

```bash
docker run --network host \
  -v /tmp/dynamo_store_kv:/tmp/dynamo_store_kv \
  -e MODEL_NAME=my-model \
  -e HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  dynamo-frontend
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `mock-model` | Model name exposed via `/v1/models` |
| `MODEL_PATH` | `/app/model` | Path to model config/tokenizer files |
| `HF_MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace repo to download tokenizer from |
| `HF_TOKEN` | _(none)_ | HuggingFace token for gated models |
| `HTTP_PORT` | `8000` | HTTP listen port |
| `DYN_DISCOVERY_BACKEND` | `file` | Discovery backend (`file` or `etcd`) |
| `DYN_REQUEST_PLANE` | `tcp` | Request plane transport (`tcp` or `nats`) |
| `DYN_EVENT_PLANE` | `zmq` | Event plane transport (`zmq` or `nats`) |

## Notes

- **Discovery volume**: Mount `-v /tmp/dynamo_store_kv:/tmp/dynamo_store_kv` so the frontend can discover backends running on the host.
- **Network mode**: Use `--network host` so the frontend can reach backends via TCP on the host network.
- **Tokenizer download**: On first startup, the entrypoint downloads `config.json`, `tokenizer.json`, and `tokenizer_config.json` from HuggingFace. To skip this, mount a volume with these files at `$MODEL_PATH`.
- **owned_by patch**: The build patches the Dynamo binary so `/v1/models` returns `"owned_by": "TT Inc"` instead of `"nvidia"`. Limited to 6 characters.
