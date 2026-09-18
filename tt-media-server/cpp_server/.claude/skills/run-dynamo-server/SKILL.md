---
name: run-dynamo-server
description: Bring up the cpp_server LLM backend behind the NVIDIA Dynamo frontend (etcd + worker + frontend) for local dev and testing. Use when the user asks to run/start the server with Dynamo, deploy the Dynamo stack, launch the frontend + cpp worker, or reach the OpenAI-compatible endpoint that routes through Dynamo.
---

# Running cpp_server with Dynamo

## At a glance

| Step | Command (from repo root) |
|------|--------------------------|
| Build host binary | `cd tt-media-server/cpp_server && ./build.sh` |
| Launch stack (mock) | `cd dynamo_frontend && LLM_DEVICE_BACKEND=mock ./deploy.sh --deepseek --local-build --no-monitoring` |
| Resolve frontend addr | `docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' dynamo-frontend` |
| Smoke test | `curl -s "http://<FE_IP>:8000/v1/models"` |
| Tear down | `docker rm -f dynamo-frontend tt-cpp-worker etcd` |

## How the stack connects

```
client ──HTTP──▶ Dynamo frontend ──(discovers via)──▶ etcd ◀──(registers)── cpp_server worker
                        └──────── Dynamo TCP call-home protocol ────────────────┘
```

All three run as Docker containers on the `dynamo-net` network, orchestrated by
`dynamo_frontend/deploy.sh`. `--local-build` **bind-mounts `cpp_server/build/`**
into the worker image and runs the local binary — so iterate with `./build.sh`,
not a Docker image rebuild.

The recurring failure mode: forgetting `LLM_DEVICE_BACKEND=mock` (it defaults to
`pipeline_manager`, which claims **real** TT devices), or hitting `localhost:8080`
when the host port-publish is flaky instead of the container bridge IP.

## Quick start (mock backend — no TT hardware)

1. **Build** the host binary (default = Release, mock device backend, blaze OFF):
   `./build.sh` in `tt-media-server/cpp_server`.
2. **Launch** (mock!):
   `LLM_DEVICE_BACKEND=mock ./deploy.sh --deepseek --local-build --no-monitoring`
   from `dynamo_frontend`.
   - `--deepseek` (default) | `--kimi` selects the model.
   - `--no-monitoring` skips Prometheus/Grafana (and their 9090/3000 port checks).
   - `deploy.sh` stays foreground tailing the worker log; **Ctrl+C tears the whole stack down** (`trap cleanup`). Background it if you need the shell.
3. **Wait for readiness** — it blocks until the worker registers with etcd (≤60s), then logs `frontend on http://localhost:8080`.

## Reaching the frontend

Normally `http://localhost:8080` (the frontend publishes `-p 8080:8000`). If that
refuses connections (a docker iptables quirk), use the container bridge IP — it's
directly routable from the host on Linux:

```bash
FE=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' dynamo-frontend)
curl -s "http://${FE}:8000/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-R1-0528","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

## Real hardware

Direct (no Dynamo): `cpp_server/run_server.sh` (sets `pipeline_manager`,
`DEVICE_IDS`, `LD_PRELOAD` of the blaze engine). Dynamo on real HW: drop
`LLM_DEVICE_BACKEND=mock` and make `DEVICE_IDS` in `deploy.sh` match free devices.

## Troubleshooting

- **Worker "did not register within 60s"** → `deploy.sh` dumps worker logs and dies. Usually a stale/missing mounted `build/` binary (rebuild) or `pipeline_manager` claiming busy/absent devices (use `mock`).
- **Connection refused on :8080** → use the bridge-IP method above.
- **Logs:** `docker logs -f tt-cpp-worker`, `docker logs dynamo-frontend`.
- **Discovery:** `docker exec etcd etcdctl get --prefix --keys-only v1/instances/`.
- **Port in use** → a prior stack didn't tear down: `docker rm -f dynamo-frontend tt-cpp-worker etcd`.

## Related skills

`benchmark-dynamo` (load-test this stack) · `add-model-dynamo` (make `--<model>` work).
