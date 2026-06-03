<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# PrefillGateway

Stateless service that lets a single decode server fan requests out across N
prefill servers. The gateway manages prefill liveness and routes each request
using sticky-hash → least-in-flight → round-robin order.

## Build

```bash
cd prefill_gateway
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/prefill_gateway` — the gateway binary.
- `build/prefill_selector_test`, `build/affinity_cache_test`,
  `build/prefill_registry_test`, `build/dispatcher_test` — unit tests
  (no I/O).
- `build/gateway_e2e_test` — integration test with real
  `SocketManager` instances over loopback (decode + gateway + 2 prefills in
  one process).

Run them all with:

```bash
SOCKET_TRANSPORT=tcp ctest --test-dir build --output-on-failure
```

To run against the ZMQ transport instead of TCP:

```bash
SOCKET_TRANSPORT=zmq ctest --test-dir build --output-on-failure
```

## Run the gateway

```bash
./build/prefill_gateway \
  --decode-port=7100 \
  --metrics-port=9091 \
  --health-port=9092 \
  --prefill=127.0.0.1:7200 \
  --prefill=127.0.0.1:7201
```

- `--decode-port` is the port the gateway *listens on* for the decode server.
- `--prefill=HOST:PORT` is a TCP prefill the gateway *dials out to*. Repeat the
  flag for each prefill.
- `--prefill-bind=HOST:PORT` is the ZMQ prefill-side ROUTER bind endpoint.
  ZMQ prefills dial this single endpoint and register themselves.
- `--prefill-stale-timeout-ms=MS` controls how long the ZMQ gateway waits
  without a prefill registration before marking that prefill down. Default:
  `3000`.
- `--request-timeout-ms=MS` controls how long the gateway lets a prefill
  request stay in-flight before failing it back to decode with
  `generated_text="timeout"`. Use `0` to disable. Default: `300000`.
- `--timeout-window-ms=MS`, `--timeout-threshold=N`, and
  `--timeout-cooldown-ms=MS` control repeated-timeout protection. By default,
  if a prefill times out `3` requests within `60000`ms, the gateway stops
  assigning new requests to it for `30000`ms. Use `--timeout-threshold=0` to
  disable this protection.
- `--metrics-port=PORT` exposes Prometheus metrics at `GET /metrics`. Default:
  `9091`. Use `--metrics-port=0` to disable the endpoint.
- `--health-port=PORT` — `GET /tt-liveness` and `GET /health` (JSON). Default:
  `0` (disabled). Must differ from `--metrics-port`.

## HTTP endpoints

| Flag | Default | Paths |
| ---- | ------- | ----- |
| `--metrics-port` | `9091` | `GET /metrics` (Prometheus) |
| `--health-port` | `0` | `GET /tt-liveness`, `GET /health` (same JSON) |

Health JSON includes `status`, `transport`, `registered_prefills`,
`healthy_prefills`, `accepting_prefills`, and `decode_connected`.

```bash
curl -s http://127.0.0.1:9091/metrics | head
curl -s http://127.0.0.1:9092/tt-liveness | jq .
curl -s http://127.0.0.1:9092/health | jq .
```

For Grafana, see [`monitoring/README.md`](../monitoring/README.md). Notable
series: `tt_gateway_routing_decisions_total`, `tt_prefill_*`, `tt_gateway_*`.

## End-to-end curl test (real cpp_server + gateway)

The gateway runs side-by-side with one decode `tt_media_server_cpp` and N
prefill `tt_media_server_cpp` instances. Two new env vars on `cpp_server`
flip the inter-server socket roles to talk through the gateway:

| Env var                         | Set on  | Effect                                                                                         |
| ------------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `USE_PREFILL_GATEWAY=1`         | both    | Decode dials gateway as CLIENT. TCP prefills listen for the gateway; ZMQ prefills dial the gateway's prefill ROUTER. |
| `SOCKET_TRANSPORT`              | all     | `tcp` (default) or `zmq`. Must be the same on all three processes.                             |
| `PREFILL_SERVER_ID=...`         | prefill | Identity advertised in `PrefillRegistrationMessage`. Default: `<hostname>:<port>`.             |
| `PREFILL_MAX_IN_FLIGHT=N`       | prefill | Capacity hint sent to the gateway (0 = unlimited).                                             |
| `MAX_TOKENS_TO_PREFILL_ON_DECODE=0` | decode  | Set to 0 to force all requests through the gateway. Default 1000 keeps short prompts local. |

The default (`USE_PREFILL_GATEWAY=0`) keeps the existing direct 1:1 wiring.

### Topology

```
   ┌──────────┐                ┌──────────────────┐               ┌───────────┐
   │  decode  │── CLIENT ────► │  PrefillGateway  │── CLIENT ────►│ prefill-0 │
   │ (8001)   │  port 7100     │                  │  port 7200    │ (8002)    │
   └──────────┘                │                  │               └───────────┘
                               │                  │
                               │                  │── CLIENT ────►┌───────────┐
                               │                  │  port 7201    │ prefill-1 │
                               └──────────────────┘               │ (8003)    │
                                                                  └───────────┘
```

The commands below use TCP (default), where the gateway dials each prefill.
For ZMQ, use the separate ZMQ commands below: the gateway binds one prefill
ROUTER endpoint and every prefill connects to it.

### Terminal A — gateway

```bash
cd tt-media-server/prefill_gateway
TT_LOG_LEVEL=info SOCKET_TRANSPORT=tcp ./build/prefill_gateway \
  --decode-port=7100 \
  --metrics-port=9091 \
  --health-port=9092 \
  --request-timeout-ms=2000 \
  --timeout-window-ms=10000 \
  --timeout-threshold=2 \
  --timeout-cooldown-ms=15000 \
  --prefill=127.0.0.1:7200 \
  --prefill=127.0.0.1:7201
```

### Terminal B — decode (talks to gateway on :7100)

```bash
cd tt-media-server/cpp_server
LLM_MODE=decode \
SOCKET_TRANSPORT=tcp \
USE_PREFILL_GATEWAY=1 \
MAX_TOKENS_TO_PREFILL_ON_DECODE=0 \
SOCKET_HOST=127.0.0.1 \
SOCKET_PORT=7100 \
TT_LOG_LEVEL=info \
./build/tt_media_server_cpp -p 8001
```

### Terminal C — prefill-0 (listens on :7200 for the gateway)

```bash
cd tt-media-server/cpp_server
TT_IPC_SHM_C2P=tt_ipc_c2p_8002 \
TT_IPC_SHM_P2C=tt_ipc_p2c_8002 \
PREFILL_TIMEOUT_MS=15000 \
TT_LOG_LEVEL=info \
LLM_MODE=prefill \
LLM_DEVICE_BACKEND=mock \
MOCK_PREFILL_SLEEP_MS=10000 \
SOCKET_TRANSPORT=tcp \
USE_PREFILL_GATEWAY=1 \
SOCKET_HOST=0.0.0.0 \
SOCKET_PORT=7200 \
PREFILL_SERVER_ID=prefill-0 \
./build/tt_media_server_cpp -p 8002
```

### Terminal D — prefill-1 (listens on :7201 for the gateway)

```bash
cd tt-media-server/cpp_server
TT_IPC_SHM_C2P=tt_ipc_c2p_8003 \
TT_IPC_SHM_P2C=tt_ipc_p2c_8003 \
PREFILL_TIMEOUT_MS=15000 \
TT_LOG_LEVEL=info \
LLM_MODE=prefill \
LLM_DEVICE_BACKEND=mock \
SOCKET_TRANSPORT=tcp \
USE_PREFILL_GATEWAY=1 \
SOCKET_HOST=0.0.0.0 \
SOCKET_PORT=7201 \
PREFILL_SERVER_ID=prefill-1 \
./build/tt_media_server_cpp -p 8003
```

### ZMQ variant

For ZMQ, the decode side still connects to the gateway on `:7100`, but prefills
connect to one gateway ROUTER endpoint on `:7200`.

#### Terminal A — gateway

```bash
cd tt-media-server/prefill_gateway
TT_LOG_LEVEL=info SOCKET_TRANSPORT=zmq ./build/prefill_gateway \
  --decode-port=7100 \
  --metrics-port=9091 \
  --health-port=9092 \
  --request-timeout-ms=2000 \
  --timeout-window-ms=10000 \
  --timeout-threshold=2 \
  --timeout-cooldown-ms=15000 \
  --prefill-bind=127.0.0.1:7200
```

#### Terminal B — decode

```bash
cd tt-media-server/cpp_server
LLM_MODE=decode \
SOCKET_TRANSPORT=zmq \
USE_PREFILL_GATEWAY=1 \
MAX_TOKENS_TO_PREFILL_ON_DECODE=0 \
SOCKET_HOST=127.0.0.1 \
SOCKET_PORT=7100 \
TT_LOG_LEVEL=info \
./build/tt_media_server_cpp -p 8001
```

#### Terminal C — prefill-0

```bash
cd tt-media-server/cpp_server
TT_IPC_SHM_C2P=tt_ipc_c2p_8002 \
TT_IPC_SHM_P2C=tt_ipc_p2c_8002 \
PREFILL_TIMEOUT_MS=15000 \
TT_LOG_LEVEL=info \
LLM_MODE=prefill \
LLM_DEVICE_BACKEND=mock \
MOCK_PREFILL_SLEEP_MS=10000 \
SOCKET_TRANSPORT=zmq \
USE_PREFILL_GATEWAY=1 \
SOCKET_HOST=127.0.0.1 \
SOCKET_PORT=7200 \
PREFILL_SERVER_ID=prefill-0 \
./build/tt_media_server_cpp -p 8002
```

#### Terminal D — prefill-1

```bash
cd tt-media-server/cpp_server
TT_IPC_SHM_C2P=tt_ipc_c2p_8003 \
TT_IPC_SHM_P2C=tt_ipc_p2c_8003 \
PREFILL_TIMEOUT_MS=15000 \
TT_LOG_LEVEL=info \
LLM_MODE=prefill \
LLM_DEVICE_BACKEND=mock \
SOCKET_TRANSPORT=zmq \
USE_PREFILL_GATEWAY=1 \
SOCKET_HOST=127.0.0.1 \
SOCKET_PORT=7200 \
PREFILL_SERVER_ID=prefill-1 \
./build/tt_media_server_cpp -p 8003
```

### Terminal E — drive a request through decode

After prefills register, `curl -s http://127.0.0.1:9092/tt-liveness | jq .`
should show `healthy_prefills: 2` and `decode_connected: true`.

```bash
curl -N http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -d '{
    "model":"deepseek-r1",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":1,
    "stream":true,
    "skip_special_tokens":false
  }'
```

Expected gateway log lines for a successful request:

```
[InterServerService] Sent PrefillRegistration: id='prefill-0' max_in_flight=...
[InterServerService] Sent PrefillRegistration: id='prefill-1' max_in_flight=...
[Gateway] Running. Send SIGINT/SIGTERM to stop.
... PrefillRequest received from decode, dispatched to prefill-X ...
... PrefillResult forwarded back to decode ...
```

If a prefill goes down mid-request, the gateway emits a
`PrefillResultMessage` with `error=true` and `generated_text="prefill_down"`
to the decode for any task that was on that prefill, plus evicts the
affected affinity entries.

If a prefill accepts a request but does not return a result before
`--request-timeout-ms`, the gateway emits a `PrefillResultMessage` with
`error=true` and `generated_text="timeout"`, then drops any later result for
that task. The gateway also sends a best-effort `CancelPrefillMessage` to the
assigned prefill so it can stop the slow request if possible. Repeated timeouts
temporarily make that prefill ineligible for new tasks according to
`--timeout-window-ms`, `--timeout-threshold`, and `--timeout-cooldown-ms`.

---

## Direct prefill/decode split (no gateway)

Without the gateway the decode server is the socket **server** and the prefill
is the socket **client** that dials into it. Two terminals suffice.

### TCP

#### Terminal A — decode (listens on :9000)

```bash
cd tt-media-server/cpp_server
LLM_MODE=decode \
SOCKET_TRANSPORT=tcp \
MAX_TOKENS_TO_PREFILL_ON_DECODE=0 \
SOCKET_HOST=0.0.0.0 \
SOCKET_PORT=9000 \
TT_LOG_LEVEL=info \
./build/tt_media_server_cpp -p 8001
```

#### Terminal B — prefill (connects to decode on :9000)

```bash
cd tt-media-server/cpp_server
LLM_MODE=prefill \
SOCKET_TRANSPORT=tcp \
SOCKET_HOST=127.0.0.1 \
SOCKET_PORT=9000 \
LLM_DEVICE_BACKEND=mock \
TT_LOG_LEVEL=info \
./build/tt_media_server_cpp -p 8002
```

For ZMQ — just swap `SOCKET_TRANSPORT=tcp` → `SOCKET_TRANSPORT=zmq` on
**both** processes.

#### Terminal C — drive a request

```bash
curl -N http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -d '{
    "model":"deepseek-r1",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":1,
    "stream":true,
    "skip_special_tokens":false
  }'
```
