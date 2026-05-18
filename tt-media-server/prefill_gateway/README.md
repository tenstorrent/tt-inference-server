<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# PrefillGateway

Stateless service that lets a single decode server fan requests out across N

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
ctest --test-dir build --output-on-failure
```

## Run the gateway

```bash
./build/prefill_gateway \
  --decode-port=7100 \
  --prefill=127.0.0.1:7200 \
  --prefill=127.0.0.1:7201
```

- `--decode-port` is the port the gateway *listens on* for the decode server.
- `--prefill=HOST:PORT` is a prefill the gateway *dials out to*. Repeat the
  flag for each prefill.

## End-to-end curl test (real cpp_server + gateway)

The gateway runs side-by-side with one decode `tt_media_server_cpp` and N
prefill `tt_media_server_cpp` instances. Two new env vars on `cpp_server`
flip the inter-server socket roles to talk through the gateway:

| Env var                  | Effect                                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `USE_PREFILL_GATEWAY=1`  | Decode dials gateway as CLIENT. Prefill listens on `SOCKET_PORT` for the gateway as SERVER.     |
| `PREFILL_SERVER_ID=...`  | Identity the prefill advertises in `PrefillRegistrationMessage`. Default: `<hostname>:<port>`.  |
| `PREFILL_MAX_IN_FLIGHT`  | Capacity hint sent to the gateway (0 = unlimited).                                              |

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

### Terminal A — gateway

```bash
cd tt-media-server/prefill_gateway
TT_LOG_LEVEL=info ./build/prefill_gateway \
  --decode-port=7100 \
  --prefill=127.0.0.1:7200 \
  --prefill=127.0.0.1:7201
```

### Terminal B — decode (talks to gateway on :7100)

```bash
cd tt-media-server/cpp_server
LLM_MODE=decode \
USE_PREFILL_GATEWAY=1 \
MAX_TOKENS_TO_PREFILL_ON_DECODE=0 \
SOCKET_HOST=127.0.0.1 SOCKET_PORT=7100 \
TT_LOG_LEVEL=debug \
./build/tt_media_server_cpp -p 8001
```

### Terminal C — prefill-0 (listens on :7200 for the gateway)

```bash
cd tt-media-server/cpp_server
TT_IPC_SHM_C2P=tt_ipc_c2p_8002 TT_IPC_SHM_P2C=tt_ipc_p2c_8002 \
PREFILL_TIMEOUT_MS=15000 TT_LOG_LEVEL=debug \
LLM_MODE=prefill LLM_DEVICE_BACKEND=mock \
USE_PREFILL_GATEWAY=1 \
SOCKET_HOST=0.0.0.0 SOCKET_PORT=7200 \
PREFILL_SERVER_ID=prefill-0 \
./build/tt_media_server_cpp -p 8002
```

### Terminal D — prefill-1 (listens on :7201 for the gateway)

```bash
cd tt-media-server/cpp_server
TT_IPC_SHM_C2P=tt_ipc_c2p_8003 TT_IPC_SHM_P2C=tt_ipc_p2c_8003 \
PREFILL_TIMEOUT_MS=15000 TT_LOG_LEVEL=debug \
LLM_MODE=prefill LLM_DEVICE_BACKEND=mock \
USE_PREFILL_GATEWAY=1 \
SOCKET_HOST=0.0.0.0 SOCKET_PORT=7201 \
PREFILL_SERVER_ID=prefill-1 \
./build/tt_media_server_cpp -p 8003
```

### Terminal E — drive a request through decode

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
